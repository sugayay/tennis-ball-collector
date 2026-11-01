"""Whole-frame and tiled YOLO on fixed-camera RGB; no target-state inputs."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from camera_geometry import pixel_to_plane,world_to_pixel


def ball_search_tiles(tiles,zones,calibration,radius,margin_m=.75):
    """Keep full-size ball tiles intersecting configured trial areas.

    Uses declared operational geometry, never a detected or test ball location.
    Person detection always retains the entire original image.
    """
    k=np.asarray(calibration['intrinsics']);transform=np.asarray(calibration['transform'])
    polygons=[]
    for xmin,ymin,xmax,ymax in zones:
        x1,y1,x2,y2=xmin-margin_m,ymin-margin_m,xmax+margin_m,ymax+margin_m
        corners=[world_to_pixel([x,y,radius],k,transform) for x,y in ((x1,y1),(x2,y1),(x2,y2),(x1,y2))]
        if any(p is None for p in corners):return tiles
        corners=np.asarray(corners)
        if not np.isfinite(corners).all():return tiles
        edges=np.roll(corners,-1,axis=0)-corners
        axes=np.vstack((np.eye(2),np.column_stack((-edges[:,1],edges[:,0]))))
        projected=corners@axes.T
        polygons.append((axes,projected.min(axis=0),projected.max(axis=0)))
    kept=[]
    for tile in tiles:
        centre=(np.array(tile[:2])+tile[2:])/2
        half=(np.array(tile[2:])-tile[:2])/2+20
        # Separating-axis test against the projected court quadrilateral.
        # The same 20 px margin remains, but empty corners of its bounding
        # rectangle no longer consume inference tiles. Every zone point is kept.
        for axes,low,high in polygons:
            centre_projection=axes@centre;extent=np.abs(axes)@half
            if np.all(centre_projection+extent>=low) and np.all(centre_projection-extent<=high):
                kept.append(tile);break
    return kept


def tile_starts(length, size, overlap):
    if length <= size:
        return [0]
    step = max(1, round(size*(1-overlap)))
    return sorted(set(list(range(0,length-size+1,step))+[length-size]))


def tile_owns_pixel(tile, pixel, image_size, margin):
    """Use overlap interiors for people while retaining physical image edges.

    An artificial crop boundary can cut feet off or make a robot fragment look
    like a person. Every image pixel remains covered by an interior tile.
    No robot positions, assumed human heights, or aspect-ratio masks are used.
    """
    x1,y1,x2,y2=tile; x,y=pixel; width,height=image_size
    return ((x1==0 or x>=x1+margin) and (x2==width or x<x2-margin)
            and (y1==0 or y>=y1+margin) and (y2==height or y<y2-margin))


def suppress_overlapping_boxes(items, threshold):
    remaining = sorted(items,key=lambda item:item['confidence'],reverse=True)
    result = []
    while remaining:
        best = remaining.pop(0)
        result.append(best)
        a = np.array(best['bbox_xyxy'])
        survivors = []
        for item in remaining:
            b = np.array(item['bbox_xyxy'])
            width,height = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
            intersection = width*height
            area_a = np.prod(np.maximum(0,a[2:]-a[:2]))
            area_b = np.prod(np.maximum(0,b[2:]-b[:2]))
            iou = intersection/max(1e-9,area_a+area_b-intersection)
            if iou <= threshold:
                survivors.append(item)
        remaining = survivors
    return result


class FixedBallDetector:
    def __init__(self,weights,vision_config,fixed_config):
        from ultralytics import YOLO,settings
        settings.update({'sync':False})
        self.model = YOLO(str(weights))
        self.vision = vision_config
        self.config = fixed_config
        self.classes = [int(key) for key,name in self.model.names.items()
                        if name.lower() in ('sports ball','tennis ball','ball')]
        if not self.classes:
            raise ValueError('The model has no ball class.')

        self.person_model = self.model
        self.person_weights_path = Path(weights)
        if fixed_config.get('person_safety_enabled') and fixed_config.get('person_model'):
            person_weights=Path(weights).parent/fixed_config['person_model']
            if not person_weights.is_file():
                raise FileNotFoundError(f'Person model must already be installed: {person_weights}')
            self.person_model=YOLO(str(person_weights))
            self.person_weights_path=person_weights
        self.person_classes = [int(k) for k,v in self.person_model.names.items() if v.lower()=='person'] if fixed_config.get('person_safety_enabled') else []
        if fixed_config.get('person_safety_enabled') and not self.person_classes:
            raise ValueError('The safety detector requires a person class.')

    def detect(self,rgb,calibration,ball_radius,*,tiled,include_balls=True,include_people=True):
        """All tiles are uniform; their selection does not use known ball pixels."""
        started = time.perf_counter()
        image = Image.fromarray(rgb)
        width,height = image.size
        person_classes=self.person_classes if include_people else []
        overlap = self.config['tile_overlap_fraction']
        separate_people = bool(person_classes and (not include_balls or self.person_model is not self.model or
            (tiled and self.config.get('person_tile_size_pixels',self.config['tile_size_pixels'])!=self.config['tile_size_pixels'])))
        if tiled:
            size = self.config['tile_size_pixels']
            boxes = [(x,y,min(width,x+size),min(height,y+size))
                     for y in tile_starts(height,size,overlap)
                     for x in tile_starts(width,size,overlap)]
            inference_size = self.config['tile_inference_size_pixels']
        else:
            boxes = [(0,0,width,height)]
            inference_size = self.vision['image_size']
        if not include_balls:boxes=[]
        original_ball_tile_count=len(boxes)
        if tiled and self.config.get('ball_search_zones_m'):
            if person_classes and not separate_people:
                raise ValueError('Scoped ball tiles require independent full-image person detection')
            boxes=ball_search_tiles(boxes,self.config['ball_search_zones_m'],calibration,ball_radius)
        raw,raw_people = [],[]
        discarded_boundary_people = 0
        person_margin = (size-round(size*(1-overlap)))/2 if tiled else 0
        person_tile_count = len(boxes) if person_classes else 0
        batch_size = self.config['inference_batch_size']
        for offset in range(0,len(boxes),batch_size):
            batch = boxes[offset:offset+batch_size]
            results = self.model.predict(
                [image.crop(box) for box in batch],
                device=self.vision['device'],imgsz=inference_size,
                conf=self.vision['confidence'],classes=self.classes+([] if separate_people else person_classes),
                verbose=False,save=False,
            )
            for tile,result in zip(batch,results):
                for bounds,confidence,category in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
                    bounds = bounds.astype(float)+np.array([tile[0],tile[1],tile[0],tile[1]])
                    item={'bbox_xyxy':bounds.tolist(),'confidence':float(confidence)}
                    if int(category) in self.classes:
                        raw.append(item)
                    elif int(category) in person_classes and confidence>=self.config.get('person_confidence',.35):
                        if tile_owns_pixel(tile,(bounds[:2]+bounds[2:])/2,(width,height),person_margin):
                            raw_people.append(item)
                        else:
                            discarded_boundary_people += 1
        if separate_people:
            # People need broader visual context than tiny distant tennis balls.
            # Retain the original high-resolution ball inference unchanged.
            person_size=self.config.get('person_tile_size_pixels',self.config['tile_size_pixels'])
            person_margin=(person_size-round(person_size*(1-overlap)))/2 if tiled else 0
            person_tiles=[(x,y,min(width,x+person_size),min(height,y+person_size))
                          for y in tile_starts(height,person_size,overlap)
                          for x in tile_starts(width,person_size,overlap)] if tiled else [(0,0,width,height)]
            person_tile_count=len(person_tiles)
            for offset in range(0,len(person_tiles),batch_size):
                batch=person_tiles[offset:offset+batch_size]
                results=self.person_model.predict([image.crop(box) for box in batch],
                    device=self.vision['device'],imgsz=self.config['person_tile_inference_size_pixels'] if tiled else self.vision['image_size'],
                    conf=self.config.get('person_confidence',.35),classes=person_classes,verbose=False,save=False)
                for tile,result in zip(batch,results):
                    for bounds,confidence in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                        bounds=bounds.astype(float)+np.array([tile[0],tile[1],tile[0],tile[1]])
                        item={'bbox_xyxy':bounds.tolist(),'confidence':float(confidence)}
                        if tile_owns_pixel(tile,(bounds[:2]+bounds[2:])/2,(width,height),person_margin):
                            raw_people.append(item)
                        else:
                            discarded_boundary_people+=1
        detections = suppress_overlapping_boxes(raw,self.config['box_merge_iou'])
        intrinsic = np.array(calibration['intrinsics'])
        transform = np.array(calibration['transform'])
        observations = []
        for item in detections:
            bounds = np.array(item['bbox_xyxy'])
            position = pixel_to_plane((bounds[:2]+bounds[2:])/2,intrinsic,transform,ball_radius)
            if position is not None:
                observations.append({**item,'position_m':position.tolist(),
                                     'camera_id':calibration['id']})
        persons,unlocalized = localize_people(suppress_overlapping_boxes(raw_people,self.config['box_merge_iou']),calibration)
        return {'persons':persons,'unlocalized_person_count':unlocalized,'raw_person_detection_count':len(raw_people)+discarded_boundary_people,
                'discarded_person_tile_boundary_count':discarded_boundary_people,'person_tile_interior_margin_pixels':person_margin,
                'person_tile_count':person_tile_count,'person_context_separate_from_balls':separate_people,
                'person_model_separate_from_balls':self.person_model is not self.model,
                'method':'uniform_tiles' if tiled else 'whole_frame',
                'tile_count':len(boxes),'original_ball_tile_count':original_ball_tile_count,'raw_detection_count':len(raw),
                'observations':observations,'elapsed_seconds':time.perf_counter()-started}


def localize_people(boxes,calibration):
    """Project detected feet to the court plane; never use person ground truth."""
    intrinsic=np.asarray(calibration['intrinsics']);transform=np.asarray(calibration['transform'])
    persons=[];unlocalized=0
    for item in boxes:
        x1,y1,x2,y2=item['bbox_xyxy']
        foot=np.array([(x1+x2)/2,y2])
        point=pixel_to_plane(foot,intrinsic,transform,0.)
        if point is None or not np.isfinite(point).all():
            unlocalized+=1
        else:
            persons.append({**item,'foot_pixel':foot.tolist(),'position_m':point.tolist(),'camera_id':calibration['id']})
    return persons,unlocalized
