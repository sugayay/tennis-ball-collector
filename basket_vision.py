"""One downward RGB camera per chassis; YOLO-only basket count observations.

No ball entity, depth image, segmentation, or simulator ball positions enter the
sensor. The calibrated basket plane is a geometric assumption, not measured Z.
Only evaluate a lowered/stationary shovel; its arm can occlude the basket.
"""
from __future__ import annotations

import time
import numpy as np
from PIL import Image, ImageDraw

from onboard_vision import rotation_of
from scene import as_array


class BasketCamera:
    def __init__(self, scene, robot, config, initial_robot_position):
        self.robot, self.config = robot, config
        self.offset=np.asarray(config['position_in_robot_m'],dtype=float)
        self.lookat=np.asarray(config['lookat_in_robot_m'],dtype=float)
        self.up=np.asarray(config['up_in_robot'],dtype=float)
        p=np.asarray(initial_robot_position,dtype=float)
        self.camera=scene.add_camera(res=tuple(config['resolution']),pos=p+self.offset,
            lookat=p+self.lookat,up=self.up,fov=config['vertical_fov_degrees'],
            near=.01,far=10,GUI=False)

    def capture(self):
        rotation=rotation_of(self.robot);position=as_array(self.robot.get_pos())
        self.camera.set_pose(pos=position+rotation@self.offset,
            lookat=position+rotation@self.lookat,up=rotation@self.up)
        return np.asarray(self.camera.render(antialiasing=True)[0],dtype=np.uint8)

    def point_on_basket_plane(self,u,v):
        k=np.asarray(self.camera.intrinsics);transform=np.asarray(self.camera.transform)
        ray=np.array([(u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-1.])
        rotation=rotation_of(self.robot);position=as_array(self.robot.get_pos())
        origin=rotation.T@(transform[:3,3]-position)
        direction=rotation.T@(transform[:3,:3]@ray)
        if direction[2]>=-1e-8:return None
        distance=(self.config['ball_center_plane_in_robot_m']-origin[2])/direction[2]
        return (origin+distance*direction).tolist() if distance>0 else None


class BasketDetector:
    def __init__(self, ball_detector, config):
        # Share the existing YOLO instance/weights with the forward camera.
        self.detector,self.config=ball_detector,config

    def detect(self,rgb,camera):
        started=time.perf_counter()
        result=self.detector.model.predict(Image.fromarray(rgb),
            device=self.detector.config['device'],imgsz=self.detector.config['image_size'],
            conf=self.config['minimum_confidence'],classes=self.detector.classes,
            verbose=False,save=False)[0]
        accepted,rejected=[],[]
        xmin,ymin,xmax,ymax=self.config['center_bounds_in_robot_m']
        height,width=rgb.shape[:2]
        for box,confidence in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
            u,v=(box[:2]+box[2:])/2
            local=camera.point_on_basket_plane(float(u),float(v))
            inside=local is not None and xmin<=local[0]<=xmax and ymin<=local[1]<=ymax
            clipped=box[0]<2 or box[1]<2 or box[2]>width-2 or box[3]>height-2
            item=dict(bbox_xyxy=box.astype(float).tolist(),confidence=float(confidence),
                assumed_basket_plane_position_m=local,inside_basket_region=bool(inside),clipped=bool(clipped))
            (accepted if inside and not clipped else rejected).append(item)
        annotated=result.plot()[:,:,::-1].copy()
        image=Image.fromarray(annotated);draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,width,24),fill='#10232c')
        draw.text((8,6),f'Basket RGB / observed count: {len(accepted)}',fill='#d8f6bb')
        return dict(observed_count=len(accepted),observations=accepted,rejected=rejected,
            elapsed_seconds=time.perf_counter()-started,
            valid=bool(np.isfinite(rgb).all() and rgb.std()>10 and len(accepted)<=self.config['maximum_count']),
            height_measured=False),np.asarray(image)
