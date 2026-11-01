"""Refine unique, simultaneous RGB person pairs using calibrated head rays.

No person dimensions, scene entities or ground-truth positions are inputs.
Unmatched/ambiguous detections and unknown-position counts are preserved.
"""
import math
import numpy as np


def head_ray(person, calibration):
    box=np.asarray(person.get('bbox_xyxy',[]),dtype=float)
    k=np.asarray(calibration.get('intrinsics',[]),dtype=float)
    transform=np.asarray(calibration.get('transform',[]),dtype=float)
    if (box.shape!=(4,) or k.shape!=(3,3) or transform.shape!=(4,4)
            or not all(np.isfinite(a).all() for a in (box,k,transform))
            or k[0,0]<=0 or k[1,1]<=0 or box[2]<=box[0] or box[3]<=box[1]):
        return None
    u,v=(box[0]+box[2])/2,box[1]
    direction=transform[:3,:3]@np.array([(u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-1.])
    length=float(np.linalg.norm(direction))
    if length<1e-8:return None
    return transform[:3,3],direction/length


def triangulate_pair(first,second,first_calibration,second_calibration):
    rays=[head_ray(first,first_calibration),head_ray(second,second_calibration)]
    if any(ray is None for ray in rays):return None
    (oa,da),(ob,db)=rays
    matrix=np.column_stack((da,-db))
    if np.linalg.cond(matrix)>20:return None
    distances=np.linalg.lstsq(matrix,ob-oa,rcond=None)[0]
    if min(distances)<=0:return None
    pa,pb=oa+distances[0]*da,ob+distances[1]*db
    gap=float(np.linalg.norm(pa-pb));head=(pa+pb)/2
    # Broad plausibility bounds reject a bad correspondence; they do not assign
    # a fixed person height or remove detections outside this range.
    if gap>.35 or not .75<=head[2]<=2.5:return None
    for person in (first,second):
        floor=np.asarray(person.get('position_m',[]),dtype=float)
        if floor.shape!=(3,) or not np.isfinite(floor).all():return None
        if math.dist(head[:2],floor[:2])>1.5:return None
    return dict(position_m=[float(head[0]),float(head[1]),0.],
                head_ray_gap_m=gap,triangulated_head_height_m=float(head[2]))


def refine_person_views(views):
    """Receive (RGB, calibration, detections) tuples from one capture batch.

    RGB is passed through, never inspected. Call only for simultaneous views;
    the caller retains the original capture timestamp for safety freshness.
    """
    refined=[(rgb,calibration,{**detected,'persons':[
        {**person,'floor_projection_position_m':list(person['position_m']),
         'localization_method':'floor_projection'} for person in detected['persons']]})
        for rgb,calibration,detected in views]
    if len(refined)!=2 or refined[0][1]['id']==refined[1][1]['id']:return refined
    (_,ca,a),(_,cb,b)=refined
    candidates=[]
    for i,first in enumerate(a['persons']):
        for j,second in enumerate(b['persons']):
            estimate=triangulate_pair(first,second,ca,cb)
            if estimate is not None:candidates.append((i,j,estimate))
    for i,j,estimate in candidates:
        if (sum(ia==i for ia,_,_ in candidates)!=1
                or sum(jb==j for _,jb,_ in candidates)!=1):continue
        for person,other in ((a['persons'][i],cb),(b['persons'][j],ca)):
            person.update(**estimate,localization_method='dual_view_head_triangulation',
                          paired_camera_id=other['id'])
    return refined
