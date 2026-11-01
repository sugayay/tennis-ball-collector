"""Inspect two fixed RGB cameras and YOLO on a two-robot tennis scene.

This is an observation test only. Ball simulator state is written only to a
separate evaluation file and is never passed to the image detector or merger.
No moving area, dispatch policy, or two-robot driving is introduced.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from fixed_camera_visuals import system_font_path

from camera_geometry import merge_ball_estimates,world_to_pixel
from fixed_detection import FixedBallDetector

ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def capture(config,fleet,vision,fixed,output,backend):
    import genesis as gs
    from fleet_scene import build_fleet,check_fleet,hold_fleet
    from scene import as_array,save_rgb
    gs.init(backend=getattr(gs,backend),seed=13,logging_level='warning')
    try:
        scene,robots,balls,overview,top,detail,onboard = build_fleet(config,fleet,vision)
        cameras = [scene.add_camera(
            res=tuple(spec['resolution']),pos=tuple(spec['position_m']),
            lookat=tuple(spec['lookat_m']),up=(0,0,1),
            fov=spec['vertical_fov_degrees'],near=.05,far=200,GUI=False,
        ) for spec in fixed['cameras']]
        scene.build()
        hold_fleet(robots)
        for _ in range(120):
            scene.step()
        checks = check_fleet(robots,[spec['position_m'] for spec in fleet['robots']])
        save_rgb(top,output/'fleet_top.png')
        calibration = []
        for spec,camera in zip(fixed['cameras'],cameras):
            print(f"Rendering fixed camera {spec['id']} at {spec['resolution']}",flush=True)
            save_rgb(camera,output/f"camera_{spec['id']}_rgb.png")
            calibration.append({**spec,'intrinsics':np.asarray(camera.intrinsics).tolist(),
                                'transform':np.asarray(camera.transform).tolist()})
        for spec,camera in zip(fleet['robots'],onboard):
            Image.fromarray(camera.capture()).save(output/f"robot_{spec['id']}_onboard.png")
        (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
        # Evaluation only: the detector below does not receive this file or its data.
        reference = {'ball_positions_m':[as_array(ball.get_pos()).tolist() for ball in balls],
                     'robots':checks,'captured_at_utc':datetime.now(timezone.utc).isoformat()}
        (output/'evaluation_reference.json').write_text(json.dumps(reference,indent=2)+'\n')
        if not all(item['stable'] for item in checks):
            raise RuntimeError('A robot was unstable during the stationary camera test.')
    finally:
        gs.destroy()


def evaluate(observations,reference,threshold):
    """One-to-one matching; missed balls and false positives stay in the report."""
    candidates = sorted((float(np.linalg.norm(np.array(item['position_m'][:2])-np.array(point[:2]))),i,j)
                        for i,item in enumerate(observations) for j,point in enumerate(reference))
    used_detections,used_balls = set(),set()
    matches = []
    for error,i,j in candidates:
        if error > threshold or i in used_detections or j in used_balls:
            continue
        used_detections.add(i)
        used_balls.add(j)
        matches.append({'detection_index':i,'ball_index':j,'error_m':error})
    return {'matched_balls':len(matches),'total_balls':len(reference),
            'missed_ball_indices':[j for j in range(len(reference)) if j not in used_balls],
            'false_positive_indices':[i for i in range(len(observations)) if i not in used_detections],
            'matches':matches,
            'max_error_m':max((item['error_m'] for item in matches),default=None)}


def annotate(rgb,observations):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(system_font_path(),27)
    for index,item in enumerate(observations,start=1):
        box = item['bbox_xyxy']
        pad = 7
        draw.rectangle((box[0]-pad,box[1]-pad,box[2]+pad,box[3]+pad),outline='#aeea80',width=4)
        text = f"{item.get('ball_id',index)} / {item['confidence']:.2f}"
        x,y = max(0,box[0]-pad),max(0,box[1]-50)
        draw.rectangle((x,y,x+160,y+42),fill='#10232c')
        draw.text((x+5,y+2),text,font=font,fill='#f0f6f5')
    return image


def run_detection(config,vision,fixed,output):
    calibration = load_json(output/'calibration.json')
    detector = FixedBallDetector(ROOT/vision['model'],vision,fixed)
    views = []
    all_observations = []
    for camera in calibration:
        rgb = np.asarray(Image.open(output/f"camera_{camera['id']}_rgb.png").convert('RGB'))
        results = {}
        for tiled in (False,True):
            result = detector.detect(rgb,camera,config['ball_radius_m'],tiled=tiled)
            results[result['method']] = result
            print(json.dumps({'camera':camera['id'],'method':result['method'],
                              'detections':len(result['observations']),
                              'seconds':result['elapsed_seconds']},indent=2),flush=True)
        observations = results['uniform_tiles']['observations']
        all_observations.extend(observations)
        views.append({'camera_id':camera['id'],'results':results})
    merged = merge_ball_estimates(all_observations,fixed['cross_camera_merge_distance_m'])
    for index,group in enumerate(merged,start=1):
        group['ball_id'] = index
        for observation in group['observations']:
            observation['ball_id'] = index
    for view in views:
        rgb = np.asarray(Image.open(output/f"camera_{view['camera_id']}_rgb.png").convert('RGB'))
        annotate(rgb,view['results']['uniform_tiles']['observations']).save(
            output/f"camera_{view['camera_id']}_yolo.jpg",quality=95)
    # Freeze image-only outputs before reading any evaluation ground truth.
    payload = {'views':views,'merged_ball_estimates':merged,
               'source':'YOLO RGB boxes and calibrated flat-floor ray intersections',
               'robot_dispatch_connected':False}
    (output/'detections.json').write_text(json.dumps(payload,indent=2)+'\n')
    reference = load_json(output/'evaluation_reference.json')
    evaluation = evaluate(merged,reference['ball_positions_m'],fixed['evaluation_match_distance_m'])
    for view,camera in zip(views,calibration):
        points = reference['ball_positions_m']
        for result in view['results'].values():
            result['evaluation'] = evaluate(result['observations'],points,fixed['evaluation_match_distance_m'])
        projected = []
        for point in points:
            pixel = world_to_pixel(point,np.array(camera['intrinsics']),np.array(camera['transform']))
            local = np.linalg.solve(np.array(camera['transform']),np.r_[point,1])[:3]
            projected.append({'pixel':None if pixel is None else pixel.tolist(),
                              'approx_ball_diameter_pixels':float(camera['intrinsics'][0][0]*2*config['ball_radius_m']/(-local[2]))})
        view['evaluation_only_ball_projections'] = projected
    report = {**payload,'evaluation':evaluation,'camera_count':len(calibration),
              'robot_count':len(reference['robots']),'robot_checks':reference['robots'],
              'captured_at_utc':reference['captured_at_utc'],
              'checked_at_utc':datetime.now(timezone.utc).isoformat(),
              'model_sha256':hashlib.sha256((ROOT/vision['model']).read_bytes()).hexdigest(),
              'operating_area_status':'pending_user_decision',
              'scope':'stationary synthetic scene; two fixed cameras; no moving players, flying balls or robot dispatch',
              'observation_pipeline_executed':True,
              'all_reference_balls_detected':evaluation['matched_balls']==evaluation['total_balls'],
              'no_false_positive_estimates':not evaluation['false_positive_indices']}
    (output/'fixed_camera_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'camera_count':len(calibration),'evaluation':evaluation},indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['amdgpu','cpu'],default='amdgpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'fixed_cameras')
    parser.add_argument('--replay',action='store_true',help='Re-run YOLO on saved RGB and calibration, without Genesis.')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    config,vision,fleet,fixed = [load_json(ROOT/name) for name in
                               ('scene_config.json','vision_config.json','fleet_config.json','fixed_camera_config.json')]
    if not args.replay:
        capture(config,fleet,vision,fixed,output,args.backend)
    report = run_detection(config,vision,fixed,output)
    from fixed_camera_visuals import save_visuals
    save_visuals(output,config,fleet,fixed,report)


if __name__ == '__main__':
    main()
