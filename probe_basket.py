"""Static sensor calibration only. Ball placement is NOT a pickup/control test."""
import argparse,json
from pathlib import Path
import genesis as gs
import numpy as np
from PIL import Image
from fleet_scene import build_fleet,hold_fleet
from scene import ROOT,as_array
from onboard_vision import BallDetector,rotation_of
from basket_vision import BasketCamera,BasketDetector

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg,vision,fleet,basket=[json.loads((ROOT/n).read_text()) for n in ['scene_config.json','vision_config.json','fleet_config.json','basket_camera_config.json']]
    cfg['robot_urdf']='collector_beveled.urdf';cfg['physics_substeps']=8
    cfg['ball_positions_m']=[[-10,-7],[10,7],[12,7]]
    gs.init(backend=gs.amdgpu,seed=13,logging_level='warning')
    try:
        scene,robots,balls,_,_,_,_=build_fleet(cfg,fleet,vision)
        cameras=[BasketCamera(scene,r,basket,s['position_m']) for r,s in zip(robots,fleet['robots'])]
        scene.build();hold_fleet(robots)
        for _ in range(120):scene.step()
        detector=BasketDetector(BallDetector(ROOT/vision['model'],vision),basket)
        rows=[]
        def capture(label):
            for rid,robot,camera in zip(['A','B'],robots,cameras):
                rgb=camera.capture();observed,annotated=detector.detect(rgb,camera)
                Image.fromarray(rgb).save(args.output_dir/f'{label}_{rid}_rgb.png')
                Image.fromarray(annotated).save(args.output_dir/f'{label}_{rid}_yolo.png')
                local=[(rotation_of(robot).T@(as_array(b.get_pos())-as_array(robot.get_pos()))).tolist() for b in balls]
                row=dict(label=label,robot=rid,**observed,evaluation_only_local_ball_positions=local)
                rows.append(row);print(json.dumps(row),flush=True)
        capture('empty')
        # Known static placements to inspect empty/one/two-ball visibility.
        # This diagnostic never commands a drive, lift, or collection attempt.
        for ball,robot,local in [(balls[0],robots[0],[-.08,0,.15]),
                                  (balls[1],robots[1],[-.11,-.055,.15]),
                                  (balls[2],robots[1],[-.11,.055,.15])]:
            ball.set_pos(as_array(robot.get_pos())+rotation_of(robot)@np.asarray(local))
        for _ in range(240):scene.step()
        capture('placed')
        (args.output_dir/'probe.json').write_text(json.dumps(dict(scope='static_sensor_calibration_not_a_pickup_trial',ball_positions_changed_for_static_fixture=True,rows=rows),indent=2)+'\n')
    finally:gs.destroy()

if __name__=='__main__':main()
