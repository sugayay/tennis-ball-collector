"""YOLO RGB detection -> visual approach -> physical one-ball shovel pickup.

Simulator ball state is isolated to test initialization and result evaluation.
The approach controller receives only YOLO observations and the robot's own pose.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import av
import genesis as gs
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fixed_camera_visuals import system_font_path

from scene import ROOT, as_array, build_scene, initialize_shovel, save_rgb
from shovel_demo import eased, local_position
from onboard_vision import OnboardCamera, BallDetector, rotation_of
from vision_control import VisualApproach, court_position_allowed


LABELS = {
    'approach':'球を探して接近', 'scoop':'前進しながら球をすくう',
    'secure':'球をショベルに保持', 'lift':'持ち上げる', 'tip':'カゴへ投入',
    'level':'ショベルを戻す', 'lower':'ショベルを下げる', 'retain':'カゴに保持',
}


def panel_frame(camera, robot, ego_image, phase, seconds):
    p = as_array(robot.get_pos())
    camera.set_pose(pos=(p[0]+1.25, p[1]-1.5, 1.3), lookat=(p[0]+0.06, p[1], 0.32))
    rgb = np.asarray(camera.render()[0], dtype=np.uint8)
    canvas = Image.new('RGB', (1280, 720), (16, 35, 44))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    canvas.paste(Image.fromarray(ego_image).resize((320, 240)), (960, 60))
    font_path = system_font_path()
    font = ImageFont.truetype(font_path, 18) if Path(font_path).exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((976, 18), '車載カメラ / YOLO', font=font, fill=(192, 239, 147))
    draw.text((976, 330), LABELS.get(phase, phase), font=font, fill=(240, 246, 245))
    draw.text((976, 370), f'{seconds:05.2f} 秒', font=font, fill=(185, 203, 208))
    draw.text((976, 430), 'コート外で回収', font=font, fill=(185, 203, 208))
    draw.text((976, 470), '球の位置は画像から推定', font=font, fill=(185, 203, 208))
    return np.asarray(canvas)


def episode(scene, robot, target, onboard, detector, camera, config, vision, output, *, mode, record_video):
    output.mkdir(parents=True, exist_ok=True)
    (output/'vision_check.json').write_text(json.dumps({'verified':False,'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat()})+'\n')
    shovel_dofs = initialize_shovel(robot)
    wheel_dofs = [robot.get_joint(n).dofs_idx_local[0] for n in ('left_wheel_joint','right_wheel_joint')]
    robot.set_dofs_kp(np.zeros(2), dofs_idx_local=wheel_dofs)
    robot.set_dofs_kv(np.full(2, 12.0), dofs_idx_local=wheel_dofs)
    robot.set_dofs_force_range(np.full(2,-6.0), np.full(2,6.0), dofs_idx_local=wheel_dofs)
    robot.control_dofs_velocity(np.zeros(2), dofs_idx_local=wheel_dofs)
    for _ in range(120):
        scene.step()
    controller = VisualApproach(config['playing_area_m'], mode)
    start_robot = as_array(robot.get_pos())
    # Evaluation below is observational only; neither value is passed to the controller.
    start_ball = as_array(target.get_pos())
    max_ball_height = float(start_ball[2])
    max_base_tilt = 0.0
    min_base_height = float(start_robot[2])
    court_rule_violated = False
    steps = 0
    phases, detections, retention, control_trace = [], [], [], []
    frames = 0
    heading_reference = None
    previous_heading = None
    heading_rate = 0.0
    ego_rgb = onboard.capture()
    observations, ego_annotated = detector.detect(ego_rgb, onboard, config['ball_radius_m'])
    Image.fromarray(ego_rgb).save(output/'onboard_start_rgb.png')
    Image.fromarray(ego_annotated).save(output/'onboard_start_yolo.png')
    controller.observe(observations, start_robot, rotation_of(robot), 0.0)
    writer = av.open(str(output/'vision_pickup.mp4'), 'w') if record_video else None
    stream = None
    if writer:
        stream = writer.add_stream('libx264', rate=12)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, 'yuv420p'
        stream.options = {'crf':'20', 'preset':'fast'}

    def snapshot(name):
        state = {'name':name, 'seconds':steps/120,
                 'robot_position_m':as_array(robot.get_pos()).tolist(),
                 'ball_position_m':as_array(target.get_pos()).tolist(),
                 'ball_in_robot_frame_m':local_position(robot, as_array(target.get_pos())).tolist(),
                 'shovel_joint_radians':as_array(robot.get_dofs_position(dofs_idx_local=shovel_dofs)).tolist(),
                 'wheel_velocity_rad_s':as_array(robot.get_dofs_velocity(dofs_idx_local=wheel_dofs)).tolist(),
                 'robot_yaw_radians':math.atan2(rotation_of(robot)[1,0],rotation_of(robot)[0,0])}
        phases.append(state)
        print(json.dumps(state), flush=True)
        return state

    def sense():
        nonlocal ego_rgb, ego_annotated, observations
        ego_rgb = onboard.capture()
        observations, ego_annotated = detector.detect(ego_rgb, onboard, config['ball_radius_m'])
        detections.append({'seconds':steps/120, 'observations':[asdict(o) for o in observations]})

    def step(phase):
        nonlocal steps, frames, max_ball_height, max_base_tilt, min_base_height, court_rule_violated
        scene.step()
        steps += 1
        # Test evaluator: these measurements never feed VisualApproach or the shovel timing.
        ball_position = as_array(target.get_pos())
        base = as_array(robot.get_pos())
        max_ball_height = max(max_ball_height, float(ball_position[2]))
        min_base_height = min(min_base_height, float(base[2]))
        q = as_array(robot.get_quat())
        up_z = 1-2*(q[1]**2+q[2]**2)
        max_base_tilt = max(max_base_tilt, math.degrees(math.acos(float(np.clip(up_z,-1,1)))))
        court_rule_violated |= not court_position_allowed(base, config['playing_area_m'], mode)
        if writer and steps % 10 == 0:
            if phase != 'approach':
                sense()
            rgb = panel_frame(camera, robot, ego_annotated, phase, steps/120)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                writer.mux(packet)
            frames += 1
        return local_position(robot, ball_position)

    def move(name, seconds, a0, a1, b0, b1, wheel=0.0, brake=False):
        nonlocal previous_heading, heading_rate
        points = []
        total = round(seconds*120)
        for i in range(total):
            t = (i+1)/total
            arm, bucket = eased(a0,a1,t), eased(b0,b1,t)
            robot.control_dofs_position(np.array([arm,bucket-arm]), dofs_idx_local=shovel_dofs)
            speed = wheel*(1-t if brake else 1)
            p = as_array(robot.get_pos())
            rotation = rotation_of(robot)
            yaw = math.atan2(rotation[1,0],rotation[0,0])
            if previous_heading is not None:
                delta = math.atan2(math.sin(yaw-previous_heading),math.cos(yaw-previous_heading))
                heading_rate = 0.8*heading_rate+0.2*delta*120
            previous_heading = yaw
            error = math.atan2(math.sin(heading_reference-yaw),math.cos(heading_reference-yaw))
            angular = float(np.clip(6.0*error-1.2*heading_rate,-1.2,1.2)) if speed else 0.0
            forecast = p+rotation@np.array([0.15 if speed else 0,0,0])
            if not court_position_allowed(forecast, config['playing_area_m'], mode):
                speed = 0.0
                angular = 0.0
            correction = angular*0.45/(2*0.11)
            robot.control_dofs_velocity(np.array([speed-correction,speed+correction]), dofs_idx_local=wheel_dofs)
            points.append(step(name))
        snapshot(name)
        return points

    picked_up = False
    status = 'waiting_for_ball'
    try:
        for _ in range(8*120):
            if steps % 6 == 0:
                sense()
                controller.observe(observations, as_array(robot.get_pos()), rotation_of(robot), steps/120)
            wheels, ready, status = controller.command(as_array(robot.get_pos()), rotation_of(robot), steps/120)
            if steps % 6 == 0:
                control_trace.append({'seconds':steps/120,'status':status,'wheel_commands':wheels.tolist(),
                                      'robot_position_m':as_array(robot.get_pos()).tolist(),
                                      'robot_yaw_radians':math.atan2(rotation_of(robot)[1,0],rotation_of(robot)[0,0]),
                                      'estimated_world_target_m':controller.world_target.tolist() if controller.world_target is not None else None})
            if ready:
                picked_up = True
                break
            robot.control_dofs_velocity(wheels, dofs_idx_local=wheel_dofs)
            step('approach')
        snapshot('approach')
        Image.fromarray(ego_annotated).save(output/'onboard_at_pickup_yolo.png')
        if picked_up:
            heading_reference = math.atan2(rotation_of(robot)[1,0],rotation_of(robot)[0,0])
            previous_heading = heading_reference
            pickup, capture, carry = -0.448, math.radians(5), math.radians(15)
            raised, tipped = math.radians(125), math.radians(-50)
            move('scoop',0.30,pickup,-0.20,0,capture,wheel=6.0)
            move('secure',1.0,-0.20,-0.20,capture,carry,wheel=6.0,brake=True)
            p = as_array(robot.get_pos())
            camera.set_pose(pos=(p[0]+1.25,p[1]-1.5,1.3),lookat=(p[0]+0.06,p[1],0.32))
            save_rgb(camera, output/'vision_pickup.png')
            move('lift',3.0,-0.20,raised,carry,carry)
            save_rgb(camera, output/'vision_lift.png')
            move('tip',1.8,raised,raised,carry,tipped)
            move('level',1.0,raised,raised,tipped,carry)
            move('lower',3.0,raised,pickup,carry,0)
            retention = move('retain',3.0,pickup,pickup,0,0)
        else:
            robot.control_dofs_velocity(np.zeros(2), dofs_idx_local=wheel_dofs)
        p = as_array(robot.get_pos())
        camera.set_pose(pos=(p[0]+0.8,p[1]-1.3,1.3),lookat=(p[0]-0.03,p[1],0.27))
        save_rgb(camera, output/'vision_collected.png')
    finally:
        if writer:
            for packet in stream.encode():
                writer.mux(packet)
            writer.close()
    retained = len(retention) == 360 and all(-0.195<x<0.035 and abs(y)<0.106 and 0.080<z<0.180 for x,y,z in retention)
    checks = {'yolo_guided_approach_completed':picked_up,
              'ball_lifted_above_0_4m':max_ball_height>0.4,
              'ball_retained_in_basket_for_3_seconds':retained,
              'base_tilt_below_8_degrees':max_base_tilt<8,
              'base_stayed_above_0_17m':min_base_height>0.17,
              'court_entry_rule_respected':not court_rule_violated}
    if record_video:
        with av.open(str(output/'vision_pickup.mp4')) as video:
            decoded = sum(1 for _ in video.decode(video=0))
            assert decoded == frames
    report = {'verified':all(checks.values()), 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'one visible ball; YOLO RGB detections and calibrated flat-court projection; robot pose from simulator; scripted shovel after visual approach',
              'operation_mode':mode, 'controller_uses_simulator_ball_position':False,
              'ball_state_modified_during_episode':False, 'ball_attachment_used':False,
              'robot_self_collision_enabled':True, 'depth_or_segmentation_used_by_controller':False,
              'start_robot_position_m':start_robot.tolist(),'start_ball_position_m':start_ball.tolist(),
              'last_approach_status':status,'detection_updates':controller.detection_updates,
              'maximum_ball_height_m':max_ball_height,'maximum_base_tilt_degrees':max_base_tilt,
              'minimum_base_height_m':min_base_height,'robot_displacement_m':float(np.linalg.norm(as_array(robot.get_pos())[:2]-start_robot[:2])),
              'checks':checks,'phases':phases,'detections':detections,'control_trace':control_trace,
              'video':{'frames':frames,'fps':12,'seconds':frames/12,'playback_speed':1.0} if record_video else None}
    (output/'vision_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'verified':report['verified'],'checks':checks}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['amdgpu','cpu'],default='amdgpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'vision')
    parser.add_argument('--mode',choices=['practice','break'],default='practice')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--batch',action='store_true')
    parser.add_argument('--case',choices=['baseline','left','right'],default='baseline')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'vision_summary.json').write_text(json.dumps({'verified':False,'status':'running'})+'\n')
    config = json.loads((ROOT/'scene_config.json').read_text())
    vision = json.loads((ROOT/'vision_config.json').read_text())
    config['physics_substeps'] = vision['physics_substeps']
    weights = ROOT/vision['model']
    detector = BallDetector(weights, vision)
    gs.init(backend=getattr(gs,args.backend),seed=13,logging_level='warning')
    reports = []
    try:
        scene, robot, balls, camera, *_ = build_scene(config,(960,720))
        onboard = OnboardCamera(scene,robot,vision['camera'],config['robot_position_m'])
        scene.build()
        all_cases = [('baseline',0,0),('left',0.10,0.10),('right',-0.10,-0.10)]
        cases = all_cases if args.batch else [case for case in all_cases if case[0] == args.case]
        for index,(name,dx,dy) in enumerate(cases):
            if index:
                scene.reset()
            if index or dx or dy:
                # Test initialization before settling/starting an episode, never during control.
                x,y = config['ball_positions_m'][0]
                balls[0].set_pos(np.array([x+dx,y+dy,config['ball_radius_m']+0.008]))
            report = episode(scene,robot,balls[0],onboard,detector,camera,config,vision,
                             args.output_dir/name,mode=args.mode,record_video=not args.no_video and index==0)
            reports.append({'case':name,'verified':report['verified'],'checks':report['checks']})
        summary = {'verified':all(r['verified'] for r in reports),'cases':reports,
                   'model':vision['model'],'model_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                   'vision_configuration':vision,'operation_mode':args.mode,
                   'inference_mean_seconds':float(np.mean(detector.elapsed_seconds)),
                   'inference_count':len(detector.elapsed_seconds)}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'vision_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        if not summary['verified']:
            raise RuntimeError('Vision pickup did not pass every case; inspect the saved reports.')
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
