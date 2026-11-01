"""Fixed RGB -> bounded navigation -> onboard YOLO -> physical shovel pickup.

Two free robots exist in the same scene. Only A moves; B holds zero wheel
velocity. Target coordinates are obtained from current fixed-camera RGB.
Simulator ball state is used only for initialization and separate evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

import av
import genesis as gs
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from fixed_camera_visuals import system_font_path

from camera_geometry import merge_ball_estimates,world_to_pixel
from fixed_camera_demo import annotate
from fixed_detection import FixedBallDetector
from fleet_scene import build_fleet,hold_fleet
from handoff_control import FixedNavigator,HandoffVisualApproach,observations_for_target,trial_position_allowed
from onboard_vision import BallDetector,rotation_of
from scene import ROOT,as_array,save_rgb
from shovel_demo import eased,local_position
from vision_control import court_position_allowed


LABELS = {'fixed':'固定カメラで位置を見て移動','brake':'車載カメラへ引き継ぐ',
          'visual':'車載カメラで球に接近','scoop':'球をすくう','secure':'球を保持',
          'lift':'カゴの上まで持ち上げる','tip':'カゴへ投入','level':'ショベルを戻す',
          'lower':'ショベルを下げる','retain':'カゴで3秒間保持'}


def load_json(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def episode(scene,robots,balls,onboard,cameras,main_camera,config,vision,fixed,trial,output,*,record_video):
    output.mkdir(parents=True,exist_ok=True)
    (output/'handoff_check.json').write_text(json.dumps({'verified':False,'status':'running'})+'\n')
    active,waiting = robots
    ego = onboard[0]
    navigator = FixedNavigator(trial)
    fixed_detector = FixedBallDetector(ROOT/vision['model'],vision,fixed)
    onboard_detector = BallDetector(ROOT/vision['model'],vision)
    hold_fleet(robots)
    wheel_dofs = [active.get_joint(name).dofs_idx_local[0] for name in ('left_wheel_joint','right_wheel_joint')]
    shovel_dofs = [active.get_joint(name).dofs_idx_local[0] for name in ('shovel_lift_joint','shovel_tilt_joint')]
    for _ in range(120):
        scene.step()
    initial_active = as_array(active.get_pos())
    initial_waiting = as_array(waiting.get_pos())
    # Evaluation-only references. These variables never enter either detector or controller.
    reference_ball = balls[0]
    initial_ball = as_array(reference_ball.get_pos())
    calibration = [{**spec,'intrinsics':np.asarray(camera.intrinsics).tolist(),
                    'transform':np.asarray(camera.transform).tolist()}
                   for spec,camera in zip(fixed['cameras'],cameras)]
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    steps,frames = 0,0
    fixed_trace,onboard_trace,control_trace,phases,retention = [],[],[],[],[]
    fixed_image = None
    fixed_image_seconds = 0.0
    ego_image = None
    observations = []
    last_wheels = np.zeros(2)
    maximum_ball_height = float(initial_ball[2])
    maximum_active_tilt = 0.0
    minimum_active_height = float(initial_active[2])
    maximum_waiting_drift = 0.0
    maximum_waiting_tilt = 0.0
    trial_violation = False
    court_violation = False
    guard_stops = 0
    fixed_drive_distance = 0.0
    next_fixed = 0.0
    font = ImageFont.truetype(system_font_path(),18)
    small_font = ImageFont.truetype(system_font_path(),15)
    writer = av.open(str(output/'handoff.mp4'),'w',options={'movflags':'+faststart'}) if record_video else None
    stream = None
    if writer:
        stream = writer.add_stream('libx264',rate=12)
        stream.width,stream.height,stream.pix_fmt = 1280,720,'yuv420p'
        stream.options = {'crf':'20','preset':'fast'}

    def sense_fixed():
        nonlocal fixed_image,fixed_image_seconds,next_fixed
        estimated = []
        views = []
        for camera,item in zip(cameras,calibration):
            rgb = np.asarray(camera.render(antialiasing=True)[0],dtype=np.uint8)
            result = fixed_detector.detect(rgb,item,config['ball_radius_m'],tiled=True)
            estimated.extend(result['observations'])
            views.append({'camera_id':item['id'],**result})
            if item['id']=='A':
                fixed_image = np.asarray(annotate(rgb,result['observations']))
                fixed_image_seconds = steps/120
            if steps==0:
                Image.fromarray(rgb).save(output/f"fixed_initial_{item['id']}_rgb.png")
                annotate(rgb,result['observations']).save(output/f"fixed_initial_{item['id']}_yolo.jpg",quality=94)
        merged = merge_ball_estimates(estimated,fixed['cross_camera_merge_distance_m'])
        navigator.observe_fixed(merged,as_array(active.get_pos()),steps/120)
        fixed_trace.append({'seconds':steps/120,'views':views,'merged_estimates':merged,
                            'selected_target_m':None if navigator.target is None else navigator.target.tolist()})
        next_fixed = steps/120+trial['fixed_update_period_seconds']

    def sense_onboard():
        nonlocal ego_image,observations
        rgb = ego.capture()
        observations,ego_image = onboard_detector.detect(rgb,ego,config['ball_radius_m'])
        navigator.observe_onboard(observations,as_array(active.get_pos()),rotation_of(active))
        onboard_trace.append({'seconds':steps/120,'observations':[asdict(item) for item in observations],
                              'matching_target_count':len(observations_for_target(
                                  observations,as_array(active.get_pos()),rotation_of(active),navigator.target,
                                  trial['target_association_distance_m']))})
        return rgb

    def drive(wheels):
        nonlocal last_wheels,guard_stops
        wheels = np.array(wheels,dtype=float)
        p,rotation = as_array(active.get_pos()),rotation_of(active)
        forward = float(wheels.mean())*.11
        forecast = p+rotation@np.array([.15 if forward>0 else -.15 if forward<0 else 0,0,0])
        if not trial_position_allowed(forecast,trial) or not court_position_allowed(forecast,config['playing_area_m'],'practice'):
            if np.any(np.abs(wheels)>1e-8):
                guard_stops += 1
            wheels = np.zeros(2)
        last_wheels = wheels
        active.control_dofs_velocity(wheels,dofs_idx_local=wheel_dofs)

    def render_panel(phase):
        p,rotation = as_array(active.get_pos()),rotation_of(active)
        distance = 2.0 if phase in ('fixed','brake') else 1.3
        main_camera.set_pose(pos=p+np.array([distance,-distance,1.25]),
                             lookat=p+rotation@np.array([.12,0,.10]))
        canvas = Image.new('RGB',(1280,720),'#10232c')
        canvas.paste(Image.fromarray(np.asarray(main_camera.render()[0],dtype=np.uint8)),(0,0))
        canvas.paste(Image.fromarray(fixed_image).resize((320,180)),(960,46))
        canvas.paste(Image.fromarray(ego_image).resize((320,240)),(960,285))
        draw = ImageDraw.Draw(canvas)
        draw.text((974,13),f'固定A / 最終検出 {fixed_image_seconds:.1f}秒',font=small_font,fill='#aeea80')
        draw.text((974,248),'車載カメラ / YOLO',font=font,fill='#aeea80')
        draw.text((974,555),LABELS[phase],font=small_font,fill='#f0f6f5')
        draw.text((974,590),f'{steps/120:05.2f} 秒',font=font,fill='#b9cbd0')
        draw.text((974,632),'A：移動・回収 / B：待機',font=small_font,fill='#b9cbd0')
        draw.text((974,668),'人のいない試験区間のみ',font=small_font,fill='#b9cbd0')
        return np.asarray(canvas)

    def step(phase):
        nonlocal steps,frames,maximum_ball_height,maximum_active_tilt,minimum_active_height
        nonlocal maximum_waiting_drift,maximum_waiting_tilt,trial_violation,court_violation,fixed_drive_distance
        before = as_array(active.get_pos())
        scene.step()
        steps += 1
        after = as_array(active.get_pos())
        if phase=='fixed' and last_wheels.mean()>.1:
            fixed_drive_distance += float(np.linalg.norm((after-before)[:2]))
        # Evaluation only: ball state does not influence phase changes or actuator commands.
        ball_position = as_array(reference_ball.get_pos())
        maximum_ball_height = max(maximum_ball_height,float(ball_position[2]))
        minimum_active_height = min(minimum_active_height,float(after[2]))
        q = as_array(active.get_quat())
        tilt = math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))))
        maximum_active_tilt = max(maximum_active_tilt,tilt)
        q_wait = as_array(waiting.get_quat())
        tilt_wait = math.degrees(math.acos(float(np.clip(1-2*(q_wait[1]**2+q_wait[2]**2),-1,1))))
        maximum_waiting_tilt = max(maximum_waiting_tilt,tilt_wait)
        maximum_waiting_drift = max(maximum_waiting_drift,float(np.linalg.norm((as_array(waiting.get_pos())-initial_waiting)[:2])))
        trial_violation |= not trial_position_allowed(after,trial)
        court_violation |= not court_position_allowed(after,config['playing_area_m'],'practice')
        if writer and steps%10==0:
            if phase not in ('fixed','brake','visual'):
                sense_onboard()
            for packet in stream.encode(av.VideoFrame.from_ndarray(render_panel(phase),format='rgb24')):
                writer.mux(packet)
            frames += 1
        return local_position(active,ball_position)

    def snapshot(name):
        item = {'phase':name,'seconds':steps/120,
                'robot_position_m':as_array(active.get_pos()).tolist(),
                'evaluation_ball_position_m':as_array(reference_ball.get_pos()).tolist(),
                'evaluation_ball_in_robot_m':local_position(active,as_array(reference_ball.get_pos())).tolist(),
                'image_derived_target_m':None if navigator.target is None else navigator.target.tolist()}
        phases.append(item)
        print(json.dumps(item),flush=True)

    handoff_completed,visual_ready = False,False
    handoff_time = None
    initial_target_outside = False
    initial_matching_count = None
    previous_heading = None
    heading_reference = None
    heading_rate = 0.0

    def move(name,seconds,a0,a1,b0,b1,wheel=0.,brake=False):
        nonlocal previous_heading,heading_rate
        points = []
        count = round(seconds*120)
        for index in range(count):
            t = (index+1)/count
            arm,bucket = eased(a0,a1,t),eased(b0,b1,t)
            active.control_dofs_position(np.array([arm,bucket-arm]),dofs_idx_local=shovel_dofs)
            speed = wheel*(1-t if brake else 1)
            rotation = rotation_of(active)
            yaw = math.atan2(rotation[1,0],rotation[0,0])
            if previous_heading is not None:
                delta = math.atan2(math.sin(yaw-previous_heading),math.cos(yaw-previous_heading))
                heading_rate = .8*heading_rate+.2*delta*120
            previous_heading = yaw
            error = math.atan2(math.sin(heading_reference-yaw),math.cos(heading_reference-yaw))
            angular = float(np.clip(6*error-1.2*heading_rate,-1.2,1.2)) if speed else 0.
            correction = angular*.45/(2*.11)
            drive([speed-correction,speed+correction])
            points.append(step(name))
        snapshot(name)
        return points

    try:
        sense_fixed()
        rgb = sense_onboard()
        Image.fromarray(rgb).save(output/'onboard_initial_rgb.png')
        Image.fromarray(ego_image).save(output/'onboard_initial_yolo.png')
        initial_matching_count = onboard_trace[-1]['matching_target_count']
        # Evaluator only: verify that the target really started outside this camera's frustum.
        pixel = world_to_pixel(initial_ball,ego.camera.intrinsics,ego.camera.transform)
        initial_target_outside = pixel is None or not (0 <= pixel[0] < ego.camera.res[0] and 0 <= pixel[1] < ego.camera.res[1])
        snapshot('start')
        for _ in range(round(trial['maximum_fixed_navigation_seconds']*120)):
            if steps/120 >= next_fixed-1e-8:
                sense_fixed()
            if steps%6==0:
                sense_onboard()
            wheels,ready,status = navigator.command(as_array(active.get_pos()),rotation_of(active),steps/120)
            if steps%6==0:
                control_trace.append({'phase':'fixed','seconds':steps/120,'status':status,
                                      'wheel_commands':wheels.tolist(),'robot_position_m':as_array(active.get_pos()).tolist(),
                                      'target_m':None if navigator.target is None else navigator.target.tolist()})
            if ready:
                handoff_completed = True
                handoff_time = steps/120
                break
            drive(wheels)
            step('fixed')
        snapshot('handoff')
        Image.fromarray(ego_image).save(output/'onboard_handoff_yolo.png')
        if handoff_completed:
            for _ in range(60):
                drive([0,0])
                if steps%6==0:
                    sense_onboard()
                step('brake')
            visual = HandoffVisualApproach(config['playing_area_m'],'practice')
            for _ in range(round(trial['maximum_visual_approach_seconds']*120)):
                if steps%6==0:
                    sense_onboard()
                    p,rotation = as_array(active.get_pos()),rotation_of(active)
                    selected = observations_for_target(observations,p,rotation,navigator.target,trial['target_association_distance_m'])
                    visual.observe(selected,p,rotation,steps/120)
                wheels,ready,status = visual.command(as_array(active.get_pos()),rotation_of(active),steps/120)
                if steps%6==0:
                    control_trace.append({'phase':'visual','seconds':steps/120,'status':status,
                                          'wheel_commands':wheels.tolist(),'robot_position_m':as_array(active.get_pos()).tolist(),
                                          'target_m':None if visual.world_target is None else visual.world_target.tolist()})
                if ready:
                    visual_ready = True
                    break
                drive(wheels)
                step('visual')
            snapshot('visual_ready')
        if visual_ready:
            heading_reference = math.atan2(rotation_of(active)[1,0],rotation_of(active)[0,0])
            previous_heading = heading_reference
            pickup,capture,carry = -.448,math.radians(5),math.radians(15)
            raised,tipped = math.radians(125),math.radians(-50)
            move('scoop',.30,pickup,-.20,0,capture,wheel=6.)
            move('secure',1.,-.20,-.20,capture,carry,wheel=6.,brake=True)
            render_panel('secure')
            save_rgb(main_camera,output/'handoff_pickup.png')
            move('lift',3.,-.20,raised,carry,carry)
            render_panel('lift')
            save_rgb(main_camera,output/'handoff_lift.png')
            move('tip',1.8,raised,raised,carry,tipped)
            move('level',1.,raised,raised,tipped,carry)
            move('lower',3.,raised,pickup,carry,0)
            retention = move('retain',3.,pickup,pickup,0,0)
        drive([0,0])
        render_panel('retain' if visual_ready else 'fixed')
        save_rgb(main_camera,output/'handoff_collected.png')
    finally:
        if writer:
            for packet in stream.encode():
                writer.mux(packet)
            writer.close()
    retained = len(retention)==360 and all(-.195<x<.035 and abs(y)<.106 and .080<z<.180 for x,y,z in retention)
    checks = {'target_initially_outside_onboard_image':bool(initial_target_outside),
              'target_initially_not_detected_onboard':initial_matching_count==0,
              'fixed_camera_selected_target':navigator.target is not None,
              'fixed_guided_forward_travel_over_0_1m':fixed_drive_distance>.1,
              'fixed_to_onboard_handoff_completed':handoff_completed,
              'onboard_visual_approach_completed':visual_ready,
              'ball_lifted_above_0_4m':maximum_ball_height>.4,
              'ball_retained_in_basket_for_3_seconds':retained,
              'robot_B_stayed_at_rest':maximum_waiting_drift<.025 and maximum_waiting_tilt<3,
              'robot_A_stable':maximum_active_tilt<8 and minimum_active_height>.17,
              'temporary_trial_boundary_respected':not trial_violation,
              'court_entry_rule_respected':not court_violation}
    video = None
    if record_video:
        with av.open(str(output/'handoff.mp4')) as reader:
            decoded = sum(1 for _ in reader.decode(video=0))
        assert decoded==frames
        video = {'frames':frames,'fps':12,'seconds':frames/12,'playback_speed':1.0,'fixed_inset':'last observed frame with update timestamp'}
    report = {'verified':all(checks.values()),'checks':checks,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
              'scope':trial['operation_scope'],'operating_area_status':'pending_user_decision',
              'robot_count':2,'active_robot':'A','waiting_robot':'B','fixed_camera_count':2,
              'controller_uses_simulator_ball_position':False,'ball_state_modified_during_episode':False,
              'ball_attachment_used':False,'depth_or_segmentation_used_by_controller':False,
              'robot_pose_source':'simulator proprioception','fixed_target_updates':len(fixed_trace),
              'fixed_guided_forward_travel_m':fixed_drive_distance,'handoff_seconds':handoff_time,
              'maximum_ball_height_m':maximum_ball_height,'maximum_waiting_robot_drift_m':maximum_waiting_drift,
              'maximum_active_robot_tilt_degrees':maximum_active_tilt,'minimum_active_robot_height_m':minimum_active_height,
              'trial_guard_stops':guard_stops,'phases':phases,'control_trace':control_trace,
              'onboard_detections':onboard_trace,'video':video,
              'model_sha256':hashlib.sha256((ROOT/vision['model']).read_bytes()).hexdigest()}
    (output/'fixed_observations.json').write_text(json.dumps(fixed_trace,indent=2)+'\n')
    (output/'handoff_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'verified':report['verified'],'checks':checks,'handoff_seconds':handoff_time,
                      'fixed_forward_m':fixed_drive_distance,'guard_stops':guard_stops}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['amdgpu','cpu'],default='amdgpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'handoff')
    parser.add_argument('--no-video',action='store_true')
    args = parser.parse_args()
    config,vision,fleet,fixed,trial = [load_json(name) for name in
        ('scene_config.json','vision_config.json','fleet_config.json','fixed_camera_config.json','handoff_config.json')]
    config['physics_substeps'] = vision['physics_substeps']
    # Test initialization before scene construction, not a perception/control input.
    config['ball_positions_m'][0] = trial.pop('test_ball_position_m')
    gs.init(backend=getattr(gs,args.backend),seed=13,logging_level='warning')
    try:
        scene,robots,balls,overview,top,detail,onboard = build_fleet(config,fleet,vision)
        cameras = [scene.add_camera(res=tuple(spec['resolution']),pos=tuple(spec['position_m']),
                    lookat=tuple(spec['lookat_m']),up=(0,0,1),fov=spec['vertical_fov_degrees'],
                    near=.05,far=200,GUI=False) for spec in fixed['cameras']]
        scene.build()
        report = episode(scene,robots,balls,onboard,cameras,detail,config,vision,fixed,trial,args.output_dir,
                         record_video=not args.no_video)
        save_rgb(top,args.output_dir/'handoff_top.png')
        if not report['verified']:
            raise RuntimeError('Fixed-to-onboard pickup failed a required check; inspect handoff_check.json.')
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
