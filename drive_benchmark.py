"""Compare real chassis motion over 3.9 m in the existing empty trial zones.

The route is a declared waypoint test, without YOLO or ball collection. The
same robot, court, actuator limits and physics are used in both
directions. Resetting occurs only between episodes, never during a drive.
"""
from __future__ import annotations

import argparse
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

from fleet_scene import build_fleet, hold_fleet
from handoff_control import FixedNavigator, trial_position_allowed
from onboard_vision import rotation_of
from scene import as_array
from vision_control import court_position_allowed


ROOT = Path(__file__).resolve().parent


def hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*')) if p.is_file()
            and not any(s in ('__pycache__', 'models', 'artifacts', '.git') for s in p.relative_to(ROOT).parts)}


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = hashes()
    base = read('mission_straight_config.json')
    scene_config = read('scene_config.json')
    scene_config.update(ball_positions_m=[], robot_urdf='collector_beveled.urdf')
    fleet, vision = read('fleet_config.json'), read('vision_config.json')
    scene_config['physics_substeps'] = vision['physics_substeps']
    fleet['robots'][0]['position_m'][0] = -14.35
    fleet['robots'][1]['position_m'][0] = 14.35
    route_distance = 3.9
    (output/'drive_check.json').write_text('{"status":"running","verified":false}\n')
    gs.init(backend=gs.amdgpu, seed=13, logging_level='warning')
    writer = None
    rows, traces, frame_count = [], [], 0
    try:
        scene, robots, _, _, _, _, _ = build_fleet(scene_config, fleet, vision)
        cameras = [scene.add_camera(res=(640,480), pos=p, lookat=q, fov=50, GUI=False)
                   for p,q in (((-12.4,-10.7,2.5),(-12.4,-6.5,.15)),
                               ((12.4,10.7,2.5),(12.4,6.5,.15)))]
        scene.build()
        dofs = [[r.get_joint(n).dofs_idx_local[0] for n in ('left_wheel_joint','right_wheel_joint')]
                for r in robots]
        font = ImageFont.truetype(system_font_path(), 23)
        small = ImageFont.truetype(system_font_path(), 18)
        if args.video:
            writer = av.open(str(output/'drive_comparison.mp4'), 'w', options={'movflags':'+faststart'})
            stream = writer.add_stream('libx264', rate=12)
            stream.width, stream.height, stream.pix_fmt = 1280,720,'yuv420p'
            stream.options = {'crf':'21', 'preset':'fast'}
        for episode in range(2):
            if episode:
                scene.reset()
            hold_fleet(robots)
            labels = ('legacy','fast') if episode==0 else ('fast','legacy')
            for index,label in enumerate(labels):
                if label=='fast':
                    angle=base['travel_shovel_lift_angle_rad']
                    joints=[robots[index].get_joint(n).dofs_idx_local[0] for n in ('shovel_lift_joint','shovel_tilt_joint')]
                    robots[index].control_dofs_position(np.array([angle,-angle]),dofs_idx_local=joints)
            for _ in range(180):
                scene.step()
            starts = [as_array(r.get_pos()) for r in robots]
            rotations = [rotation_of(r) for r in robots]
            labels = ('legacy','fast') if episode==0 else ('fast','legacy')
            controls, state = [], []
            shovel_angles = [base['travel_shovel_lift_angle_rad'] if s=='fast' else -.448 for s in labels]
            for index, (label,p,rotation) in enumerate(zip(labels, starts, rotations)):
                config = read('drive_reference_config.json') if label=='legacy' else dict(base)
                config['trial_floor_bounds_m'] = base['trial_zones'][fleet['robots'][index]['id']]
                nav = FixedNavigator(config)
                # A route waypoint 0.95m beyond the required 3.9m stopping position.
                # No ball state, fabricated detection, or onboard match enters it.
                nav.target = p+rotation@np.array([route_distance+base['handoff_distance_m'],0.,0.])
                controls.append(nav)
                state.append(dict(maximum_tilt_degrees=0., minimum_height_m=1.,
                                  boundary_violation=False, court_violation=False,
                                  max_speed_m_s=0., minimum_shovel_floor_height_m=1., speeds=[], stop_command_seconds=None,
                                  stop_command_progress_m=None, settled_seconds=None,
                                  below_speed_steps=0, checkpoint_2m_seconds=None,
                                  progress_m=0., speed_m_s=0., lateral_m=0., max_lateral_m=0.))

            def capture(seconds):
                canvas = Image.new('RGB', (1280,720), '#10232c')
                draw = ImageDraw.Draw(canvas)
                draw.text((22,16), '車体の走行比較 / 同じ3.9m・等速再生', font=font, fill='#eff7f5')
                draw.text((22,53), f'シミュレーション時刻 {seconds:.2f} 秒  |  ショベル・車輪・モーターは共通', font=small, fill='#bacbd0')
                for index,label in enumerate(labels):
                    x=index*640
                    title = '前回の設定 0.55m/s' if label=='legacy' else '今回の設定 1.00m/s'
                    draw.text((x+18,91),title,font=font,fill='#c0ef93')
                    rgb=np.asarray(cameras[index].render()[0],dtype=np.uint8)
                    canvas.paste(Image.fromarray(rgb),(x,132))
                    s=state[index]
                    draw.text((x+18,624),f'走行距離 {s["progress_m"]:.2f} m / 速さ {s["speed_m_s"]:.2f} m/s',font=small,fill='#eff7f5')
                    status='停止' if s['settled_seconds'] is not None else 'ブレーキ中' if s['stop_command_seconds'] is not None else '走行中'
                    draw.text((x+18,659),status,font=small,fill='#bacbd0')
                return np.asarray(canvas)

            for step in range(25*120):
                now=step/120
                for index,(robot,nav,s,p0,rotation0) in enumerate(zip(robots,controls,state,starts,rotations)):
                    p, rotation = as_array(robot.get_pos()), rotation_of(robot)
                    nav.last_seen=now  # Known route, not a test of camera detection.
                    wheels,_,status=nav.command(p,rotation,now)
                    if labels[index]=='fast':
                        target_angle=base['travel_shovel_lift_angle_rad'] if nav.fast_route or nav.fast_braking else -.448
                        limit=base['travel_shovel_lift_rate_rad_s']/120
                        shovel_angles[index]+=float(np.clip(target_angle-shovel_angles[index],-limit,limit))
                        joints=[robot.get_joint(n).dofs_idx_local[0] for n in ('shovel_lift_joint','shovel_tilt_joint')]
                        robot.control_dofs_position(np.array([shovel_angles[index],-shovel_angles[index]]),dofs_idx_local=joints)
                    progress=float((p-p0)@rotation0[:,0])
                    local=rotation0.T@(p-p0)
                    s['progress_m']=progress; s['speed_m_s']=nav.forward_speed
                    s['lateral_m']=float(local[1]); s['max_lateral_m']=max(s['max_lateral_m'],abs(float(local[1])))
                    bucket=robot.get_link('shovel_bucket')
                    bucket_pos,bucket_rot=as_array(bucket.get_pos()),rotation_of(bucket)
                    floor=min(float((bucket_pos+bucket_rot@np.array([x,y,-.026]))[2]) for x in (0.,.25) for y in (-.12,.12))
                    if nav.fast_route:
                        s['minimum_shovel_floor_height_m']=min(s['minimum_shovel_floor_height_m'],floor)
                    q=as_array(robot.get_quat())
                    tilt=math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))))
                    s['maximum_tilt_degrees']=max(s['maximum_tilt_degrees'],tilt)
                    s['minimum_height_m']=min(s['minimum_height_m'],float(p[2]))
                    s['boundary_violation'] |= not trial_position_allowed(p,nav.config)
                    s['court_violation'] |= not court_position_allowed(p,scene_config['playing_area_m'],'practice')
                    speed=max(abs(nav.forward_speed),abs(float(np.mean(wheels))*.11))
                    forecast=p+rotation@np.array([max(.15,speed*nav.config['travel_brake_reaction_seconds']+speed*speed/(2*nav.config['travel_brake_deceleration_m_s2'])),0.,0.])
                    if not trial_position_allowed(forecast,nav.config) or not court_position_allowed(forecast,scene_config['playing_area_m'],'practice'):
                        wheels=np.zeros(2); s['boundary_violation']=True
                    if progress>=2 and s['checkpoint_2m_seconds'] is None:
                        s['checkpoint_2m_seconds']=now
                    if status in ('handoff_ready','waiting_for_onboard_match') and s['stop_command_seconds'] is None:
                        s['stop_command_seconds']=now; s['stop_command_progress_m']=progress
                    if s['stop_command_seconds'] is not None:
                        s['below_speed_steps']=s['below_speed_steps']+1 if abs(nav.forward_speed)<.03 else 0
                        if s['below_speed_steps']>=60 and s['settled_seconds'] is None:
                            s['settled_seconds']=now
                    if 2.0<=progress<=2.7 and step%6==0:
                        s['speeds'].append(float(nav.forward_speed))
                    s['max_speed_m_s']=max(s['max_speed_m_s'],float(nav.forward_speed))
                    robot.control_dofs_velocity(wheels,dofs_idx_local=dofs[index])
                    if step%6==0:
                        traces.append(dict(episode=episode+1,seconds=now,robot=fleet['robots'][index]['id'],
                            profile=labels[index],robot_position_m=p.tolist(),progress_m=progress,
                            filtered_own_speed_m_s=nav.forward_speed,wheel_commands_rad_s=wheels.tolist(),
                            actual_wheel_speeds_rad_s=as_array(robot.get_dofs_velocity(dofs_idx_local=dofs[index])).tolist(),
                            tilt_degrees=tilt,status=status))
                scene.step()
                if writer and episode==0 and step%10==0:
                    rgb=capture(now)
                    if frame_count==0:
                        Image.fromarray(rgb).save(output/'initial.png')
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                        writer.mux(packet)
                    frame_count+=1
                if all(s['settled_seconds'] is not None for s in state):
                    break
            if episode==0:
                Image.fromarray(capture((step+1)/120)).save(output/'final.png')
            for index,s in enumerate(state):
                speeds=s.pop('speeds')
                s.update(episode=episode+1,robot=fleet['robots'][index]['id'],profile=labels[index],
                         cruise_median_m_s=float(np.median(speeds)) if speeds else None,
                         cruise_sample_count=len(speeds),
                         braking_coast_m=None if s['stop_command_progress_m'] is None else s['progress_m']-s['stop_command_progress_m'])
                s['passed']=bool(s['settled_seconds'] is not None and route_distance-.05<=s['progress_m']<=route_distance+.12
                    and s['braking_coast_m'] is not None and abs(s['braking_coast_m'])<=.12
                    and s['maximum_tilt_degrees']<8 and s['minimum_height_m']>.17
                    and s['max_lateral_m']<(.20 if labels[index]=='legacy' else .10) and not s['boundary_violation'] and not s['court_violation'])
                rows.append(s)
                print(json.dumps(s),flush=True)
        if writer:
            for packet in stream.encode():
                writer.mux(packet)
            writer.close(); writer=None
        video=None
        if args.video:
            with av.open(str(output/'drive_comparison.mp4')) as reader:
                decoded=sum(1 for _ in reader.decode(video=0))
            assert decoded==frame_count
            video=dict(frames=frame_count,fps=12,seconds=frame_count/12,playback_speed=1.)
        legacy=[r for r in rows if r['profile']=='legacy']; fast=[r for r in rows if r['profile']=='fast']
        ratio=float(np.mean([r['cruise_median_m_s'] for r in fast])/np.mean([r['cruise_median_m_s'] for r in legacy]))
        checks=dict(all_four_drives_stable_and_stopped=all(r['passed'] for r in rows),
                    actual_fast_peak_at_least_0_9_m_s=all(r['max_speed_m_s']>=.9 for r in fast),
                    fast_track_deviation_under_10cm=all(r['max_lateral_m']<.10 for r in fast),
                    fast_cruise_median_at_least_0_9_m_s=all(r['cruise_median_m_s']>=.9 for r in fast),
                    raised_shovel_cleared_floor=all(r['minimum_shovel_floor_height_m']>.02 for r in fast),
                    both_robots_reached_2m_sooner=all(next(r for r in fast if r['robot']==rid)['checkpoint_2m_seconds']
                        <next(r for r in legacy if r['robot']==rid)['checkpoint_2m_seconds'] for rid in ('A','B')),
                    source_unchanged=hashes()==provenance)
        report=dict(status='complete',verified=all(checks.values()),checks=checks,
            created_at_utc=datetime.now(timezone.utc).isoformat(),scope='known_3_9m_waypoint_in_same_empty_temporary_zones',route_distance_m=route_distance,cruise_measurement_window_progress_m=[2.0,2.7],reference_version='drive-v21',
            measures_actual_chassis_motion=True,uses_yolo=False,ball_collection_test=False,
            physics_and_actuator_limits_unchanged=True,physics_substeps=vision['physics_substeps'],fast_travel_uses_raised_shovel=True,resets_only_between_episodes=True,
            actual_cruise_speed_ratio=ratio,rows=rows,video=video,source_sha256=provenance)
        (output/'drive_trace.json').write_text(json.dumps(traces,indent=2)+'\n')
        (output/'drive_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(verified=report['verified'],checks=checks,speed_ratio=ratio)),flush=True)
        if not report['verified']:
            raise RuntimeError('Driving comparison failed.')
    finally:
        if writer:
            writer.close()
        gs.destroy()


if __name__=='__main__':
    main()
