"""RGB-only person stops; synthetic mesh translation and stale-camera episodes.

Actor positions enter only scene animation and evaluation, never the stop gate.
This is not a real-person, hardware, real-time or certified-safety validation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import av
import genesis as gs
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from fixed_camera_visuals import system_font_path
from drive_benchmark import hashes
from fixed_detection import FixedBallDetector
from fleet_scene import build_fleet,hold_fleet
from handoff_control import FixedNavigator,trial_position_allowed
from onboard_vision import rotation_of
from person_safety import PersonStopGate
from scene import as_array
from vision_control import court_position_allowed

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args();output=args.output_dir.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);provenance=hashes()
    (output/'person_stop_check.json').write_text('{"status":"running","verified":false}\n')
    config,vision,fleet,fixed,mission=[read(n) for n in ('scene_config.json','vision_config.json','fleet_config.json','fixed_camera_config.json','mission_straight_config.json')]
    config.update(ball_positions_m=[],robot_urdf='collector_beveled.urdf',physics_substeps=vision['physics_substeps'])
    fleet['robots'][0]['position_m'][0]=-14.35;fleet['robots'][1]['position_m'][0]=14.35
    gs.init(backend=gs.amdgpu,seed=13,logging_level='warning')
    writer=None;rows=[];trace=[];image_trace=[];frames=0
    try:
        scene,robots,_,_,_,_,_=build_fleet(config,fleet,vision)
        cameras=[scene.add_camera(res=tuple(s['resolution']),pos=tuple(s['position_m']),lookat=tuple(s['lookat_m']),
            up=(0,0,1),fov=s['vertical_fov_degrees'],near=.05,far=200,GUI=False) for s in fixed['cameras']]
        observer=scene.add_camera(res=(800,450),pos=(-12,-11,4),lookat=(-12,-5,.6),fov=65,GUI=False)
        actor=scene.add_entity(gs.morphs.Mesh(file=str(ROOT/'assets/person-cesium/CesiumMan.glb'),
            pos=(-10.4,-1.,.01),scale=1.75/1.5065499544143677,euler=(0,0,30),fixed=True,
            collision=True,decimate=False),name='synthetic_person')
        scene.build();detector=FixedBallDetector(ROOT/vision['model'],vision,fixed)
        calibration=[{**s,'intrinsics':np.asarray(c.intrinsics).tolist(),'transform':np.asarray(c.transform).tolist()} for c,s in zip(cameras,fixed['cameras'])]
        (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
        font=ImageFont.truetype(system_font_path(),22)
        small=ImageFont.truetype(system_font_path(),17)
        if args.video:
            writer=av.open(str(output/'person_stop.mp4'),'w',options={'movflags':'+faststart'})
            stream=writer.add_stream('libx264',rate=12);stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p'
            stream.options={'crf':'21','preset':'fast'}
        for ep,(rid,stale) in enumerate((('A',False),('B',False),('A',True))):
            if ep:scene.reset()
            sign=1 if rid=='A' else -1;robot=robots[0 if rid=='A' else 1];hold_fleet(robots)
            wd=[robot.get_joint(n).dofs_idx_local[0] for n in ('left_wheel_joint','right_wheel_joint')]
            jd=[robot.get_joint(n).dofs_idx_local[0] for n in ('shovel_lift_joint','shovel_tilt_joint')]
            robot.control_dofs_position(np.array([-.30,.30]),dofs_idx_local=jd)
            actor.set_pos(np.array([sign*-10.4,sign*-1.,.01]))
            yaw=math.radians(30 if rid=='A' else 210)/2
            actor.set_quat(np.array([math.cos(yaw),0,0,math.sin(yaw)]))
            for _ in range(180):scene.step()
            p0=as_array(robot.get_pos());r0=rotation_of(robot)
            nav_config={**mission,'trial_floor_bounds_m':mission['trial_zones'][rid]}
            nav=FixedNavigator(nav_config);nav.target=p0+r0@np.array([4.85,0,0])
            gate=PersonStopGate(mission['person_safety'],[s['id'] for s in fixed['cameras']])
            minimum_gap=math.inf;max_tilt=0.;min_height=1.;max_speed=0.;violated=False
            max_persons=0;errors=[];stop_progress=None;stop_time=None;settled=None;hold_joints=None
            zero_after=True;next_frame=0.;last_fixed=None
            observer.set_pose(pos=(sign*-12.1,sign*-11.,4.),lookat=(sign*-12.,sign*-5.,.6))
            for step in range(10*120):
                now=step/120
                actor_y=-1. if stale else -1.-min(6.8,max(0.,now-1.5)*1.4)
                actor_position=np.array([sign*(-5. if stale else -10.4),sign*actor_y,.01]);actor.set_pos(actor_position)
                if now>=next_frame-1e-8:
                    for camera,item in zip(cameras,calibration):
                        if stale and now>=2.5 and item['id']=='B':continue
                        rgb=np.asarray(camera.render(antialiasing=True)[0],dtype=np.uint8)
                        detected=detector.detect(rgb,item,config['ball_radius_m'],tiled=True)
                        gate.observe(item['id'],detected['persons'],now,detected['unlocalized_person_count'])
                        max_persons=max(max_persons,len(detected['persons']))
                        errors.extend(float(np.linalg.norm(np.asarray(p['position_m'][:2])-actor_position[:2])) for p in detected['persons'])
                        image_trace.append(dict(episode=ep+1,seconds=now,camera_id=item['id'],persons=detected['persons'],
                            unlocalized_person_count=detected['unlocalized_person_count'],evaluation_actor_position_m=actor_position.tolist()))
                        if item['id']==rid:
                            im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
                            for person in detected['persons']:draw.rectangle(person['bbox_xyxy'],outline='#ff685a',width=7)
                            last_fixed=np.asarray(im)
                            if step==0:im.save(output/f'{ep+1}_{rid}_initial_yolo.png')
                    next_frame=now+mission['fixed_update_period_seconds']
                p=as_array(robot.get_pos());rotation=rotation_of(robot);nav.last_seen=now
                wheels,_,_=nav.command(p,rotation,now);progress=float((p-p0)@r0[:,0]);speed=nav.forward_speed
                if gate.blocked(p,speed,float(wheels.mean())*.11,now):
                    if stop_time is None:
                        stop_time=now;stop_progress=progress
                        hold_joints=as_array(robot.get_dofs_position(dofs_idx_local=jd)).copy()
                        if last_fixed is not None:Image.fromarray(last_fixed).save(output/f'{ep+1}_{rid}_stop_yolo.png')
                    wheels=np.zeros(2);robot.control_dofs_position(hold_joints,dofs_idx_local=jd)
                if stop_time is not None:
                    zero_after &= bool(np.max(np.abs(wheels))<1e-9)
                    if abs(speed)<.03 and now-stop_time>.25 and settled is None:settled=now
                robot.control_dofs_velocity(wheels,dofs_idx_local=wd)
                gap=float(np.linalg.norm(p[:2]-actor_position[:2]));minimum_gap=min(minimum_gap,gap)
                q=as_array(robot.get_quat());tilt=math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))))
                max_tilt=max(max_tilt,tilt);min_height=min(min_height,float(p[2]));max_speed=max(max_speed,speed)
                violated |= not trial_position_allowed(p,nav_config) or not court_position_allowed(p,config['playing_area_m'],'practice')
                if step%6==0:trace.append(dict(episode=ep+1,robot=rid,seconds=now,robot_position_m=p.tolist(),
                    filtered_own_speed_m_s=speed,wheel_commands_rad_s=wheels.tolist(),stop_latched=gate.latched,
                    evaluation_person_position_m=actor_position.tolist(),evaluation_centre_distance_m=gap))
                if writer and step%10==0:
                    canvas=Image.new('RGB',(1280,720),'#10232c');draw=ImageDraw.Draw(canvas)
                    draw.text((22,15),f'{rid} / '+('画像更新の停止試験' if stale else '模擬人物をYOLOで検出 → 停止'),font=font,fill='#edf6f5')
                    draw.text((22,53),f'{now:.2f} 秒 / 車体 {speed:.2f} m/s / '+('停止を保持' if gate.latched else '走行中'),font=font,fill='#c0ef93')
                    canvas.paste(Image.fromarray(np.asarray(observer.render()[0],dtype=np.uint8)),(0,108))
                    if last_fixed is not None:canvas.paste(Image.fromarray(last_fixed).resize((480,270)),(800,108))
                    draw.text((818,400),'固定カメラ / 直近の検出画像',font=small,fill='#bacbd0')
                    draw.text((22,590),'模擬人物による試験 / 等速再生',font=small,fill='#bacbd0')
                    draw.text((22,626),f'評価専用の人物との中心距離 {gap:.2f} m',font=small,fill='#bacbd0')
                    if step in (0,1190):canvas.save(output/f'{ep+1}_{rid}_{"initial" if step==0 else "final"}.png')
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):writer.mux(packet)
                    frames+=1
                scene.step()
            final_progress=float((as_array(robot.get_pos())-p0)@r0[:,0])
            reason='safety_vision_stale' if stale else 'person_proximity_stop'
            checks=dict(correct_rgb_or_freshness_stop=gate.event is not None and gate.event['reason']==reason,
                physically_stopped=settled is not None and abs(nav.forward_speed)<.03,held_stop_no_auto_resume=zero_after and gate.latched,
                yolo_person_detected=max_persons>=1,clearance_over_combined_footprints=minimum_gap>1.28,
                stable_and_inside_trial=not violated and max_tilt<8 and min_height>.17)
            row=dict(episode=ep+1,robot=rid,kind='stale_camera' if stale else 'synthetic_person_crossing',verified=all(checks.values()),checks=checks,
                event=gate.event,maximum_speed_m_s=max_speed,stop_command_seconds=stop_time,settled_seconds=settled,
                braking_coast_m=None if stop_progress is None else final_progress-stop_progress,
                minimum_evaluation_person_centre_distance_m=minimum_gap,maximum_tilt_degrees=max_tilt,
                projection_error_median_m=float(np.median(errors)) if errors else None)
            rows.append(row);print(json.dumps(row),flush=True)
        if writer:
            for packet in stream.encode():writer.mux(packet)
            writer.close();writer=None
            with av.open(str(output/'person_stop.mp4')) as reader:assert sum(1 for _ in reader.decode(video=0))==frames
        report=dict(status='complete',verified=all(r['verified'] for r in rows) and hashes()==provenance,
            source_unchanged=hashes()==provenance,source_sha256=provenance,rows=rows,person_source='RGB YOLO feet projected to court',
            controller_uses_simulator_person_positions=False,real_people_tested=False,automatic_resume=False,
            actor='CesiumMan static posed mesh translated across path; no walking animation',physics_substeps=vision['physics_substeps'],
            video=None if not args.video else dict(frames=frames,fps=12,seconds=frames/12,playback_speed=1.))
        for name,data in [('person_stop_check.json',report),('person_rgb_observations.json',image_trace),('control_trace.json',trace)]:
            (output/name).write_text(json.dumps(data,indent=2)+'\n')
        if not report['verified']:raise RuntimeError('Person stop checks failed; inspect report.')
    finally:
        if writer:writer.close()
        gs.destroy()


if __name__=='__main__':main()
