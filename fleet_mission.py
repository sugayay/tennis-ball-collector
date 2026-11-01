"""Concurrent two-robot, two-ball-per-robot physical trial on an actual-size court."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from async_perception import freeze_camera, InferenceWorkers, SameStepCapture
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import av
import genesis as gs
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fixed_camera_visuals import system_font_path

from camera_geometry import merge_ball_estimates
from fixed_camera_demo import annotate
from fixed_detection import FixedBallDetector
from fleet_scene import build_fleet, hold_fleet
from handoff_control import trial_position_allowed
from basket_view_clearance import basket_view_clearance
from mission_control import FleetCoordinator
from person_safety import PersonStopGate, PersonPauseGate
from person_localization import refine_person_views
from onboard_vision import BallDetector, rotation_of
from scene import ROOT, as_array, save_rgb
from shovel_demo import local_position
from vision_control import court_position_allowed
from basket_vision import BasketCamera, BasketDetector
from recollection import corroborate_fixed_with_onboard
from lift_storage import LiftStorage, JOINTS
from basket_frame_geometry import nominal_basket_base_from_link,basket_link_fk,validate_basket_geometry,sphere_is_contained

LABELS = {'return_home':'満杯・回収場所へ戻る','await_unload':'球の取り出しを待つ','seek':'次の球を探す', 'fixed':'固定カメラで誘導', 'brake':'車載カメラへ引き継ぐ',
          'recover_backoff':'再回収のため少し後退', 'recover_wait':'止まった球を再検出',
          'visual':'車載カメラで接近', 'retry':'停止して再検出', 'fault':'異常停止',
          'scoop':'球をすくう', 'secure':'球を保持', 'lift':'持ち上げる', 'tip':'カゴへ投入',
          'level':'ショベルを戻す', 'lower':'ショベルを下げる', 'retain':'3秒間保持', 'confirm':'カゴ画像で確認', 'done':'回収目標を達成'}


def load(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def in_basket(local):
    x,y,z = local
    return -.195<x<.035 and abs(y)<.106 and .080<z<.180


class PhysicalEvaluator:
    """Isolated evaluator. Its results are never passed back to mission control."""
    def __init__(self, robots, balls, controller, output=None, lift_variant=False):
        self.robots, self.balls, self.controller = robots, balls, controller
        self.maximum_tilt = {rid:0. for rid in robots}
        self.minimum_height = {rid:float(as_array(r.get_pos())[2]) for rid,r in robots.items()}
        self.maximum_ball_heights = [0.]*len(balls)
        self.trial_violation = self.court_violation = False
        self.minimum_separation = math.inf
        self.retention_windows = {rid:[] for rid in robots}
        self.cycle_holds = {rid:[] for rid in robots}
        self.retained_history = {rid:[] for rid in robots}
        self.last_cycles = {rid:0 for rid in robots}
        self.initial_positions = {rid:as_array(r.get_pos()) for rid,r in robots.items()}
        self.travel = {rid:0. for rid in robots}
        self.previous_positions = {rid:p.copy() for rid,p in self.initial_positions.items()}
        self.fixed_travel = {rid:0. for rid in robots}
        self.phase_snapshots = []
        self.lift_variant=lift_variant;self.sample_seconds=0.;self.writing_geometry=False
        self.geometry_valid=True;self.geometry_samples={rid:0 for rid in robots};self.geometry_writer=None
        if lift_variant:
            validate_basket_geometry(ROOT/'assets/collector_basket_lift.urdf')
            self.basket_links={rid:r.get_link('refill_basket') for rid,r in robots.items()}
            self.shovel_links={rid:r.get_link('shovel_bucket') for rid,r in robots.items()}
            self.shovel_dofs={rid:[r.get_joint(n).dofs_idx_local[0] for n in ('shovel_lift_joint','shovel_tilt_joint')] for rid,r in robots.items()}
            self.basket_dofs={rid:[r.get_joint(n).dofs_idx_local[0] for n in JOINTS] for rid,r in robots.items()}
            self.geometry_writer=(output/'actual-basket-retention-trace.jsonl').open('w',encoding='utf8')
            self.writing_geometry=True
            for rid,r in robots.items():self._contained(rid,r)
            self.writing_geometry=False

    def _contained(self,rid,robot):
        if not self.lift_variant:
            return [i for i,b in enumerate(self.balls) if in_basket(local_position(robot,as_array(b.get_pos())))]
        base_p=as_array(robot.get_pos());base_r=rotation_of(robot)
        basket_p=as_array(self.basket_links[rid].get_pos());basket_r=rotation_of(self.basket_links[rid])
        q=as_array(robot.get_dofs_position(dofs_idx_local=self.basket_dofs[rid]))
        v=as_array(robot.get_dofs_velocity(dofs_idx_local=self.basket_dofs[rid]))
        fp,fr=basket_link_fk(base_p,base_r,q)
        error_p=float(np.linalg.norm(basket_p-fp));error_r=float(np.max(np.abs(basket_r-fr)))
        frame_ok=bool(np.isfinite(np.concatenate((base_p,base_r.ravel(),basket_p,basket_r.ravel(),q,v))).all()
                      and error_p<=2e-5 and error_r<=2e-5)
        contained=[];measurements=[]
        for i,b in enumerate(self.balls):
            world=as_array(b.get_pos());local=nominal_basket_base_from_link(world,basket_p,basket_r)
            finite=bool(np.isfinite(world).all() and np.isfinite(local).all())
            inside=frame_ok and finite and sphere_is_contained(local)
            if inside:contained.append(i)
            frame_ok=frame_ok and finite
            measurements.append(dict(id=i,world_position_m=world.tolist(),nominal_basket_base_position_m=local.tolist(),full_sphere_inside=bool(inside)))
        self.geometry_valid=self.geometry_valid and frame_ok
        if self.writing_geometry:
            policy=self.controller.robots[rid]
            row=dict(seconds=self.sample_seconds,robot=rid,evaluation_only=True,geometry_valid=frame_ok,
                source_stage=policy.stage,previous_stage=getattr(self,'previous_stages',{}).get(rid,'initial'),
                cycle=policy.cycles,base_position_m=base_p.tolist(),base_rotation=base_r.tolist(),
                basket_position_m=basket_p.tolist(),basket_rotation=basket_r.tolist(),q=q.tolist(),v=v.tolist(),
                position_fk_error_m=error_p,rotation_fk_error=error_r,contained_ids=contained,balls=measurements)
            shovel=self.shovel_links[rid]
            row['shovel_measured_evaluation_only']={'position_m':as_array(shovel.get_pos()).tolist(),'rotation':rotation_of(shovel).tolist(),'joints':as_array(robot.get_dofs_position(dofs_idx_local=self.shovel_dofs[rid])).tolist()}
            self.geometry_writer.write(json.dumps(row,allow_nan=False)+'\n');self.geometry_samples[rid]+=1
            if self.geometry_samples[rid]%120==0:self.geometry_writer.flush()
        return contained

    def close(self):
        if self.geometry_writer:self.geometry_writer.close()


    def snapshot(self,event):
        robot = self.robots[event['robot']]
        self.phase_snapshots.append({'event':event,
            'evaluation_ball_positions_m':[as_array(b.get_pos()).tolist() for b in self.balls],
            'evaluation_ball_in_active_robot_m':[local_position(robot,as_array(b.get_pos())).tolist() for b in self.balls],
            'robot_quaternion':as_array(robot.get_quat()).tolist()})

    def update(self, previous_stages):
        self.sample_seconds+=1/120;self.previous_stages=previous_stages;self.writing_geometry=True
        positions = {rid:as_array(r.get_pos()) for rid,r in self.robots.items()}
        for index,ball in enumerate(self.balls):
            self.maximum_ball_heights[index] = max(self.maximum_ball_heights[index],float(as_array(ball.get_pos())[2]))
        for rid,robot in self.robots.items():
            p = positions[rid]
            distance = float(np.linalg.norm((p-self.previous_positions[rid])[:2]))
            self.travel[rid] += distance
            if previous_stages[rid]=='fixed':
                self.fixed_travel[rid] += distance
            self.previous_positions[rid] = p.copy()
            q = as_array(robot.get_quat())
            tilt = math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))))
            self.maximum_tilt[rid] = max(self.maximum_tilt[rid],tilt)
            self.minimum_height[rid] = min(self.minimum_height[rid],float(p[2]))
            policy = self.controller.robots[rid]
            self.trial_violation |= not trial_position_allowed(p,policy.config)
            self.court_violation |= not court_position_allowed(p,policy.playing_area,'practice')
            contained = self._contained(rid,robot)
            if previous_stages[rid]=='retain':
                self.retention_windows[rid].append(contained)
            if policy.cycles > self.last_cycles[rid]:
                window = self.retention_windows[rid]
                common = sorted(set.intersection(*(set(row) for row in window))) if window else []
                self.cycle_holds[rid].append({'cycle':policy.cycles,'samples':len(window),
                                             'continuously_held_ball_indices':common})
                print(json.dumps({'evaluation_only':True,'robot':rid,'cycle':policy.cycles,
                                  'continuously_held_ball_indices':common}),flush=True)
                self.last_cycles[rid] = policy.cycles
                self.retention_windows[rid] = []
            self.retained_history[rid].append(contained)
        self.minimum_separation = min(self.minimum_separation,float(np.linalg.norm(positions['A'][:2]-positions['B'][:2])))

    def final(self):
        self.writing_geometry=False
        result = {}
        for rid,robot in self.robots.items():
            contained = self._contained(rid,robot)
            # Last 360 physics steps, after both robots finish, prove simultaneous retention.
            history = self.retained_history[rid][-360:]
            held = sorted(set.intersection(*(set(row) for row in history))) if len(history)==360 else []
            result[rid] = {'final_ball_indices':contained,'last_3_seconds_held_ball_indices':held,
                           'cycle_holds':self.cycle_holds[rid],
                           'maximum_tilt_degrees':self.maximum_tilt[rid],
                           'minimum_height_m':self.minimum_height[rid],
                           'travel_m':self.travel[rid], 'fixed_guided_travel_m':self.fixed_travel[rid],
                           'final_position_m':as_array(robot.get_pos()).tolist()}
        return result


def run(scene,robots_list,balls,onboard_list,fixed_cameras,detail_cameras,overview,top,config,vision,fixed,mission,output,record_video,basket_list=None):
    output.mkdir(parents=True,exist_ok=True)
    (output/'mission_check.json').write_text('{"verified":false,"status":"running"}\n')
    ids = list(mission['trial_zones'])
    robots, onboard = dict(zip(ids,robots_list)), dict(zip(ids,onboard_list))
    hold_fleet(robots_list)
    lift={rid:LiftStorage(r) for rid,r in robots.items()} if mission.get("robot_variant")=="lift" else {}
    lift_trace=[]
    for _ in range(120):
        scene.step()
    control = FleetCoordinator(mission,config['playing_area_m'])
    evaluator = PhysicalEvaluator(robots,balls,control,output,mission.get('robot_variant')=='lift')
    if mission.get('scoped_ball_search',False):fixed['ball_search_zones_m']=list(mission['trial_zones'].values())
    fixed_detector = FixedBallDetector(ROOT/vision['model'],vision,fixed)
    ego_detector = BallDetector(ROOT/vision['model'],vision)
    basket = dict(zip(ids,basket_list)) if basket_list else {}
    basket_config = mission.get('basket_confirmation_config')
    basket_detector = BasketDetector(BallDetector(ROOT/vision['model'],vision),basket_config) if basket else None
    recovery_detector = BallDetector(ROOT/vision['model'],vision)
    asynchronous = bool(mission.get('asynchronous_perception',False))
    workers = InferenceWorkers(('person','fixed','onboard','basket')) if asynchronous else None
    # Separate model instances: a long ball search cannot delay publication of
    # a completed safety frame, or mutate another worker's YOLO predictor.
    person_detector=FixedBallDetector(ROOT/vision['model'],vision,fixed) if asynchronous and fixed.get('person_safety_enabled') else None
    wheel_dofs = {rid:[r.get_joint(name).dofs_idx_local[0] for name in ('left_wheel_joint','right_wheel_joint')]
                  for rid,r in robots.items()}
    shovel_dofs = {rid:[r.get_joint(name).dofs_idx_local[0] for name in ('shovel_lift_joint','shovel_tilt_joint')]
                   for rid,r in robots.items()}
    calibration = [{**spec,'intrinsics':np.asarray(camera.intrinsics).tolist(),
                    'transform':np.asarray(camera.transform).tolist()} for spec,camera in zip(fixed['cameras'],fixed_cameras)]
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    safety_type = PersonPauseGate if mission.get('automatic_person_resume',False) else PersonStopGate
    safety = {rid:safety_type(mission.get('person_safety',{}),[s['id'] for s in calibration]) for rid in ids} if fixed.get('person_safety_enabled') else {}
    safety_joints = {}
    fixed_images,ego_images,basket_images = {},{},{}
    fixed_time = 0.
    fixed_trace, onboard_trace, trace = [],[],[]
    person_trace=[]
    fixed_capture=SameStepCapture()
    basket_trace = []
    guard_stops = {rid:0 for rid in ids}
    last_events = {rid:0 for rid in ids}
    steps,frames = 0,0
    font_path = system_font_path()
    font,small = ImageFont.truetype(font_path,18),ImageFont.truetype(font_path,14)
    writer = av.open(str(output/'fleet_mission.mp4'),'w',options={'movflags':'+faststart'}) if record_video else None
    stream = None
    if writer:
        stream = writer.add_stream('libx264',rate=12)
        stream.width,stream.height,stream.pix_fmt = 1280,720,'yuv420p'
        stream.options = {'crf':'21','preset':'fast'}
    started = time.perf_counter()

    def positions():
        return {rid:as_array(r.get_pos()) for rid,r in robots.items()}

    def capture_onboard(rid):
        rgb=onboard[rid].capture().copy()
        p,rotation=as_array(robots[rid].get_pos()),rotation_of(robots[rid])
        camera=freeze_camera(onboard[rid].camera,p,rotation)
        return dict(rid=rid,rgb=rgb,camera=camera,stage=control.robots[rid].stage)

    def detect_onboard(packet,detector=None):
        obs,image=(detector or ego_detector).detect(packet['rgb'],packet['camera'],config['ball_radius_m'])
        return {**packet,'observations':obs,'image':image}

    def apply_onboard(packet,captured,feed_control=True):
        rid=packet['rid'];obs=packet['observations'];camera=packet['camera']
        ego_images[rid]=packet['image']
        # Reproject capture-time coordinates into the CURRENT robot frame.
        # Freshness still uses the ORIGINAL capture time, never arrival time.
        p,rotation=as_array(robots[rid].get_pos()),rotation_of(robots[rid])
        world=[camera.position+camera.rotation@np.asarray(o.position_in_robot_m) for o in obs]
        current=[replace(o,position_in_robot_m=(rotation.T@(w-p)).tolist()) for o,w in zip(obs,world)]
        if feed_control:
            control.robots[rid].observe_onboard(current,p,rotation,captured)
        onboard_trace.append(dict(seconds=captured,applied_seconds=steps/120,robot=rid,
            stage=packet['stage'],fed_to_controller=feed_control,observations=[asdict(o) for o in obs]))
        if captured==0:Image.fromarray(packet['image']).save(output/f'initial_onboard_{rid}.png')
        return [dict(position_m=w.tolist(),confidence=o.confidence,bbox_xyxy=o.bbox_xyxy,
                     camera_id='onboard-'+rid,seconds=captured) for o,w in zip(obs,world)]

    def sense_onboard(rid,feed_control=True):
        return apply_onboard(detect_onboard(capture_onboard(rid)),steps/120,feed_control)

    def capture_fixed_views():
        return fixed_capture.get(steps,lambda:((c.render(antialiasing=True)[0],item)
                                              for c,item in zip(fixed_cameras,calibration)))

    def capture_fixed():
        return dict(views=capture_fixed_views(),
            recovery=[capture_onboard(rid) for rid in ids if control.robots[rid].stage=='recover_wait'],
            positions=positions())

    def detect_fixed(packet):
        return dict(views=[(rgb,item,fixed_detector.detect(rgb,item,config['ball_radius_m'],tiled=True,include_people=not asynchronous))
                           for rgb,item in packet['views']],positions=packet['positions'],
                    recovery=[detect_onboard(p,recovery_detector) for p in packet['recovery']])

    def capture_people():
        return capture_fixed_views()

    def detect_people(views):
        return [(rgb,item,person_detector.detect(rgb,item,config['ball_radius_m'],tiled=True,include_balls=False)) for rgb,item in views]

    def apply_people(views,captured):
        if fixed.get('person_dual_view_localization',False):
            views=refine_person_views(views)
        for rgb,item,detected in views:
            if detected['raw_person_detection_count'] and sum(1 for _ in output.glob(f"person_{item['id']}_*_rgb.png"))<8:
                Image.fromarray(rgb).save(output/f"person_{item['id']}_{round(captured*120):06d}_rgb.png")
            for gate in safety.values():gate.observe(item['id'],detected['persons'],captured,detected['unlocalized_person_count'])
            person_trace.append(dict(seconds=captured,applied_seconds=steps/120,camera_id=item['id'],**detected))

    def apply_fixed(packet,captured):
        nonlocal fixed_time
        estimated,views=[],[]
        if not asynchronous:apply_people(packet['views'],captured)
        for rgb,item,detected in packet['views']:
            estimated.extend(detected['observations']);views.append({'camera_id':item['id'],**detected})
            fixed_images[item['id']]=np.asarray(annotate(rgb,detected['observations']))
            if captured==0:
                Image.fromarray(rgb).save(output/f"initial_fixed_{item['id']}_rgb.png")
                Image.fromarray(fixed_images[item['id']]).save(output/f"initial_fixed_{item['id']}_yolo.jpg",quality=94)
        merged=merge_ball_estimates(estimated,fixed['cross_camera_merge_distance_m'])
        recovery_views={}
        for item in packet['recovery']:
            recovery_views['onboard-'+item['rid']]=apply_onboard(item,captured,False)
        if recovery_views:merged=corroborate_fixed_with_onboard(merged,recovery_views)
        # Exclusion uses capture-time poses. Delayed frames cannot assign tasks
        # or advance recovery if they exceed the existing observation timeout.
        age=steps/120-captured
        accepted=0<=age<=min(mission['fixed_observation_timeout_seconds'],mission['recovery_config']['fixed_max_age_seconds'])
        if accepted:control.observe_fixed(merged,packet['positions'],captured,received_seconds=steps/120,current_positions=positions())
        fixed_time=captured
        fixed_trace.append(dict(seconds=captured,applied_seconds=steps/120,accepted=accepted,views=views,
                                recovery_onboard_views=recovery_views,merged_estimates=merged))

    def sense_fixed():apply_fixed(detect_fixed(capture_fixed()),steps/120)

    def capture_basket(rid):
        rgb=basket[rid].capture().copy()
        camera=freeze_camera(basket[rid].camera,as_array(robots[rid].get_pos()),rotation_of(robots[rid]),
                             basket_config['ball_center_plane_in_robot_m'])
        joints=as_array(robots[rid].get_dofs_position(dofs_idx_local=shovel_dofs[rid]))
        return dict(rid=rid,rgb=rgb,camera=camera,joints=joints,stage=control.robots[rid].stage)

    def detect_basket(packet):
        observed,image=basket_detector.detect(packet['rgb'],packet['camera'])
        return {**packet,'observed':observed,'image':image}

    def apply_basket(packet,captured):
        rid=packet['rid'];observed=packet['observed'];actual_joints=packet['joints'];stage=packet['stage']
        basket_images[rid]=packet['image']
        lowered=bool(np.max(np.abs(actual_joints-np.array([-.448,.448])))<=.04)
        eligible=('seek','fixed','brake','visual','retain','confirm','recover_backoff','recover_wait','done','fault','return_home','await_unload')
        clearance=basket_view_clearance(actual_joints)
        valid=bool(observed['valid'] and clearance['clear'] and stage in eligible
            and control.robots[rid].stage in eligible
            and 0<=steps/120-captured<=basket_config['frame_max_age_seconds']
            and (rid not in lift or lift[rid].observation()['settled']))
        # A frame captured before retain must not populate its new evidence window.
        attempt=control.robots[rid].basket_confirmation.attempt
        if attempt and attempt['confirmation_started_seconds'] is not None and captured<attempt['confirmation_started_seconds']:
            valid=False
        control.robots[rid].observe_basket(observed['observed_count'],captured,valid)
        basket_trace.append(dict(seconds=captured,applied_seconds=steps/120,robot=rid,stage=stage,**observed,
            valid_for_confirmation=valid,own_shovel_lowered=lowered,actual_shovel_joint_angles_rad=actual_joints.tolist(),
            shovel_view_clearance=clearance,stable_observed_count=control.robots[rid].basket_confirmation.stable_count(steps/120)))
        folder=output/'basket_rgb';folder.mkdir(exist_ok=True)
        Image.fromarray(packet['rgb']).save(folder/f'{rid}_{round(captured*120):06d}.png')
        if captured==0:Image.fromarray(packet['image']).save(output/f'initial_basket_{rid}.png')

    def sense_basket(rid):apply_basket(detect_basket(capture_basket(rid)),steps/120)

    def sense_async(now):
        for channel,captured,packets in workers.poll(now):
            if channel=='person':apply_people(packets,captured)
            elif channel=='fixed':apply_fixed(packets,captured)
            else:
                for packet in packets:
                    (apply_onboard if channel=='onboard' else apply_basket)(packet,captured)
        if person_detector and workers.idle('person') and now>=async_due['person']:
            views=capture_people()
            workers.submit('person',now,detect_people,views,capture_started_wall=fixed_capture.started_wall)
            async_due['person']=now+.15
        if workers.idle('fixed') and now>=async_due['fixed']:
            packet=capture_fixed()
            workers.submit('fixed',now,detect_fixed,packet,capture_started_wall=fixed_capture.started_wall)
            async_due['fixed']=now+mission['fixed_update_period_seconds']
        if workers.idle('onboard') and now>=async_due['onboard']:
            needed=[rid for rid in ids if control.robots[rid].stage in ('brake','visual','retry') or
                (control.robots[rid].stage=='scoop' and control.robots[rid].scoop_lift_started is None and mission.get('scoop_require_rgb_seating',False)) or
                (control.robots[rid].stage=='fixed' and control.robots[rid].target is not None and
                 np.linalg.norm((control.robots[rid].target-as_array(robots[rid].get_pos()))[:2])<=1.10)]
            if writer and not needed:needed=ids
            if needed:
                captured_wall=time.perf_counter()
                packets=[capture_onboard(rid) for rid in needed]
                workers.submit('onboard',now,lambda items:[detect_onboard(p) for p in items],packets,capture_started_wall=captured_wall)
                async_due['onboard']=now+.05
        if basket and workers.idle('basket') and now>=async_due['basket']:
            captured_wall=time.perf_counter()
            packets=[capture_basket(rid) for rid in ids]
            workers.submit('basket',now,lambda items:[detect_basket(p) for p in items],packets,capture_started_wall=captured_wall)
            async_due['basket']=now+basket_config['update_period_seconds']

    def panel():
        canvas = Image.new('RGB',(1280,720),'#10232c')
        draw = ImageDraw.Draw(canvas)
        for index,rid in enumerate(ids):
            p,rotation = as_array(robots[rid].get_pos()),rotation_of(robots[rid])
            sign = 1 if rid=='A' else -1
            detail_cameras[index].set_pose(pos=p+np.array([sign*1.5,-sign*1.55,1.25]),
                                          lookat=p+rotation@np.array([.10,0,.10]))
            x = index*640
            canvas.paste(Image.fromarray(np.asarray(detail_cameras[index].render()[0],dtype=np.uint8)),(x,58))
            canvas.paste(Image.fromarray(ego_images[rid]).resize((320,240)),(x,455))
            if rid in basket_images:
                canvas.paste(Image.fromarray(basket_images[rid]).resize((320,240)),(x+320,455))
            else:
                canvas.paste(Image.fromarray(fixed_images[rid]).resize((320,180)),(x+320,455))
            robot = control.robots[rid]
            draw.text((x+18,14),f'{rid} / {LABELS[robot.stage]}  確認 {robot.confirmed_cycles}/{mission["cycles_per_robot"]}・動作 {robot.cycles}',font=font,fill='#d8f6bb')
            draw.text((x+12,425),'車載YOLO（直近の検出画像）',font=small,fill='#b9cbd0')
            if rid in basket_images:
                status=robot.basket_confirmation.stable_count(steps/120)
                draw.text((x+330,425),f'カゴYOLO / 安定検出 {status if status is not None else "不明"}球',font=small,fill='#b9cbd0')
            else:
                draw.text((x+330,425),f'固定{rid} / 検出時刻 {fixed_time:.1f}秒',font=small,fill='#b9cbd0')
            draw.text((x+330,699),f'シミュレーション {steps/120:.2f} 秒 / 等速',font=small,fill='#d8f6bb')
        return np.asarray(canvas)

    def collect_events():
        for rid,robot in control.robots.items():
            for event in robot.events[last_events[rid]:]:
                print(json.dumps(event),flush=True)
                evaluator.snapshot(event)
                if event['to'] in ('brake','scoop','lift','retain','seek','recover_backoff','recover_wait','done','fault'):
                    Image.fromarray(ego_images[rid]).save(output/f"{rid}_cycle{event['cycle']}_{event['to']}_onboard.png")
                    if rid in basket_images:
                        Image.fromarray(basket_images[rid]).save(output/f"{rid}_cycle{event['cycle']}_{event['to']}_basket.png")
            last_events[rid] = len(robot.events)

    try:
        if asynchronous:
            apply_fixed(workers.warm('fixed',detect_fixed,capture_fixed()),0.)
            if person_detector:apply_people(workers.warm('person',detect_people,capture_people()),0.)
        else:sense_fixed()
        for rid in ids:
            if asynchronous:
                apply_onboard(workers.warm('onboard',detect_onboard,capture_onboard(rid)),0.,False)
                if rid in basket:apply_basket(workers.warm('basket',detect_basket,capture_basket(rid)),0.)
            else:
                sense_onboard(rid,feed_control=False)
                if rid in basket:sense_basket(rid)
        save_rgb(top,output/'initial_top.png')
        save_rgb(overview,output/'initial_overview.png')
        final_hold_started = None
        next_fixed = mission['fixed_update_period_seconds']
        next_basket = basket_config['update_period_seconds'] if basket else math.inf
        async_due=dict(person=.05,fixed=.05,onboard=.05,basket=basket_config['update_period_seconds'] if basket else math.inf)
        while steps < round(mission['maximum_mission_seconds']*120):
            now = steps/120
            if asynchronous:
                sense_async(now)
            else:
                needs_fixed = bool(safety) or control.tracker is not None or any(r.stage in ('seek','fixed','retry') for r in control.robots.values())
                if needs_fixed and now >= next_fixed-1e-8:
                    sense_fixed()
                    next_fixed = now+mission['fixed_update_period_seconds']
                for rid in ids:
                    stage = control.robots[rid].stage
                    agent = control.robots[rid]
                    # Skip off-frustum onboard renders during fixed-only travel.
                    # This gate uses the image-derived target, not simulator ball state.
                    near = agent.target is not None and np.linalg.norm((agent.target-as_array(robots[rid].get_pos()))[:2]) <= 1.10
                    seating=stage=='scoop' and agent.scoop_lift_started is None and mission.get('scoop_require_rgb_seating',False)
                    if ((stage in ('brake','visual','retry') or seating) and steps%6==0) or (stage=='fixed' and near and steps%6==0):
                        sense_onboard(rid)
                    elif writer and steps%20==0:
                        sense_onboard(rid,feed_control=False)
                if basket and now >= next_basket-1e-8:
                    for rid in ids:sense_basket(rid)
                    next_basket = now+basket_config['update_period_seconds']
            control.guard_separation(positions(),now)
            previous_stages = {rid:r.stage for rid,r in control.robots.items()}
            for rid,robot in control.robots.items():
                p,rotation = as_array(robots[rid].get_pos()),rotation_of(robots[rid])
                # Fault commands keep proprioception current without advancing
                # pickup joints. The latched safety pose is held below.
                wheels,joints = robot.command(p,rotation,now)
                if rid in lift:
                    storage=lift[rid].check(now)
                    lift_trace.append(dict(seconds=now,robot=rid,**storage))
                    if storage['fault']:
                        if lift[rid].shovel_hold is None:
                            lift[rid].shovel_hold=as_array(robots[rid].get_dofs_position(dofs_idx_local=shovel_dofs[rid])).copy()
                        robot.stop(now,storage['fault']);wheels=np.zeros(2)
                        robot.joints=lift[rid].shovel_hold.copy();joints=robot.joints
                forward = float(wheels.mean())*.11
                blocked=rid in safety and safety[rid].blocked(p,robot.filtered_forward_speed,forward,now)
                if blocked:
                    if rid not in safety_joints:
                        safety_joints[rid]=as_array(robots[rid].get_dofs_position(dofs_idx_local=shovel_dofs[rid])).copy()
                    # A proximity pause can become a camera/feedback fault later.
                    # Re-evaluate its reason even while the measured pose is held.
                    resumable=mission.get('automatic_person_resume',False) and safety[rid].event['reason']=='person_proximity_stop' and robot.stage in ('seek','fixed','brake','visual','recover_wait','return_home','await_unload','done')
                    if resumable:robot.pause(now,safety_joints[rid])
                    else:
                        robot.stop(now,safety[rid].event['reason'])
                        robot.joints=safety_joints[rid].copy()
                    wheels,joints=np.zeros(2),safety_joints[rid];forward=0.
                elif rid in safety_joints:
                    robot.resume(now);del safety_joints[rid]
                    wheels=np.zeros(2);forward=0.
                forecast = p+rotation@np.array([.15 if forward>0 else -.15 if forward<0 else 0,0,0])
                if not trial_position_allowed(forecast,robot.config) or not court_position_allowed(forecast,config['playing_area_m'],'practice'):
                    if np.any(np.abs(wheels)>1e-8):
                        guard_stops[rid] += 1
                        robot.stop(now,'braking_boundary_guard')
                    wheels = np.zeros(2)
                robots[rid].control_dofs_velocity(wheels,dofs_idx_local=wheel_dofs[rid])
                robots[rid].control_dofs_position(joints,dofs_idx_local=shovel_dofs[rid])
                if steps%6==0:
                    trace.append({'seconds':now,'robot':rid,'stage':robot.stage,'status':robot.last_status,
                                  'wheel_commands':wheels.tolist(),'shovel_joint_targets_rad':joints.tolist(),
                                  'filtered_own_speed_m_s':robot.filtered_forward_speed,
                                  'person_stop_latched':rid in safety_joints,
                                  'travel_speed_mode':None if robot.navigator is None else
                                      'fast' if getattr(robot.navigator,'fast_route',None) else 'legacy_approach',
                                  'robot_position_m':p.tolist(),
                                  'robot_yaw_rad':float(np.arctan2(rotation[1,0],rotation[0,0])),
                                  'target_m':None if robot.target is None else robot.target.tolist()})
            scene.step()
            steps += 1
            if steps%600==0:
                print(json.dumps({'progress_seconds':steps/120,'robot_stages':{rid:r.stage for rid,r in control.robots.items()}}),flush=True)
            evaluator.update(previous_stages)
            collect_events()
            if writer and steps%10==0:
                for packet in stream.encode(av.VideoFrame.from_ndarray(panel(),format='rgb24')):
                    writer.mux(packet)
                frames += 1
            if control.finished:
                if final_hold_started is None:
                    final_hold_started = steps
                if steps-final_hold_started >= 360:
                    break
        Image.fromarray(panel()).save(output/'final_panel.png')
        save_rgb(top,output/'final_top.png')
        for index,rid in enumerate(ids):
            p,rotation = as_array(robots[rid].get_pos()),rotation_of(robots[rid])
            detail_cameras[index].set_pose(pos=p+np.array([.65,-.72,.95]),lookat=p+rotation@np.array([-.04,0,.12]))
            save_rgb(detail_cameras[index],output/f'final_robot_{rid}.png')
            if rid in basket_images:Image.fromarray(basket_images[rid]).save(output/f'final_basket_{rid}.png')
    finally:
        evaluator.close()
        if workers:workers.close()
        if writer:
            for packet in stream.encode():
                writer.mux(packet)
            writer.close()
    physics = evaluator.final()
    all_held = [i for report in physics.values() for i in report['last_3_seconds_held_ball_indices']]
    goal=mission['cycles_per_robot']
    terminal=('done','await_unload') if mission.get('end_trial_when_full') else ('done',)
    checks = {
        'actual_basket_geometry_valid_every_sample':evaluator.geometry_valid,
        'two_robots_each_completed_two_cycles':all(r.cycles==goal and r.stage in terminal for r in control.robots.values()),
        'four_unique_image_derived_assignments':len(control.assignments)==2*goal and all(
            np.linalg.norm(np.array(a['image_target_m'][:2])-b['image_target_m'][:2])>.3
            for i,a in enumerate(control.assignments) for b in control.assignments[i+1:]),
        'both_robots_used_fixed_and_onboard':all(sum(e['to']=='brake' for e in r.events)==goal and sum(e['to']=='scoop' for e in r.events)==goal
                                                for r in control.robots.values()),
        'both_robots_physically_drove':all(item['travel_m']>.5 and item['fixed_guided_travel_m']>.1 for item in physics.values()),
        'two_balls_per_basket_held_for_final_3_seconds':all(len(item['last_3_seconds_held_ball_indices'])==goal for item in physics.values()),
        'all_four_balls_in_unique_baskets':sorted(all_held)==list(range(len(balls))),
        'each_cycle_had_3_second_basket_retention':all(len(item['cycle_holds'])==goal and all(
            cycle['samples']>=359 and len(cycle['continuously_held_ball_indices'])>=cycle['cycle'] for cycle in item['cycle_holds'])
            for item in physics.values()),
        'all_four_balls_lifted_above_0_4m':all(z>.4 for z in evaluator.maximum_ball_heights),
        'both_robots_stable':all(item['maximum_tilt_degrees']<8 and item['minimum_height_m']>.17 for item in physics.values()),
        'temporary_trial_boundaries_respected':not evaluator.trial_violation,
        'practice_court_entry_rule_respected':not evaluator.court_violation,
        'robot_separation_respected':evaluator.minimum_separation>=mission['minimum_robot_separation_m'],
        'no_actuator_guard_interventions':all(n==0 for n in guard_stops.values()),
    }
    if goal!=2:
        renames={'two_robots_each_completed_two_cycles':'both_robots_completed_goal_cycles',
            'four_unique_image_derived_assignments':'all_goal_assignments_unique',
            'two_balls_per_basket_held_for_final_3_seconds':'goal_balls_per_basket_held_for_final_3_seconds',
            'all_four_balls_in_unique_baskets':'all_test_balls_in_unique_baskets',
            'all_four_balls_lifted_above_0_4m':'all_test_balls_lifted_above_0_4m'}
    else:renames={}
    if lift:
        checks['new_lift_all_four_axes_held_at_home']=bool(lift_trace) and all(e['home'] for e in lift_trace)
        checks['no_lift_storage_faults']=all(h.fault is None for h in lift.values())
    basket_check = None
    if control.tracker is not None:
        # Retain failed-attempt evidence; successful goals count only confirmed
        # RGB increases. Ground truth is used here after control has finished.
        del checks['two_robots_each_completed_two_cycles']
        del checks['each_cycle_had_3_second_basket_retention']
        checks['two_robots_each_confirmed_two_collections']=all(r.confirmed_cycles==goal and r.stage in terminal for r in control.robots.values())
        checks['both_robots_used_fixed_and_onboard']=all(sum(e['to']=='brake' for e in r.events)>=goal and sum(e['to']=='scoop' for e in r.events)==r.cycles for r in control.robots.values())
        confirmed_retention=[];decision_agreement=[]
        for rid,robot in control.robots.items():
            prior=set()
            holds={c['cycle']:c for c in physics[rid]['cycle_holds']}
            for decision in robot.basket_confirmation.records:
                hold=holds[decision['cycle']];held=set(hold['continuously_held_ball_indices'])
                positive=decision['status']=='confirmed_increase'
                physically_increased=len(held-prior)==1 and not prior-held
                decision_agreement.append(positive==physically_increased)
                if positive:confirmed_retention.append(hold['samples']>=359 and physically_increased)
                prior=held
        checks['each_confirmed_collection_had_3_second_basket_retention']=len(confirmed_retention)==2*goal and all(confirmed_retention)
        checks['image_decisions_match_independent_physical_increases']=all(decision_agreement)
    if goal!=2:renames['two_robots_each_confirmed_two_collections']='both_robots_confirmed_goal_collections'
    checks={renames.get(k,k):v for k,v in checks.items()}
    if basket:
        basket_check = dict(required=mission['basket_confirmation_required'],camera_count=len(basket),
            confirmed_cycles={rid:r.confirmed_cycles for rid,r in control.robots.items()},
            cycles={rid:r.basket_confirmation.records for rid,r in control.robots.items()},
            all_attempts_confirmed=all(len(r.basket_confirmation.records)==r.cycles and all(
                c['status']=='confirmed_increase' for c in r.basket_confirmation.records) for r in control.robots.values()),
            physical_mount_mass_and_collisions_modelled=False)
    video = None
    if record_video:
        with av.open(str(output/'fleet_mission.mp4')) as reader:
            decoded = sum(1 for _ in reader.decode(video=0))
        assert decoded==frames
        video = {'frames':frames,'fps':12,'seconds':frames/12,'playback_speed':1.0}
    report = {'verified':all(checks.values()),'checks':checks,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
              'scope':mission['operation_scope'],'operating_area_status':'pending_user_decision',
              'scenario':mission.get('scenario_name','straight_development_trial'),
              'robot_variant':mission.get('robot_variant','stock'),
              'navigation_control':{'fast_travel_minimum_distance_m':mission.get('fast_travel_minimum_distance_m'),
                                    'fast_travel_end_distance_m':mission.get('fast_travel_end_distance_m'),
                                    'travel_speed_profile':mission.get('travel_speed_profile','legacy'),
                                    'acceleration_m_s2':mission.get('travel_acceleration_m_s2'),
                                    'speed_feedback_gain':mission.get('travel_speed_feedback_gain'),
                                    'slow_speed_distance_m':mission.get('travel_slow_speed_distance_m'),
                                    'cruise_speed_m_s':mission.get('cruise_speed_m_s',.22),
                                    'maximum_travel_command_m_s':mission.get('maximum_travel_command_m_s',.45),
                                    'travel_slowdown_distance_m':mission.get('travel_slowdown_distance_m'),
                                    'travel_full_cruise_distance_m':mission.get('travel_full_cruise_distance_m'),
                                    'near_handoff_speed_m_s':mission.get('near_handoff_speed_m_s'),
                                    'brake_reaction_seconds':mission.get('travel_brake_reaction_seconds',.15),
                                    'brake_deceleration_m_s2':mission.get('travel_brake_deceleration_m_s2',.5)},
              'pickup_control':{'progress_source':'robot_forward_displacement' if mission.get('scoop_distance_m') else 'elapsed_time',
                                'scoop_distance_m':mission.get('scoop_distance_m'),
                                'scoop_preinsert_m':mission.get('scoop_preinsert_m',0.),
                                'rgb_seating_required':mission.get('scoop_require_rgb_seating',False),
                                'rgb_seating_events':{rid:r.seating_events for rid,r in control.robots.items()},
                                'own_pitch_pickup_targets':{rid:r.pickup_pitch_records for rid,r in control.robots.items()},
                                'own_pitch_scope':'Own rotation only;arm and absolute bucket pitch compensated before seating within original limits/rate. Start actual lift from seated target;no ball truth input.',
                                'rgb_seating_observations':{rid:r.seating_observation_events for rid,r in control.robots.items()},
                                'rgb_seating_keep_tolerance_m':.005,
                                'rgb_seating_region_m':{'minimum_x':.39,'maximum_x':.530,'maximum_abs_y':.070},
                                'rgb_motion_dropout_max_age_seconds':.15,
                                'rgb_motion_dropout_resets_lift_evidence':True,
                                'pickup_forward_speed_m_s':mission.get('pickup_forward_speed_m_s'),
                                'scoop_insertion_speed_m_s':mission.get('scoop_insertion_speed_m_s',mission.get('pickup_forward_speed_m_s')),
                                'requested_scoop_insertion_speed_m_s':mission.get('scoop_insertion_speed_m_s',mission.get('pickup_forward_speed_m_s')),
                                'precision_insertion_policy':'Original case insertion/slew and visual approach restored;own-pitch compensation retained;RGB/travel/time/final success guards unchanged',
                                'scoop_insertion_acceleration_m_s2':mission.get('scoop_insertion_acceleration_m_s2'),
                                'requested_scoop_insertion_acceleration_m_s2':mission.get('scoop_insertion_acceleration_m_s2'),
                                'minimum_scoop_seconds':mission.get('minimum_scoop_seconds',0.),
                                'maximum_scoop_seconds':mission.get('maximum_scoop_seconds',1.5)},
              'trial_zones':mission['trial_zones'],
              'controller_uses_simulator_ball_state':False,'ball_state_modified_during_episode':False,
              'ball_attachment_used':False,'pickup_confirmation_in_controller':'basket_rgb_stable_count_increase' if mission.get('basket_confirmation_required') else 'mechanism_cycle_only_not_a_basket_sensor',
              'basket_confirmation':basket_check,
              'collection_service':dict(enabled=mission.get('collection_service_enabled',False),capacity=mission.get('basket_capacity',2),full_behavior=mission.get('full_basket_behavior'),visits={rid:r.service_visits for rid,r in control.robots.items()},final_stages={rid:r.stage for rid,r in control.robots.items()}),
              'robot_pose_source':'simulator proprioception','robot_count':2,'fixed_camera_count':2,
              'simulated_seconds':steps/120,'wall_clock_seconds':time.perf_counter()-started,
              'perception_timing':dict(asynchronous=asynchronous,initial_warmup_paused=True,real_time_guaranteed=False,
                  person_inference_separate_from_ball_search=bool(person_detector),warmup_on_runtime_workers=asynchronous,
                  completed_inferences=len(workers.records) if workers else 0,physics_advanced_during_inference=sum(r['simulation_age_seconds']>0 for r in workers.records) if workers else 0),
              'physical_evaluation_frame':'actual_authored_basket_link_full_sphere' if lift else 'legacy_stock_base_frame',
              'actual_basket_geometry_samples':evaluator.geometry_samples,
              'physics':physics,'maximum_ball_heights_m':evaluator.maximum_ball_heights,
              'minimum_robot_separation_m':evaluator.minimum_separation,'guard_stops':guard_stops,'person_safety':{'enabled':bool(safety),'parameters':mission.get('person_safety'),'stop_events':{rid:g.event for rid,g in safety.items()},'automatic_resume':mission.get('automatic_person_resume',False),'events':{rid:getattr(g,'events',[]) for rid,g in safety.items()}},
              'assignments':control.assignments,'events':[e for r in control.robots.values() for e in r.events],
              'evaluation_phase_snapshots':evaluator.phase_snapshots,
              'retries':{rid:r.retry_total for rid,r in control.robots.items()},'video':video,
              'recollection':dict(enabled=control.tracker is not None,settings=mission.get('recovery_config'),
                  attempts={rid:r.cycles for rid,r in control.robots.items()},
                  recollection_requests={rid:r.recollection_total for rid,r in control.robots.items()},
                  accepted_candidates=control.recoveries,identity_after_occlusion_proven=False),
              'test_actuator_fault':dict(empty_scoop_attempts=mission.get('test_empty_scoop_attempts',0),
                  mode='drive_and_shovel_commands_held_at_rest',ball_state_changed=False),
              'model_sha256':hashlib.sha256((ROOT/vision['model']).read_bytes()).hexdigest(),
              'person_detection':dict(model=fixed_detector.person_weights_path.name,
                  model_sha256=hashlib.sha256(fixed_detector.person_weights_path.read_bytes()).hexdigest(),
                  separate_model=fixed_detector.person_model is not fixed_detector.model,
                  confidence=fixed['person_confidence'],tile_size_pixels=fixed['person_tile_size_pixels'],
                  full_image_coverage=True)}
    if lift:(output/'lift-storage-trace.json').write_text(json.dumps(lift_trace,indent=2)+'\n')
    if workers:(output/'perception_timing.json').write_text(json.dumps(workers.records,indent=2)+'\n')
    (output/'fixed_capture_timing.json').write_text(json.dumps(fixed_capture.records,indent=2)+'\n')
    for name,data in [('mission_check.json',report),('fixed_observations.json',fixed_trace),('person_observations.json',person_trace),('onboard_observations.json',onboard_trace),('control_trace.json',trace),('basket_observations.json',basket_trace),('tracking_observations.json',control.tracking_frames)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'verified':report['verified'],'checks':checks,'seconds':steps/120,'physics':physics}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['amdgpu','cpu'],default='amdgpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'mission')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--scenario',choices=['turn','straight'],default='turn')
    parser.add_argument('--robot-variant',choices=['stock','wedge','beveled','lift'],default='stock')
    parser.add_argument('--case-config',type=Path,help='Initialization/configuration for a repeatable evaluation case.')
    parser.add_argument('--basket-confirmation-mode',choices=['require','observe'],default='require')
    parser.add_argument('--recollection-mode',choices=['auto','stop'],default='auto')
    parser.add_argument('--test-empty-scoop-attempts',type=int,choices=range(4),default=0,
        help='Explicit actuator-omission fixture: hold drive and shovel at rest during these initial attempts.')
    args = parser.parse_args()
    config,vision,fleet,fixed,mission = [load(name) for name in ('scene_config.json','vision_config.json','fleet_config.json','fixed_camera_config.json',
        'mission_config.json' if args.scenario=='turn' else 'mission_straight_config.json')]
    if args.case_config:
        mission = json.loads(args.case_config.read_text(encoding='utf-8'))
    mission['scenario_name'] = args.case_config.stem if args.case_config else args.scenario
    mission['robot_variant'] = args.robot_variant
    mission['basket_confirmation_config'] = load('basket_camera_config.json')
    mission['basket_confirmation_config']['maximum_count']=mission.get('basket_capacity',2)
    mission['basket_confirmation_required'] = args.basket_confirmation_mode == 'require'
    mission['recovery_config']=load('recovery_config.json')
    mission['recovery_config']['enabled']=args.recollection_mode=='auto'
    mission['perception_floor_size_m']=config['court_size_m']
    mission['test_empty_scoop_attempts']=args.test_empty_scoop_attempts
    if args.robot_variant=='wedge':
        config['robot_urdf'] = 'collector_wedge.urdf'
    elif args.robot_variant=='beveled':
        config['robot_urdf'] = 'collector_beveled.urdf'
    if args.robot_variant=='lift':config['robot_urdf']='collector_basket_lift.urdf'
    config['physics_substeps'] = vision['physics_substeps']
    # Scenario coordinates are initialization-only and removed before control creation.
    config['ball_positions_m'] = mission.pop('test_ball_positions_m')
    initial_robots=mission.pop('test_robot_positions_m',{})
    if initial_robots:
        if set(initial_robots)!={s['id'] for s in fleet['robots']}:raise ValueError('All robot starts are required.')
        for spec in fleet['robots']:
            p=np.asarray(initial_robots[spec['id']],dtype=float)
            local={**mission,'trial_floor_bounds_m':mission['trial_zones'][spec['id']]}
            if p.shape!=(3,) or not np.isfinite(p).all() or not trial_position_allowed(p,local) or not court_position_allowed(p,config['playing_area_m'],'practice'):
                raise ValueError('Robot start lies outside its existing temporary trial zone.')
            spec['position_m']=p.tolist()
    gs.init(backend=getattr(gs,args.backend),seed=13,logging_level='warning')
    try:
        scene,robots,balls,overview,top,detail,onboard = build_fleet(config,fleet,vision)
        basket_cameras = [BasketCamera(scene,r,mission['basket_confirmation_config'],spec['position_m'])
                          for r,spec in zip(robots,fleet['robots'])]
        fixed_cameras = [scene.add_camera(res=tuple(s['resolution']),pos=tuple(s['position_m']),
            lookat=tuple(s['lookat_m']),up=(0,0,1),fov=s['vertical_fov_degrees'],near=.05,far=200,GUI=False)
            for s in fixed['cameras']]
        detail_cameras = [scene.add_camera(res=(640,360),pos=(-12,-6,2),lookat=(-13.6,-6.5,.25),
            fov=55,near=.01,far=100,GUI=False) for _ in robots]
        scene.build()
        report = run(scene,robots,balls,onboard,fixed_cameras,detail_cameras,overview,top,
                     config,vision,fixed,mission,args.output_dir,not args.no_video,basket_cameras)
        if not report['verified']:
            raise RuntimeError('Two-robot mission failed a required check; inspect mission_check.json.')
    finally:
        gs.destroy()


if __name__=='__main__':
    main()
