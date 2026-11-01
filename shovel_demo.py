"""Test one real contact-based shovel cycle, using known starting positions.

No AI or autonomous route planning is used. The free robot drives with wheel
velocity control; the shovel moves with joint position controllers. Ball state
is only read, never moved or attached by the script.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import av
import genesis as gs
import numpy as np

from scene import ROOT, as_array, build_scene, initialize_shovel, save_rgb


def local_position(entity, world_position):
    q = as_array(entity.get_quat())
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return rotation.T @ (world_position - as_array(entity.get_pos()))


def eased(start, end, t):
    blend = t * t * (3 - 2 * t)
    return start + (end - start) * blend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['amdgpu', 'cpu'], default='amdgpu')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'scene_config.json')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--motion-only', action='store_true', help='Validate joint motion only; no ball pickup is attempted.')
    args = parser.parse_args()
    output = (args.output_dir or ROOT / 'artifacts' / ('shovel_motion' if args.motion_only else 'shovel')).resolve()
    video_name = 'shovel_motion.mp4' if args.motion_only else 'shovel_demo.mp4'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    gs.init(backend=getattr(gs, args.backend), seed=13, logging_level='warning')
    container = None
    try:
        scene, robot, balls, camera, top_camera, detail_camera = build_scene(config, (1280, 720))
        scene.build()
        shovel_dofs = initialize_shovel(robot)
        wheel_dofs = [robot.get_joint(name).dofs_idx_local[0] for name in ('left_wheel_joint', 'right_wheel_joint')]
        robot.set_dofs_kp(np.array([0.0, 0.0]), dofs_idx_local=wheel_dofs)
        robot.set_dofs_kv(np.array([12.0, 12.0]), dofs_idx_local=wheel_dofs)
        robot.set_dofs_force_range(np.array([-6.0, -6.0]), np.array([6.0, 6.0]), dofs_idx_local=wheel_dofs)
        robot.control_dofs_velocity(np.zeros(2), dofs_idx_local=wheel_dofs)
        target = balls[0]
        bucket = robot.get_link('shovel_bucket')
        for _ in range(120):
            scene.step()
        start_robot = as_array(robot.get_pos())
        start_ball = as_array(target.get_pos())
        maximum_ball_height = float(start_ball[2])
        minimum_base_height = float(start_robot[2])
        maximum_base_tilt = 0.0
        phases = []
        frame_count = 0
        step_count = 0
        fps = 12
        frame_stride = 10
        if not args.no_video:
            container = av.open(str(output / video_name), mode='w')
            stream = container.add_stream('libx264', rate=fps)
            stream.width, stream.height = 1280, 720
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '20', 'preset': 'fast'}

        def camera_pose():
            p = as_array(robot.get_pos())
            camera.set_pose(pos=(p[0] + 1.6, p[1] - 1.9, 1.45), lookat=(p[0] + 0.1, p[1], 0.35))

        camera_pose()
        save_rgb(camera, output / 'shovel_start.png')

        def phase(name, seconds, arm_start, arm_end, bucket_start, bucket_end, drive=False, stop_on_lip=False, wheel_speed=2.8):
            nonlocal maximum_ball_height, minimum_base_height, maximum_base_tilt, frame_count, step_count
            positions = []
            total = round(seconds * 120)
            for index in range(total):
                t = (index + 1) / total
                arm_angle = eased(arm_start, arm_end, t)
                absolute_bucket_angle = eased(bucket_start, bucket_end, t)
                robot.control_dofs_position(np.array([arm_angle, absolute_bucket_angle - arm_angle]), dofs_idx_local=shovel_dofs)
                if drive == 'approach':
                    ramp = min(1.0, (index + 1) / 60)
                elif drive == 'cruise':
                    ramp = 1.0
                elif drive == 'brake':
                    ramp = 1 - t
                else:
                    ramp = min(1.0, (index + 1) / 60, (total - index) / 60) if drive else 0.0
                robot.control_dofs_velocity(np.full(2, wheel_speed * ramp), dofs_idx_local=wheel_dofs)
                scene.step()
                position = as_array(target.get_pos())
                positions.append(local_position(robot, position).tolist())
                maximum_ball_height = max(maximum_ball_height, float(position[2]))
                base = as_array(robot.get_pos())
                minimum_base_height = min(minimum_base_height, float(base[2]))
                q = as_array(robot.get_quat())
                up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
                maximum_base_tilt = max(maximum_base_tilt, math.degrees(math.acos(float(np.clip(up_z, -1, 1)))))
                step_count += 1
                if container is not None and step_count % frame_stride == 0:
                    camera_pose()
                    rgb = np.asarray(camera.render()[0], dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    frame_count += 1
                if stop_on_lip:
                    lip_position = local_position(bucket, position)
                    if 0 < lip_position[0] < 0.265 and abs(lip_position[1]) < 0.105 and position[2] > start_ball[2] + 0.0005:
                        break
            state = {
                'name': name, 'seconds': len(positions) / 120,
                'robot_position_m': as_array(robot.get_pos()).tolist(),
                'ball_position_m': as_array(target.get_pos()).tolist(),
                'ball_in_robot_frame_m': local_position(robot, as_array(target.get_pos())).tolist(),
                'ball_in_bucket_frame_m': local_position(bucket, as_array(target.get_pos())).tolist(),
                'shovel_joint_radians': as_array(robot.get_dofs_position(dofs_idx_local=shovel_dofs)).tolist(),
                'wheel_joint_radians': as_array(robot.get_dofs_position(dofs_idx_local=wheel_dofs)).tolist(),
                'wheel_velocity_rad_s': as_array(robot.get_dofs_velocity(dofs_idx_local=wheel_dofs)).tolist(),
            }
            phases.append(state)
            print(json.dumps(state), flush=True)
            return positions

        pickup = -0.448
        capture = math.radians(5)
        carry = math.radians(15)
        raised = math.radians(125)
        tipped = math.radians(-50)
        if args.motion_only:
            lift_start, bucket_start = pickup, 0.0
        else:
            phase('approach', 4.0, pickup, pickup, 0.0, 0.0, drive='approach', stop_on_lip=True)
            scoop_height = -0.20
            phase('scoop_while_advancing', 0.30, pickup, scoop_height, 0.0, capture, drive='cruise', wheel_speed=6.0)
            phase('secure_ball', 1.0, scoop_height, scoop_height, capture, carry, drive='brake', wheel_speed=6.0)
            camera_pose()
            save_rgb(camera, output / 'shovel_pickup.png')
            lift_start, bucket_start = scoop_height, carry
        phase('lift', 3.0, lift_start, raised, bucket_start, carry)
        camera_pose()
        save_rgb(camera, output / 'shovel_lift.png')
        phase('tip_into_basket', 1.8, raised, raised, carry, tipped)
        camera_pose()
        save_rgb(camera, output / 'shovel_tip.png')
        phase('empty_bucket_level', 1.0, raised, raised, tipped, carry)
        phase('lower_empty_bucket', 3.0, raised, pickup, carry, 0.0)
        retention = phase('hold_rest_pose' if args.motion_only else 'retain_in_basket', 1.0 if args.motion_only else 3.0, pickup, pickup, 0.0, 0.0)
        p = as_array(robot.get_pos())
        camera.set_pose(pos=(p[0] + 0.8, p[1] - 1.3, 1.3), lookat=(p[0] - 0.03, p[1], 0.27))
        save_rgb(camera, output / ('shovel_rest.png' if args.motion_only else 'shovel_collected.png'))

        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            with av.open(str(output / video_name)) as video:
                decoded_count = sum(1 for _ in video.decode(video=0))
            assert decoded_count == frame_count

        retained = all(
            -0.195 < x < 0.035 and abs(y) < 0.106 and 0.080 < z < 0.180
            for x, y, z in retention
        )
        collection_checks = {
            'ball_lifted_above_0_4m': maximum_ball_height > 0.4,
            'ball_retained_in_basket_for_3_seconds': retained,
        }
        by_name = {p['name']: p for p in phases}
        lift_position = by_name['lift']['shovel_joint_radians']
        tip_position = by_name['tip_into_basket']['shovel_joint_radians']
        rest_position = phases[-1]['shovel_joint_radians']
        motion_checks = {
            'lift_joint_reached_target': abs(lift_position[0] - raised) < 0.05,
            'bucket_tilt_reached_target': abs(sum(tip_position) - tipped) < 0.05,
            'shovel_returned_to_rest': abs(rest_position[0] - pickup) < 0.03 and abs(sum(rest_position)) < 0.03,
            'base_tilt_below_8_degrees': maximum_base_tilt < 8,
            'base_stayed_above_0_17m': minimum_base_height > 0.17,
        }
        checks = motion_checks if args.motion_only else {**motion_checks, **collection_checks}
        report = {
            'verified': all(checks.values()),
            'mode': 'motion_only' if args.motion_only else 'pickup_trial',
            'motion_verified': all(motion_checks.values()),
            'collection_verified': None if args.motion_only else all(collection_checks.values()),
            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'empty shovel joint motion only; no pickup attempted' if args.motion_only else 'one ball, simulator state used for pickup timing; scripted wheel and shovel control; no AI',
            'ball_state_modified_by_script': False,
            'ball_attachment_used': False,
            'robot_has_free_base': True,
            'robot_self_collision_enabled': True,
            'start_robot_position_m': start_robot.tolist(),
            'start_ball_position_m': start_ball.tolist(),
            'maximum_ball_height_m': maximum_ball_height,
            'maximum_base_tilt_degrees': maximum_base_tilt,
            'minimum_base_height_m': minimum_base_height,
            'physics_steps': step_count,
            'configuration': {'ball_positions_m': config['ball_positions_m'], 'ball_rolling_resistance_m': config.get('ball_rolling_resistance_m')},
            'checks': checks,
            'phases': phases,
            'video': None if args.no_video else {'frames': frame_count, 'fps': fps, 'seconds': frame_count / fps, 'playback_speed': 1.0},
        }
        (output / 'shovel_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'mode': report['mode'], 'verified': report['verified'], 'motion_verified': report['motion_verified'], 'collection_verified': report['collection_verified'], 'checks': checks}), flush=True)
        if not report['verified']:
            raise RuntimeError('The physical shovel cycle did not pass; inspect shovel_check.json and phase images.')
    finally:
        if container is not None:
            container.close()
        gs.destroy()


if __name__ == '__main__':
    main()
