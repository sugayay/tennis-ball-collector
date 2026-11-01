"""Two collectors placed on opposite ends; operating areas remain undecided.

The existing single-robot pickup and its controller are left unchanged.
This module adds a second physical robot, not a copy drawn on an image.
"""
from __future__ import annotations

import copy
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import genesis as gs
import numpy as np

from scene import ROOT, as_array, build_scene, initialize_shovel, save_rgb
from onboard_vision import OnboardCamera
from drive_actuators import configure_wheels


def build_fleet(config, fleet_config, vision_config):
    specs = fleet_config['robots']
    if len(specs) != 2 or specs[0]['yaw_degrees'] != 0:
        raise ValueError('This inspection expects two robots and robot A facing +x.')
    scene_config = copy.deepcopy(config)
    scene_config['robot_position_m'] = specs[0]['position_m']
    scene, first, balls, overview, top, detail = build_scene(scene_config, (1600, 900))
    robots = [first]
    for spec in specs[1:]:
        angle = math.radians(spec['yaw_degrees']) / 2
        robots.append(scene.add_entity(
            gs.morphs.URDF(
                file=str(ROOT / 'assets' / config.get('robot_urdf', 'collector.urdf')),
                pos=tuple(spec['position_m']),
                quat=(math.cos(angle), 0, 0, math.sin(angle)),
                fixed=False, prioritize_urdf_material=True, decimate=False,
            ),
            material=gs.materials.Rigid(friction=0.8),
            name=f"collector_robot_{spec['id']}",
        ))
    onboard = [OnboardCamera(scene, robot, vision_config['camera'], spec['position_m'])
               for robot, spec in zip(robots, specs)]
    return scene, robots, balls, overview, top, detail, onboard


def hold_fleet(robots):
    """Only hold wheels and shovel at rest; no target selection or travel."""
    for robot in robots:
        initialize_shovel(robot)
        configure_wheels(robot)


def check_fleet(robots, initial_positions):
    checks = []
    for robot, initial in zip(robots, initial_positions):
        p, q = as_array(robot.get_pos()), as_array(robot.get_quat())
        tilt = math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1))))
        drift = float(np.linalg.norm(p[:2]-np.array(initial[:2])))
        checks.append({'position_m':p.tolist(), 'drift_m':drift,
                       'tilt_degrees':tilt,
                       'stable':bool(np.isfinite(p).all() and drift < 0.025
                                     and 0.17 < p[2] < 0.24 and tilt < 3)})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['amdgpu', 'cpu'], default='amdgpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts'/'fleet')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'scene_config.json').read_text())
    fleet = json.loads((ROOT/'fleet_config.json').read_text())
    vision = json.loads((ROOT/'vision_config.json').read_text())
    gs.init(backend=getattr(gs,args.backend), seed=13, logging_level='warning')
    try:
        scene, robots, balls, overview, top, detail, onboard = build_fleet(config, fleet, vision)
        scene.build()
        hold_fleet(robots)
        for _ in range(120):
            scene.step()
        checks = check_fleet(robots, [spec['position_m'] for spec in fleet['robots']])
        save_rgb(overview, output/'fleet_overview.png')
        save_rgb(top, output/'fleet_top.png')
        for spec, robot, camera in zip(fleet['robots'], robots, onboard):
            from PIL import Image
            Image.fromarray(camera.capture()).save(output/f"robot_{spec['id']}_onboard.png")
            p = as_array(robot.get_pos())
            sign = 1 if spec['id'] == 'A' else -1
            detail.set_pose(pos=p+np.array([sign*1.35, -sign*1.4, 1.1]),
                            lookat=p+np.array([sign*0.2, 0, 0.08]))
            save_rgb(detail, output/f"robot_{spec['id']}_detail.png")
        report = {'checked_at_utc':datetime.now(timezone.utc).isoformat(),
                  'robot_count':len(robots), 'robots':checks,
                  'operating_area_status':fleet['operating_area_status'],
                  'autonomous_two_robot_driving':False,
                  'verified':all(item['stable'] for item in checks)}
        (output/'fleet_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not report['verified']:
            raise RuntimeError('A robot was unstable at rest.')
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
