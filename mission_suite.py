"""Repeat the same RGB-driven controller on five declared ball configurations.

Each episode is a fresh process. Initialization coordinates are removed before
controller creation by fleet_mission.py. Failed physical trials remain results;
an execution error never masquerades as a zero-ball physical trial.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
BEHAVIOR_FILES = (
    'mission_control.py','handoff_control.py','vision_control.py','fleet_mission.py',
    'scene.py','fleet_scene.py','onboard_vision.py','fixed_detection.py','camera_geometry.py',
    'fixed_camera_demo.py','fixed_camera_visuals.py','shovel_demo.py',
    'scene_config.json','vision_config.json','fleet_config.json','fixed_camera_config.json',
    'mission_straight_config.json','mission_cases.json','assets/collector.urdf',
    'assets/collector_basket_lift.urdf','lift_storage.py',
    'mission_suite.py','drive_speed.py','person_safety.py','assets/collector_beveled.urdf','assets/collector_beveled_floor.obj',
    'basket_camera_config.json','basket_vision.py','basket_confirmation.py',
    'recovery_config.json','recollection.py','rolling_tracker.py','basket_view_clearance.py',
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in BEHAVIOR_FILES}


def prepare_case(base, case):
    if not re.fullmatch(r'[a-z0-9_]+',case['id']):
        raise ValueError('Case IDs must be simple, unique directory names.')
    points = case['ball_positions_m']
    if len(points)!=4 or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('Each case requires four finite XY initialization positions.')
    config = copy.deepcopy(base)
    config['test_ball_positions_m'] = copy.deepcopy(points)
    if 'robot_positions_m' in case:
        config['test_robot_positions_m']=copy.deepcopy(case['robot_positions_m'])
    if 'trial_zones' in case:
        config['trial_zones'] = copy.deepcopy(case['trial_zones'])
    for x,y in points:
        if abs(x)<23.77/2 and abs(y)<10.97/2:
            raise ValueError('Practice tests must not select balls inside the playing lines.')
        if not any(b[0]<=x<=b[2] and b[1]<=y<=b[3] for b in config['trial_zones'].values()):
            raise ValueError('Every initial ball must lie in a declared temporary trial zone.')
    for i,p in enumerate(points):
        if any(math.dist(p,q)<.067 for q in points[i+1:]):
            raise ValueError('Initial ball geometry overlaps.')
    return config


def result_row(case, report, returncode, elapsed):
    if not report or 'checks' not in report:
        return {'id':case['id'],'label_ja':case['label_ja'],'status':'execution_error',
                'returncode':returncode,'wall_clock_seconds':elapsed,'held_ball_count':None}
    held = sorted({i for p in report['physics'].values() for i in p['last_3_seconds_held_ball_indices']})
    return {'id':case['id'],'label_ja':case['label_ja'],'status':'complete',
            'held_ball_count':len(held),'expected_ball_count':4,
            'held_per_robot':{rid:len(p['last_3_seconds_held_ball_indices']) for rid,p in report['physics'].items()},
            'collection_goal_met':held==[0,1,2,3],
            'all_physical_checks_passed':report['verified'],
            'passed_checks':sum(report['checks'].values()),'total_checks':len(report['checks']),
            'failed_checks':[name for name,ok in report['checks'].items() if not ok],
            'uncollected_ball_indices':[i for i in range(4) if i not in held],
            'maximum_ball_heights_m':report['maximum_ball_heights_m'],
            'faults':[e for e in report['events'] if e['to']=='fault'],
            'retries':report['retries'],'simulated_seconds':report['simulated_seconds'],
            'wall_clock_seconds':elapsed,'returncode':returncode,
            'controller_uses_simulator_ball_state':report['controller_uses_simulator_ball_state'],
            'ball_state_modified_during_episode':report['ball_state_modified_during_episode'],
            'model_sha256':report['model_sha256'],
            'robot_variant':report.get('robot_variant','stock')}


def suite_summary(cases, rows, provenance, unchanged, jobs, robot_variant='stock'):
    complete = [r for r in rows.values() if r['status']=='complete']
    ordered = [rows[c['id']] for c in cases]
    return {'scope':'declared_development_cases_not_a_randomized_or_real_world_success_rate',
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'complete' if len(complete)==len(cases) else 'running_or_execution_error',
            'case_count':len(cases),'completed_cases':len(complete),
            'expected_ball_count':4*len(cases),'evaluated_ball_count':4*len(complete),
            'held_ball_count':sum(r['held_ball_count'] for r in complete),
            'all_cases_collected_all_balls':len(complete)==len(cases) and all(r['collection_goal_met'] for r in complete),
            'all_cases_passed_all_physical_checks':len(complete)==len(cases) and all(r['all_physical_checks_passed'] for r in complete),
            'source_unchanged_during_suite':unchanged,'behavior_source_sha256':provenance,
            'parallel_processes':jobs,'robot_variant':robot_variant,
            'wall_time_is_not_robot_real_time_performance':True,
            'operating_areas_status':'temporary_unoccupied_regions_real_operation_pending',
            'cases':ordered}


def run_suite(args):
    suite = read(ROOT/'mission_cases.json')
    cases = [c for c in suite['cases'] if not args.cases or c['id'] in args.cases]
    if not cases or (args.cases and set(args.cases)!={c['id'] for c in cases}):
        raise ValueError('Choose existing case IDs.')
    if len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate case ID.')
    output = args.output_dir.resolve()
    if (output/'suite_check.json').exists():
        raise FileExistsError('Preserve previous runs; choose a fresh output directory.')
    output.mkdir(parents=True,exist_ok=True)
    configs = output/'initialization_configs';configs.mkdir(exist_ok=True)
    base = read(ROOT/'mission_straight_config.json')
    for case in cases:
        (configs/f"{case['id']}.json").write_text(json.dumps(prepare_case(base,case),indent=2)+'\n',encoding='utf-8')
    provenance = hashes()
    rows = {c['id']:{'id':c['id'],'label_ja':c['label_ja'],'status':'pending'} for c in cases}
    lock = threading.Lock()

    def save():
        summary = suite_summary(cases,rows,provenance,hashes()==provenance,args.jobs,args.robot_variant)
        (output/'suite_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return summary

    save()

    def episode(case):
        directory = output/case['id'];directory.mkdir()
        with lock:
            rows[case['id']]['status']='running';save()
            print(json.dumps({'case_started':case['id'],'label':case['label_ja']},ensure_ascii=False),flush=True)
        command = [sys.executable,str(ROOT/'fleet_mission.py'),'--backend',args.backend,
                   '--case-config',str(configs/f"{case['id']}.json"),'--output-dir',str(directory),
                   '--robot-variant',args.robot_variant]
        if args.no_video or args.video_cases is not None and case['id'] not in args.video_cases:
            command.append('--no-video')
        start = time.perf_counter()
        with (directory/'process.log').open('w',encoding='utf-8') as log:
            try:
                done = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                code = done.returncode
            except subprocess.TimeoutExpired:
                code = -9
                log.write('\nSuite runner wall-time limit reached. This is an execution timeout.\n')
        path = directory/'mission_check.json'
        report = read(path) if path.exists() else None
        row = result_row(case,report,code,time.perf_counter()-start)
        with lock:
            rows[case['id']] = row;save()
            print(json.dumps({'case_completed':row},ensure_ascii=False),flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(episode,c) for c in cases]
        for future in as_completed(futures):
            future.result()
    summary = save()
    print(json.dumps({'completed_cases':summary['completed_cases'],'held_ball_count':summary['held_ball_count'],
                      'expected_ball_count':summary['expected_ball_count'],
                      'all_physical_checks_passed':summary['all_cases_passed_all_physical_checks'],
                      'source_unchanged':summary['source_unchanged_during_suite']}),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['amdgpu','cpu'],default='amdgpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'suite')
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--jobs',type=int,choices=(1,2),default=1)
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--video-cases',nargs='+',help='Record only these cases; RGB control and all physical checks still run in every case.')
    parser.add_argument('--robot-variant',choices=['stock','beveled','lift'],default='stock')
    args = parser.parse_args()
    result = run_suite(args)
    if not result['all_cases_passed_all_physical_checks'] or not result['source_unchanged_during_suite']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
