"""Replay RGB observations causally. Never reads mission_check/ball ground truth."""
import argparse
from bisect import bisect_right
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from rolling_tracker import RollingBallTracker, TrackerConfig, inspect_candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(directory, policy, config=None):
    camera_file, pose_file = directory/'fixed_observations.json', directory/'control_trace.json'
    frames = json.loads(camera_file.read_text(encoding='utf-8'))
    # Narrow the adapter to proprioception; targets/stages/commands are unused.
    poses = {rid: [] for rid in policy['trial_zones']}
    for row in json.loads(pose_file.read_text(encoding='utf-8')):
        if row['robot'] in poses:
            poses[row['robot']].append((float(row['seconds']),row['robot_position_m']))
    for rows in poses.values():
        rows.sort()
    times = {rid:[row[0] for row in rows] for rid,rows in poses.items()}
    tracker = RollingBallTracker(config, floor_size_m=policy['court_size_m'])
    results, retired = [], []
    for frame in frames:
        now = float(frame['seconds'])
        robot_positions = {}
        for rid, rows in poses.items():
            index = bisect_right(times[rid], now+1e-9)-1
            if index >= 0 and now-rows[index][0] <= .1+1e-9:
                robot_positions[rid] = rows[index][1]
        if set(robot_positions) != set(poses):
            # No unmasked update from a stale/missing robot pose. End identities
            # rather than bridge an unknown robot interval with a future pose.
            retired.extend(tracker.retired)
            retired.extend({'track_id':t.identity,'seconds':now,'reason':'pose_unavailable',
                            'history':t.history} for t in tracker.tracks.values())
            next_identity = tracker.next_identity
            tracker = RollingBallTracker(config, floor_size_m=policy['court_size_m'])
            tracker.next_identity = next_identity
            results.append(dict(seconds=now,tracks=[],filtered={},status='pose_unavailable'))
            continue
        result = tracker.update(frame['merged_estimates'], robot_positions, now)
        for row in result['tracks']:
            row.update(inspect_candidate(row, policy))
        result['status'] = 'observed'
        results.append(result)
    retired.extend(tracker.retired)
    history = retired + [dict(track_id=t.identity,seconds=None,reason='end_of_recording',history=t.history)
                         for t in tracker.tracks.values()]
    return dict(scope='offline_rgb_observer_not_a_collection_or_navigation_controller',
        inputs_sha256={camera_file.name:digest(camera_file),pose_file.name:digest(pose_file)},
        uses_ground_truth=False,uses_future_poses=False,collection_confirmed=False,
        actuators_connected=False,policy=policy,tracker_config=asdict(tracker.config),
        frames=results,trajectories=sorted(history,key=lambda t:t['track_id']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = replay(args.directory,json.loads(args.policy.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'frames':len(result['frames']),'trajectories':len(result['trajectories'])}))
