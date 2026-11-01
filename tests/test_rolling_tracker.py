import json
import math
from pathlib import Path
import tempfile
import unittest

from rolling_tracker import RollingBallTracker, TrackerConfig, inspect_candidate
from replay import replay


ROBOTS = {'A':[20.,20.,.2], 'B':[-20.,-20.,.2]}
POLICY = {'playing_area_m':[23.77,10.97], 'court_size_m':[34.73,17.07],
          'trial_zones':{'A':[-15.2,-8.535,-9.5,-5.485], 'B':[9.5,5.485,15.2,8.535]}}


def detection(x,y=0.,dual=True):
    p = [x,y,.0335]
    cameras = ['A','B'] if dual else ['A']
    return dict(position_m=p,confidence=.8,sources=cameras,
                observations=[dict(camera_id=c,position_m=p.copy()) for c in cameras])


class TrackerTests(unittest.TestCase):
    def test_rolling_more_than_old_quarter_metre_gate(self):
        tracker = RollingBallTracker()
        identities = []
        for i in range(20):
            rows=tracker.update([detection(.32*i)],ROBOTS,.5*i)['tracks']
            identities.append(rows[0]['track_id'])
        self.assertEqual(len(set(identities)),1)
        self.assertEqual(rows[0]['state'],'observed')
        self.assertAlmostEqual(rows[0]['speed_m_s'],.64)
        self.assertGreater(rows[0]['position_m'][0],6)
        self.assertFalse(rows[0]['stationary'])

    def test_one_missed_frame_preserves_moving_identity(self):
        tracker=RollingBallTracker()
        for i in range(3):
            tracker.update([detection(.3*i)],ROBOTS,.5*i)
        missing=tracker.update([],ROBOTS,1.5)['tracks'][0]
        self.assertEqual(missing['state'],'unseen')
        self.assertEqual(missing['collection_status'],'unknown')
        rows=tracker.update([detection(1.2)],ROBOTS,2)['tracks']
        self.assertEqual(rows[0]['track_id'],1)

    def test_long_loss_starts_new_identity(self):
        tracker=RollingBallTracker()
        tracker.update([detection(0)],ROBOTS,0)
        tracker.update([],ROBOTS,.5)
        tracker.update([],ROBOTS,1)
        result=tracker.update([detection(0)],ROBOTS,1.5)
        self.assertEqual(result['tracks'][0]['track_id'],2)
        self.assertEqual(result['tracks'][0]['state'],'tentative')
        self.assertEqual(tracker.retired[0]['reason'],'expired_identity_unknown_collection')

    def test_single_false_detection_never_confirms(self):
        tracker=RollingBallTracker()
        row=tracker.update([detection(-12,-7)],ROBOTS,0)['tracks'][0]
        self.assertFalse(inspect_candidate(row,POLICY)['candidate_eligible'])
        self.assertEqual(tracker.update([],ROBOTS,1.2)['tracks'],[])

    def test_two_observations_never_confirm(self):
        tracker=RollingBallTracker()
        tracker.update([detection(0)],ROBOTS,0)
        row=tracker.update([detection(0)],ROBOTS,1)['tracks'][0]
        self.assertEqual(row['state'],'tentative')

    def test_high_rate_burst_does_not_supply_elapsed_evidence(self):
        tracker=RollingBallTracker()
        for i in range(20):
            row=tracker.update([detection(0)],ROBOTS,.01*i)['tracks'][0]
        self.assertEqual(row['state'],'tentative')

    def test_crossing_withholds_identity_instead_of_swapping(self):
        tracker=RollingBallTracker()
        for t,x in [(0,-.9),(.5,-.6),(1,-.3)]:
            tracker.update([detection(x),detection(-x)],ROBOTS,t)
        result=tracker.update([detection(0)],ROBOTS,1.5)
        self.assertEqual(result['tracks'],[])
        self.assertEqual(len(tracker.retired),2)
        self.assertTrue(all('ambiguous' in r['reason'] for r in tracker.retired))

    def test_duplicate_detections_do_not_choose_arbitrarily(self):
        tracker=RollingBallTracker()
        tracker.update([detection(0)],ROBOTS,0)
        result=tracker.update([detection(.1),detection(.12)],ROBOTS,.5)
        self.assertEqual(result['tracks'],[])

    def test_close_static_neighbours_keep_separate_ids(self):
        tracker=RollingBallTracker()
        for i in range(5):
            result=tracker.update([detection(.75),detection(0)],ROBOTS,.5*i)
        self.assertEqual([r['track_id'] for r in result['tracks']],[1,2])
        self.assertTrue(all(r['state']=='observed' for r in result['tracks']))

    def test_order_independence(self):
        a,b=RollingBallTracker(),RollingBallTracker()
        for i in range(8):
            values=[detection(.2*i),detection(5-.2*i)]
            self.assertEqual(a.update(values,ROBOTS,.5*i),b.update(values[::-1],ROBOTS,.5*i))

    def test_teleport_does_not_reuse_identity(self):
        tracker=RollingBallTracker()
        tracker.update([detection(0)],ROBOTS,0)
        rows=tracker.update([detection(5)],ROBOTS,.5)['tracks']
        self.assertEqual([(r['track_id'],r['state']) for r in rows],[(1,'unseen'),(2,'tentative')])

    def test_robot_mask_includes_both_robots_and_exact_boundary(self):
        tracker=RollingBallTracker()
        result=tracker.update([detection(0),detection(.65),detection(10)],
                              {'A':[0,0,0],'B':[10,0,0]},0)
        self.assertEqual(result['tracks'],[])
        self.assertEqual(result['filtered']['near_robot'],3)

    def test_invalid_detections_are_not_positions(self):
        tracker=RollingBallTracker()
        bad=[{},detection(float('nan')),detection(float('inf')),
             {**detection(0),'confidence':float('nan')}, {**detection(0),'confidence':1.1}]
        self.assertEqual(tracker.update(bad,ROBOTS,0)['filtered']['invalid'],5)

    def test_low_confidence_does_not_refresh_track(self):
        tracker=RollingBallTracker()
        tracker.update([detection(0)],ROBOTS,0)
        row=tracker.update([{**detection(0),'confidence':.2}],ROBOTS,.5)['tracks'][0]
        self.assertEqual(row['last_seen_seconds'],0)
        self.assertFalse(row['fresh_observation'])

    def test_time_validation(self):
        tracker=RollingBallTracker()
        tracker.update([],ROBOTS,1)
        for t in [1,.5,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): tracker.update([],ROBOTS,t)

    def test_pose_required(self):
        with self.assertRaises(ValueError): RollingBallTracker().update([detection(0)],{},0)
        with self.assertRaises(ValueError): RollingBallTracker().update([],{'A':[math.nan,0]},0)

    def test_bad_configuration_rejected(self):
        for kwargs in [dict(maximum_speed_m_s=0),dict(maximum_gap_seconds=math.inf),
                       dict(minimum_confidence=1.1),dict(confirmation_observations=2.5)]:
            with self.assertRaises(ValueError): TrackerConfig(**kwargs)

    def test_outside_floor_rays_cannot_extend_ground_tracks(self):
        tracker=RollingBallTracker(floor_size_m=[34.73,17.07])
        tracker.update([detection(0,8.4)],ROBOTS,0)
        result=tracker.update([detection(0,8.6)],ROBOTS,.5)
        self.assertEqual(result['filtered']['outside_floor'],1)
        self.assertEqual(result['tracks'][0]['state'],'unseen')
        self.assertEqual(result['tracks'][0]['collection_status'],'unknown')
        self.assertEqual(tracker.update([],ROBOTS,1.5)['tracks'],[])

    def test_stationary_candidate_is_observation_only(self):
        tracker=RollingBallTracker()
        for i in range(4): row=tracker.update([detection(-12,-7)],ROBOTS,.5*i)['tracks'][0]
        result=inspect_candidate(row,POLICY)
        self.assertTrue(result['candidate_eligible'])
        self.assertEqual(result['eligible_robots'],['A'])
        self.assertFalse(result['actuator_command_authorized'])
        self.assertEqual(row['collection_status'],'unknown')

    def test_motion_and_dropout_reset_stationarity(self):
        tracker=RollingBallTracker()
        for i in range(4): tracker.update([detection(-12,-7)],ROBOTS,.5*i)
        row=tracker.update([detection(-11.7,-7)],ROBOTS,2)['tracks'][0]
        self.assertFalse(row['stationary'])
        tracker.update([],ROBOTS,2.5)
        row=tracker.update([detection(-11.7,-7)],ROBOTS,3)['tracks'][0]
        self.assertFalse(row['stationary'])

    def test_court_and_trial_boundaries_block_candidates(self):
        for point,reason in [((0,0),'practice_court_entry_forbidden'),
                             ((-8,-7),'outside_trial_zones'),((0,9),'outside_simulated_floor')]:
            tracker=RollingBallTracker()
            for i in range(4): row=tracker.update([detection(*point)],ROBOTS,.5*i)['tracks'][0]
            self.assertIn(reason,inspect_candidate(row,POLICY)['withheld_reasons'])

    def test_single_camera_is_not_ground_confirmation(self):
        tracker=RollingBallTracker()
        for i in range(4): row=tracker.update([detection(-12,-7,dual=False)],ROBOTS,.5*i)['tracks'][0]
        self.assertIn('ground_plane_not_cross_checked',inspect_candidate(row,POLICY)['withheld_reasons'])

    def test_sources_label_does_not_substitute_camera_provenance(self):
        tracker=RollingBallTracker()
        obs=detection(-12,-7)
        obs['observations']=[obs['observations'][0],obs['observations'][0]]
        for i in range(4): row=tracker.update([obs],ROBOTS,.5*i)['tracks'][0]
        self.assertFalse(row['dual_view_agreement'])

    def test_disagreeing_ground_projections_are_withheld(self):
        tracker=RollingBallTracker()
        obs=detection(-12,-7)
        obs['observations'][1]['position_m'][0]+=.2
        for i in range(4): row=tracker.update([obs],ROBOTS,.5*i)['tracks'][0]
        self.assertFalse(inspect_candidate(row,POLICY)['candidate_eligible'])

    def test_snapshot_does_not_fabricate_future_observations(self):
        tracker=RollingBallTracker()
        for i in range(4): tracker.update([detection(-12,-7)],ROBOTS,.5*i)
        row=tracker.snapshot(2)[0]
        self.assertFalse(row['fresh_observation'])
        self.assertFalse(inspect_candidate(row,POLICY)['candidate_eligible'])

    def test_replay_uses_past_poses_and_never_loads_ground_truth(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)
            (p/'mission_check.json').write_text('invalid forbidden evaluator data')
            frames=[dict(seconds=t,merged_estimates=[detection(-12,-7)]) for t in [0,.5,1]]
            (p/'fixed_observations.json').write_text(json.dumps(frames))
            rows=[dict(seconds=t,robot=rid,robot_position_m=position,forbidden_ball_state='unused')
                  for t in [0,.55,1] for rid,position in ROBOTS.items()]
            (p/'control_trace.json').write_text(json.dumps(rows))
            result=replay(p,POLICY)
            self.assertEqual(result['frames'][1]['status'],'pose_unavailable')
            self.assertEqual(result['frames'][2]['tracks'][0]['track_id'],2)
            self.assertFalse(result['uses_ground_truth'])


if __name__=='__main__': unittest.main()
