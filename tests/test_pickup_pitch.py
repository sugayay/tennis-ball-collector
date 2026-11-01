import json,math,unittest
from pathlib import Path
import numpy as np
from pickup_pitch import flat_targets
from mission_control import MissionRobot


def nose_up(p):
    return np.array([[math.cos(p),0,-math.sin(p)],[0,1,0],[math.sin(p),0,math.cos(p)]])


class PickupPitchTests(unittest.TestCase):
    def test_nose_up_keeps_absolute_bucket_world_pitch_level(self):
        q,d=flat_targets([-.448,.448],nose_up(.025),.1)
        self.assertAlmostEqual(q[0],-.473);self.assertAlmostEqual(sum(q)+.025,0.)
    def test_nose_down_is_compensated_in_opposite_direction(self):
        q,d=flat_targets([-.448,.448],nose_up(-.02),.1)
        self.assertAlmostEqual(q[0],-.428);self.assertAlmostEqual(sum(q)-.02,0.)
    def test_existing_mechanical_stroke_is_respected(self):
        q=np.array([-.448,.448])
        for _ in range(100):q,d=flat_targets(q,nose_up(.09),.01)
        self.assertAlmostEqual(q[0],-.48);self.assertTrue(d['arm_lower_stroke_saturated']);self.assertGreaterEqual(q[1],-3.1);self.assertLessEqual(q[1],.85)
    def test_target_slew_and_zero_time_do_not_jump_pose(self):
        q,d=flat_targets([-.3,.3],nose_up(.06),.01);self.assertLessEqual(abs(q[0]+.3),.003+1e-12)
        q,d=flat_targets([-.3,.3],nose_up(.06),0.);np.testing.assert_array_equal(q,[-.3,.3])
    def test_invalid_feedback_is_rejected(self):
        with self.assertRaises(ValueError):flat_targets([float('nan'),0],np.eye(3),.01)
    def test_person_pause_does_not_follow_new_pitch(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text());r=MissionRobot('A',cfg,[23.77,10.97]);r.stage='brake';r.paused_since=0.;r.paused_joints=np.array([-.4,.4])
        wheels,q=r.command(np.array([-13.5,-6.5,.205]),nose_up(.06),.1)
        np.testing.assert_array_equal(wheels,[0,0]);np.testing.assert_array_equal(q,[-.4,.4]);self.assertEqual(r.pickup_pitch_records,[])
    def test_non_rgb_seating_mode_is_not_pitch_modified(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text());cfg['scoop_require_rgb_seating']=False;r=MissionRobot('A',cfg,[23.77,10.97]);r.stage='brake';previous=r.joints.copy()
        wheels,q=r.command(np.array([-13.5,-6.5,.205]),nose_up(.06),.1)
        np.testing.assert_array_equal(q,previous);self.assertEqual(r.pickup_pitch_records,[])


if __name__=='__main__':unittest.main()
