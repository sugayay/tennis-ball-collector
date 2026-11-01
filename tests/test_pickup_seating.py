from pathlib import Path
from types import SimpleNamespace
import json,unittest
import numpy as np
from pickup_seating import ImageSeatingGate
from mission_control import MissionRobot,PICKUP

def ball(x,y=0.):return SimpleNamespace(position_in_robot_m=[x,y,-.1715])

class SeatingTests(unittest.TestCase):
    def setUp(self):
        self.position=np.array([-13.6,-6.5,.205]);self.rotation=np.eye(3)
        self.target=self.position+np.array([.6,0,-.1715])
        self.gate=ImageSeatingGate()
    def observe(self,x,time,y=0.):
        self.gate.observe([ball(x,y)],self.position,self.rotation,self.target,time,0.)
    def test_two_fresh_distinct_seated_frames_are_required(self):
        self.observe(.50,.1)
        self.assertFalse(self.gate.ready(self.position,self.rotation,.1))
        self.observe(.50,.1)
        self.assertEqual(self.gate.frames,1)
        self.observe(.50,.15)
        self.assertTrue(self.gate.ready(self.position,self.rotation,.2))
        self.assertFalse(self.gate.ready(self.position,self.rotation,.31))
    def test_ball_at_front_or_side_and_other_target_never_authorize_lift(self):
        for x,y in ((.55,0),(.5,.08),(1.2,0),(.3,0)):
            gate=ImageSeatingGate()
            for t in (.1,.15):gate.observe([ball(x,y)],self.position,self.rotation,self.target,t,0.)
            self.assertFalse(gate.ready(self.position,self.rotation,.2))
    def test_hysteresis_never_starts_without_a_strict_seated_frame(self):
        self.observe(.533,.1);self.observe(.533,.15)
        self.assertEqual(self.gate.frames,0)
        self.assertFalse(self.gate.ready(self.position,self.rotation,.15))
    def test_hysteresis_keeps_count_but_never_relaxes_lift_region(self):
        self.observe(.527,.1);self.observe(.533,.15)
        self.assertEqual(self.gate.frames,2)
        self.assertFalse(self.gate.ready(self.position,self.rotation,.15))
        self.assertTrue(self.gate.ready(self.position+np.array([.005,0,0]),self.rotation,.16))
    def test_ramp_region_is_allowed_but_front_edge_is_not(self):
        self.observe(.528,.1);self.observe(.528,.15)
        self.assertTrue(self.gate.ready(self.position,self.rotation,.15))
        self.observe(.54,.2)
        self.assertFalse(self.gate.ready(self.position,self.rotation,.2))
    def test_evidence_outside_hysteresis_band_resets_count(self):
        self.observe(.527,.1);self.observe(.536,.15)
        self.assertEqual(self.gate.frames,0)
        self.observe(.512,.20)
        self.assertEqual(self.gate.frames,1)
    def test_missing_detection_or_gap_breaks_consecutive_evidence(self):
        self.observe(.5,.1);self.observe(.5,.4)
        self.assertEqual(self.gate.frames,1)
        self.gate.observe([],self.position,self.rotation,self.target,.45,0.)
        self.assertTrue(self.gate.fresh(.45));self.assertEqual(self.gate.frames,0)
    def test_missing_images_never_refresh_valid_timestamp_or_unlock_lift(self):
        self.observe(.5,.1);self.observe(.5,.15)
        for t in (.20,.25,.29,.31):
            self.gate.observe([],self.position,self.rotation,self.target,t,0.)
            self.assertEqual(self.gate.valid_seconds,.15)
            self.assertFalse(self.gate.ready(self.position,self.rotation,t))
        self.assertFalse(self.gate.fresh(.31))
    def test_reacquisition_requires_two_new_frames_after_a_miss(self):
        self.observe(.5,.1);self.observe(.5,.15)
        self.gate.observe([],self.position,self.rotation,self.target,.2,0.)
        self.observe(.5,.25)
        self.assertFalse(self.gate.ready(self.position,self.rotation,.25))
        self.observe(.5,.3)
        self.assertTrue(self.gate.ready(self.position,self.rotation,.3))
    def test_no_initial_detection_or_other_ball_cannot_create_fresh_evidence(self):
        self.observe(1.2,.1)
        self.assertFalse(self.gate.fresh(.1))
        self.observe(.5,.15)
        self.observe(1.2,.2)
        self.assertEqual(self.gate.valid_seconds,.15)
        self.assertFalse(self.gate.matched_current_frame)
        self.assertFalse(self.gate.ready(self.position,self.rotation,.2))
    def test_short_dropout_keeps_flat_motion_but_timeout_stops(self):
        r=self.robot();p=self.position+np.array([.06,0,0])
        r.command(p,self.rotation,.1)
        r.observe_onboard([ball(.54)],p,self.rotation,.2)
        r.observe_onboard([],p,self.rotation,.25)
        wheels,joints=r.command(p,self.rotation,.25)
        self.assertGreater(float(wheels.mean()),0)
        np.testing.assert_allclose(joints,[PICKUP,-PICKUP])
        self.assertIsNone(r.scoop_lift_started)
        event=r.seating_observation_events[-1]
        self.assertIsNone(event['ball_in_robot_at_delivery'])
        self.assertEqual(event['last_valid_captured_seconds'],.2)
        wheels,joints=r.command(p,self.rotation,.351)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertIsNone(r.scoop_lift_started)
    def test_pre_attempt_frame_and_out_of_order_frame_cannot_count(self):
        self.gate.observe([ball(.5)],self.position,self.rotation,self.target,.1,.2)
        self.assertEqual(self.gate.frames,0)
        self.observe(.5,.3);self.observe(.5,.25)
        self.assertEqual(self.gate.frames,1)
    def test_capture_pose_is_reprojected_into_current_robot_pose(self):
        for t in (.1,.15):self.observe(.50,t)
        behind=self.position-np.array([.05,0,0])
        self.assertFalse(self.gate.ready(behind,self.rotation,.2))
    def robot(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
        cfg.update(scoop_require_rgb_seating=True,scoop_seating_max_travel_m=.18)
        r=MissionRobot('A',cfg,[23.77,10.97]);r.assign({'position_m':self.target.tolist()},self.position,0.)
        r.transition('scoop',0.);r.heading_reference=0.;r.scoop_start_position=self.position.copy()
        return r
    def test_travel_alone_cannot_raise_the_arm_and_stale_camera_stops_flat(self):
        r=self.robot();w,j=r.command(self.position+np.array([.1,0,0]),self.rotation,.3)
        np.testing.assert_array_equal(w,[0,0]);np.testing.assert_allclose(j,[PICKUP,-PICKUP])
        self.assertIsNone(r.scoop_lift_started)
        r.command(self.position+np.array([.1,0,0]),self.rotation,1.51)
        self.assertEqual(r.stage,'fault');self.assertEqual(r.events[-1]['reason'],'rgb_seating_camera_timeout')
    def test_seating_unlocks_lift_at_measured_insertion_then_retains_rate_limit(self):
        r=self.robot();p=self.position+np.array([.10,0,0])
        r.command(p,self.rotation,.1)
        for t in (.2,.25):r.observe_onboard([ball(.50)],p,self.rotation,t)
        r.command(p,self.rotation,.25)
        self.assertEqual(len(r.seating_events),1);self.assertAlmostEqual(r.seating_insertion_m,.10)
        _,j=r.command(self.position+np.array([.16,0,0]),self.rotation,.30)
        self.assertEqual(r.stage,'scoop');self.assertLess(j[0],-.20)
        r.command(self.position+np.array([.16,0,0]),self.rotation,.60)
        self.assertEqual(r.stage,'secure')
    def test_unseated_ball_cannot_drive_past_translation_limit(self):
        r=self.robot();w,_=r.command(self.position+np.array([.181,0,0]),self.rotation,.6)
        self.assertEqual(r.stage,'fault');self.assertEqual(r.events[-1]['reason'],'rgb_seating_travel_limit')
        np.testing.assert_array_equal(w,[0,0])
    def test_slower_speed_applies_only_before_seating(self):
        r=self.robot();r.config['scoop_insertion_speed_m_s']=.2
        r.filtered_forward_speed=.2
        r.observe_onboard([ball(.54)],self.position,self.rotation,.1)
        wheels,_=r.command(self.position,self.rotation,.1)
        self.assertAlmostEqual(float(wheels.mean())*.11,.2)
        r.scoop_lift_started=.1;r.seating_insertion_m=0.
        wheels,_=r.command(self.position,self.rotation,.1)
        self.assertAlmostEqual(float(wheels.mean())*.11,.5)
    def test_invalid_insertion_speed_is_rejected(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
        cfg.update(scoop_require_rgb_seating=True,scoop_seating_max_travel_m=.18)
        for speed in (0.,-.1,.31,float('nan'),float('inf')):
            with self.subTest(speed=speed):
                cfg['scoop_insertion_speed_m_s']=speed
                with self.assertRaises(ValueError):MissionRobot('A',cfg,[23.77,10.97])
        cfg['scoop_insertion_speed_m_s']=.2;cfg['scoop_require_rgb_seating']=False
        with self.assertRaises(ValueError):MissionRobot('A',cfg,[23.77,10.97])
    def test_config_cannot_exceed_total_measured_travel_bound(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
        cfg.update(scoop_require_rgb_seating=True,scoop_seating_max_travel_m=.23)
        with self.assertRaises(ValueError):MissionRobot('A',cfg,[23.77,10.97])

if __name__=='__main__':unittest.main()
