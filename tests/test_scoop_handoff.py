"""The handoff may preserve motion, never pre-attempt lift evidence."""
from pathlib import Path
from types import SimpleNamespace
import json,unittest
import numpy as np
from mission_control import MissionRobot,PICKUP
from pickup_seating import ImageSeatingGate

class ScoopHandoffTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([-13.6,-6.5,.205]);self.R=np.eye(3)
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
        self.cfg.update(scoop_require_rgb_seating=True,scoop_seating_max_travel_m=.18,
            scoop_insertion_speed_m_s=.2,scoop_insertion_acceleration_m_s2=.4)
    def robot(self):
        r=MissionRobot('A',self.cfg,[23.77,10.97])
        r.target=self.p+np.array([.59,0,-.1715])
        r.stage='visual';r.filtered_forward_speed=.31;r.last_visual_forward_command=.32
        r.visual=SimpleNamespace(world_target=r.target.copy(),last_seen=.17,yaw_rate=0.,
            command=lambda p,R,t:(np.zeros(2),True,'at_shovel'))
        return r
    def test_handoff_has_no_zero_command_and_keeps_arm_flat(self):
        r=self.robot();w,j=r.command(self.p,self.R,.2,dt=.01)
        self.assertAlmostEqual(w.mean()*.11,.316)
        np.testing.assert_allclose(j,[PICKUP,-PICKUP])
        self.assertEqual(r.seating_gate.frames,0)
        self.assertEqual(r.seating_gate.valid_seconds,.17)
        self.assertIsNone(r.scoop_lift_started)
    def test_seed_does_not_count_or_refresh_precapture_for_lift(self):
        gate=ImageSeatingGate();target=self.p+np.array([.5,0,-.1715])
        gate.seed_motion(target,.17,.2)
        obs=[SimpleNamespace(position_in_robot_m=[.5,0,-.1715])]
        gate.observe(obs,self.p,self.R,target,.19,.2)
        self.assertEqual(gate.frames,0)
        self.assertEqual(gate.valid_seconds,.17)
        gate.observe(obs,self.p,self.R,target,.21,.2)
        self.assertFalse(gate.ready(self.p,self.R,.21))
        gate.observe(obs,self.p,self.R,target,.25,.2)
        self.assertTrue(gate.ready(self.p,self.R,.25))
    def test_invalid_or_stale_seed_never_allows_motion(self):
        for point,stamp,now in ((self.p,.01,.2),(self.p,.21,.2),
                ([float('nan'),0,0],.17,.2),(self.p,float('nan'),.2),
                (None,.17,.2),(self.p,.17,float('inf'))):
            gate=ImageSeatingGate();gate.seed_motion(point,stamp,now)
            self.assertFalse(gate.fresh(.2))
    def test_acceleration_and_deceleration_are_bounded(self):
        for previous,measured in ((0.,0.),(.32,.31)):
            r=self.robot();r.last_visual_forward_command=previous;r.filtered_forward_speed=measured
            w,_=r.command(self.p,self.R,.2,dt=.01)
            self.assertLessEqual(abs(w.mean()*.11-previous),.00400001)
    def test_stale_stop_is_immediate_and_reacquisition_starts_from_zero(self):
        r=self.robot();r.command(self.p,self.R,.2,dt=.01)
        w,_=r.command(self.p,self.R,.321,dt=.01)
        np.testing.assert_array_equal(w,[0,0])
        self.assertEqual(r.insertion_forward_command,0.)
        obs=[SimpleNamespace(position_in_robot_m=[.54,0,-.1715])]
        r.observe_onboard(obs,self.p,self.R,.33)
        w,_=r.command(self.p,self.R,.33,dt=.01)
        self.assertLessEqual(w.mean()*.11,.00400001)
    def test_person_pause_remains_immediate_and_resets_ramp(self):
        r=self.robot();r.command(self.p,self.R,.2,dt=.01)
        r.pause(.21,r.joints);w,_=r.command(self.p,self.R,.22,dt=.01)
        np.testing.assert_array_equal(w,[0,0])
        self.assertEqual(r.insertion_forward_command,0.)
        r.resume(.23);w,_=r.command(self.p,self.R,.24,dt=.01)
        self.assertLessEqual(w.mean()*.11,.00400001)
    def test_invalid_acceleration_is_rejected(self):
        for value in (0.,-.1,1.1,float('nan'),float('inf')):
            self.cfg['scoop_insertion_acceleration_m_s2']=value
            with self.assertRaises(ValueError):MissionRobot('A',self.cfg,[23.77,10.97])
        self.cfg['scoop_insertion_acceleration_m_s2']=.4
        del self.cfg['scoop_insertion_speed_m_s']
        with self.assertRaises(ValueError):MissionRobot('A',self.cfg,[23.77,10.97])

if __name__=='__main__':unittest.main()
