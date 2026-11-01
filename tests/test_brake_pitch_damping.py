import unittest,json,math
from pathlib import Path
import numpy as np
from handoff_control import damped_braking_speed,FixedNavigator
class Tests(unittest.TestCase):
 def test_backward_pitch_or_pitch_rate_reduces_brake_speed(self):
  self.assertLess(damped_braking_speed(.6,-.1,0.,.5,.15),.6)
  self.assertLess(damped_braking_speed(.6,0.,-1.,.5,.15),.6)
 def test_compensation_can_never_boost_reverse_or_exceed_25cm_s(self):
  for v in (0.,.1,.4,1.4):
   for p in (-.3,0.,.3):
    for rate in (-3.,0.,3.):
     out=damped_braking_speed(v,p,rate,.5,.15)
     self.assertGreaterEqual(out,0.);self.assertLessEqual(out,v);self.assertLessEqual(v-out,.25+1e-10)
 def test_invalid_attitude_is_not_a_nonfinite_motor_command(self):
  for bad in (float('nan'),float('inf')):
   self.assertEqual(damped_braking_speed(.6,bad,0.,.5,.15),0.)
 def test_disabled_gains_preserve_existing_braking_speed(self):
  self.assertEqual(damped_braking_speed(.6,-.2,-1.,0.,0.),.6)
 def test_pitch_history_can_start_after_pose_time_rebase(self):
  c=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
  c['trial_floor_bounds_m']=[-2.,-2.,12.,2.]
  nav=FixedNavigator(c);nav.target=np.array([8.,0.,0.]);nav.last_seen=1.
  nav.previous_position=np.array([0.,0.,.205]);nav.previous_yaw=0.;nav.previous_time=.99
  wheels,_,_=nav.command(nav.previous_position,np.eye(3),1.)
  self.assertTrue(np.isfinite(wheels).all());self.assertEqual(nav.pitch_rate,0.)
 def test_invalid_gain_is_rejected(self):
  c=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
  for g in (True,-.1,1.01,float('nan')):
   with self.subTest(g=g):
    with self.assertRaises(ValueError):FixedNavigator({**c,'brake_pitch_damping_kp':g})
if __name__=='__main__':unittest.main()
