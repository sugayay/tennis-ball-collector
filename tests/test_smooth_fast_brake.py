import json,unittest
from pathlib import Path
import numpy as np
from handoff_control import FixedNavigator

class SmoothFastBrakeTests(unittest.TestCase):
 def config(self):
  d=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
  d['trial_floor_bounds_m']=[-5,-5,15,5];return d
 def test_speed_measurement_noise_never_reaccelerates_fast_brake_target(self):
  n=FixedNavigator(self.config());n.target=np.array([10.,0,0]);n.route_origin=np.array([0.,0,.205]);n.fast_route=False;n.fast_braking=True;n.brake_desired=1.;n.brake_previous_time=0.
  position=np.array([0.,0,.205]);previous=1.
  for i in range(1,100):
   t=i/120;n.last_seen=t;position[0]+=(.3 if i%2 else 1.3)/120
   wheels,ready,status=n.command(position.copy(),np.eye(3),t);forward=float(np.mean(wheels)*.11)
   self.assertLessEqual(forward,previous+1e-9);self.assertAlmostEqual(previous-forward,.4/120,places=8);self.assertFalse(ready);previous=forward
 def test_invalid_feedback_gain_is_rejected(self):
  for value in (-.1,.51,float('nan')):
   c=self.config();c['fast_brake_feedback_gain']=value
   with self.subTest(value=value):
    with self.assertRaises(ValueError):FixedNavigator(c)
if __name__=='__main__':unittest.main()
