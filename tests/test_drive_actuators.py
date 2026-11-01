import unittest
from types import SimpleNamespace
import numpy as np
from drive_actuators import settings,configure_wheels

class Robot:
 def __init__(self):self.calls={}
 def get_joint(self,name):return SimpleNamespace(dofs_idx_local=[7 if name=='left_wheel_joint' else 4])
 def set_dofs_kp(self,x,**kw):self.calls['kp']=(x,kw)
 def set_dofs_kv(self,x,**kw):self.calls['kv']=(x,kw)
 def set_dofs_force_range(self,a,b,**kw):self.calls['force']=(a,b,kw)
 def control_dofs_velocity(self,x,**kw):self.calls['velocity']=(x,kw)

class WheelActuatorTests(unittest.TestCase):
 def test_source_configuration_is_lower_torque_without_speed_reduction(self):self.assertEqual(settings(),(12.,2.))
 def test_actuation_uses_named_dofs_and_symmetric_physical_limit(self):
  r=Robot();self.assertEqual(configure_wheels(r),[7,4]);a,b,k=r.calls['force']
  np.testing.assert_array_equal(a,[-2,-2]);np.testing.assert_array_equal(b,[2,2]);self.assertEqual(k['dofs_idx_local'],[7,4])
  np.testing.assert_array_equal(r.calls['kv'][0],[12,12]);np.testing.assert_array_equal(r.calls['velocity'][0],[0,0])
 def test_invalid_or_greater_than_baseline_torque_is_rejected(self):
  for n in (0,-1,6.01,True,float('nan'),float('inf')):
   with self.subTest(n=n):
    with self.assertRaises(ValueError):settings({'wheel_velocity_gain':12.,'maximum_wheel_torque_nm':n})
 def test_invalid_gain_is_rejected(self):
  for n in (0,12.1,True,float('nan')):
   with self.subTest(n=n):
    with self.assertRaises(ValueError):settings({'wheel_velocity_gain':n,'maximum_wheel_torque_nm':2.})
 def test_bad_configuration_never_mutates_actuator(self):
  r=Robot()
  with self.assertRaises(ValueError):configure_wheels(r,{'wheel_velocity_gain':12.,'maximum_wheel_torque_nm':float('nan')})
  self.assertEqual(r.calls,{})
if __name__=='__main__':unittest.main()
