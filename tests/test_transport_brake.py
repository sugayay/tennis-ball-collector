import unittest
from types import SimpleNamespace
import numpy as np
from lift_storage import LiftStorage
class Tensor:
 def __init__(self,x):self.x=np.array(x)
 def detach(self):return self
 def cpu(self):return self
 def numpy(self):return self.x
class Robot:
 def __init__(self):self.q=[0.]*4;self.v=[0.]*4;self.brakes=[]
 def get_joint(self,name):return SimpleNamespace(dofs_idx_local=[{'refill_lift_joint':3,'refill_slide_joint':5,'refill_tip_joint':6,'refill_gate_joint':8}[name]])
 def get_dofs_position(self,**kw):return Tensor(self.q)
 def get_dofs_velocity(self,**kw):return Tensor(self.v)
 def set_dofs_position(self,*a,**kw):pass
 def set_dofs_kp(self,*a,**kw):pass
 def set_dofs_kv(self,*a,**kw):pass
 def set_dofs_force_range(self,*a,**kw):pass
 def control_dofs_position(self,*a,**kw):pass
 def set_dofs_frictionloss(self,a,**kw):self.brakes.append((a.tolist(),kw))
class TransportBrakeTests(unittest.TestCase):
 def setUp(self):self.robot=Robot();self.storage=LiftStorage(self.robot)
 def test_initial_physical_brake_uses_named_slide_dof(self):
  a,k=self.robot.brakes[-1];self.assertEqual(a,[0.,30.,0.,0.]);self.assertEqual(k['dofs_idx_local'],[3,5,6,8]);self.assertTrue(self.storage.brake_engaged)
 def test_release_only_when_stationary_clear_and_settled(self):
  self.assertFalse(self.storage.release_for_service(.03,safety_clear=True));self.assertFalse(self.storage.release_for_service(0.,safety_clear=False))
  self.robot.v[1]=.07;self.assertFalse(self.storage.release_for_service(0.,safety_clear=True));self.assertEqual(len(self.robot.brakes),1)
 def test_release_removes_physical_friction_and_denies_travel(self):
  self.assertTrue(self.storage.release_for_service(0.,safety_clear=True));self.assertEqual(self.robot.brakes[-1][0],[0.]*4)
  self.assertFalse(self.storage.observation()['home']);self.assertFalse(self.storage.check(0.)['home']);self.assertEqual(self.storage.fault,'lift_transport_brake_not_engaged')
 def test_extended_mechanism_cannot_be_locked_for_travel(self):
  self.storage.release_for_service(0.,safety_clear=True);self.robot.q[1]=.5
  self.assertFalse(self.storage.engage_for_travel(0.,safety_clear=True));self.assertFalse(self.storage.brake_engaged)
 def test_service_release_and_reengage_does_not_clear_prior_fault(self):
  self.storage.check(0.);self.storage.fault='existing_fault'
  self.assertFalse(self.storage.release_for_service(0.,safety_clear=True));self.assertFalse(self.storage.engage_for_travel(0.,safety_clear=True));self.assertEqual(self.storage.fault,'existing_fault')
if __name__=='__main__':unittest.main()
