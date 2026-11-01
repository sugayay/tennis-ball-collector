import unittest
from lift_storage import stored,StorageInterlock

class LiftStorageTests(unittest.TestCase):
    def test_all_axes_at_rest(self):self.assertTrue(stored([0.]*4,[0.]*4))
    def test_one_extended_axis_denies_motion(self):
        for q in ([.021,0,0,0],[0,.021,0,0],[0,0,.041,0],[0,0,0,.021]):self.assertFalse(stored(q,[0.]*4))
    def test_closed_but_moving_denies_motion(self):self.assertFalse(stored([0.]*4,[0,0,0,.061]))
    def test_nonfinite_denies_motion(self):self.assertFalse(stored([0,0,float('nan'),0],[0.]*4))
    def test_missing_joint_denies_motion(self):self.assertFalse(stored([0.]*3,[0.]*4))

class TransitStorageTests(unittest.TestCase):
    def gate(self):
        g=StorageInterlock();self.assertTrue(g.observe([0.]*4,[0.]*4,0.)['home']);return g
    def test_small_short_vibration_is_not_extension(self):
        g=self.gate();o=g.observe([0,.002,0,0],[0,.08,0,0],.1)
        self.assertTrue(o['home']);self.assertFalse(o['settled'])
        self.assertTrue(g.observe([0,.001,0,0],[0,.02,0,0],.15)['home'])
    def test_persistent_motion_latches_after100ms(self):
        g=self.gate();g.observe([0,.002,0,0],[0,.08,0,0],.1)
        self.assertFalse(g.observe([0,.003,0,0],[0,.08,0,0],.2)['home'])
        self.assertEqual(g.fault,'persistent_lift_storage_motion')
        self.assertFalse(g.observe([0.]*4,[0.]*4,.3)['home'])
    def test_position_violation_is_immediate(self):
        g=self.gate();self.assertFalse(g.observe([0,.021,0,0],[0.]*4,.01)['home'])
    def test_mechanical_velocity_limit_is_immediate(self):
        g=self.gate();self.assertFalse(g.observe([0,.001,0,0],[0,.141,0,0],.01)['home'])
    def test_arm_requires_original_stillness(self):
        g=StorageInterlock();self.assertFalse(g.observe([0.]*4,[0,.08,0,0],0.)['home'])
        self.assertEqual(g.fault,'lift_not_settled_before_travel')
    def test_invalid_numbers_latch(self):
        g=self.gate();self.assertFalse(g.observe([float('nan'),0,0,0],[0.]*4,.1)['home'])
    def test_clock_reversal_latches(self):
        g=self.gate();self.assertFalse(g.observe([0.]*4,[0.]*4,-1.)['home'])

if __name__=='__main__':unittest.main()
