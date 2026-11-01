import unittest
from person_safety import PersonPauseGate

class PersonResumeTests(unittest.TestCase):
    def gate(self):
        g=PersonPauseGate({},['A','B'])
        g.observe('A',[{'position_m':[9,0,0]}],0);g.observe('B',[],0)
        self.assertTrue(g.blocked([0,0,0],1.5,1.5,0))
        return g

    def clear(self,g,t):
        # Clearance needs a still-observed person beyond the resume radius.
        g.observe('A',[{'position_m':[10,0,0]}],t)
        g.observe('B',[],t)

    def test_both_cameras_must_stay_clear_for_multiple_frames(self):
        g=self.gate()
        for t in (.5,1.,1.5):
            self.clear(g,t);self.assertTrue(g.blocked([0,0,0],0,0,t))
        self.clear(g,2.);self.assertFalse(g.blocked([0,0,0],0,0,2.))
        self.assertEqual(g.resume_count,1)

    def test_stale_or_repeated_images_cannot_complete_clearance(self):
        g=self.gate();self.clear(g,.5)
        for t in (.5,.6,.7,.8,1.,2.):self.assertTrue(g.blocked([0,0,0],0,0,t))
        self.assertEqual(g.resume_count,0)

    def test_person_return_resets_clearance_timer(self):
        g=self.gate()
        for t in (.5,1.):self.clear(g,t);g.blocked([0,0,0],0,0,t)
        g.observe('A',[{'position_m':[9,0,0]}],1.5);g.observe('B',[],1.5)
        self.assertTrue(g.blocked([0,0,0],0,0,1.5))
        for t in (2.,2.5,3.):self.clear(g,t);self.assertTrue(g.blocked([0,0,0],0,0,t))

    def test_unknown_person_position_does_not_auto_resume(self):
        g=PersonPauseGate({},['A']);g.observe('A',[],0,1)
        self.assertTrue(g.blocked([0,0,0],0,0,0))
        for t in (.5,1,1.5,2,2.5):g.observe('A',[],t);self.assertTrue(g.blocked([0,0,0],0,0,t))

    def test_camera_failure_during_person_pause_requires_manual_reset(self):
        g=self.gate()
        self.assertTrue(g.blocked([0,0,0],0,0,1.))
        self.assertEqual(g.event['reason'],'safety_vision_stale')
        for t in (1.5,2.,2.5,3.,3.5):
            self.clear(g,t);self.assertTrue(g.blocked([0,0,0],0,0,t))
