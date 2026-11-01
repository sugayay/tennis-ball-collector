import unittest
import numpy as np
from person_safety import PersonStopGate
from fixed_detection import localize_people


class PersonSafetyTests(unittest.TestCase):
    def fresh(self):
        g=PersonStopGate({},['A','B'])
        for camera in ('A','B'):g.observe(camera,[],0.)
        return g

    def test_movement_requires_both_fresh_cameras(self):
        g=PersonStopGate({},['A','B']);g.observe('A',[],0.)
        self.assertTrue(g.blocked([0,0,0],0.,1.,0.))
        self.assertEqual(g.event['reason'],'safety_vision_stale')

    def test_clear_fresh_images_allow_motion(self):
        self.assertFalse(self.fresh().blocked([0,0,0],1.,1.,.2))

    def test_close_person_latches_after_leaving(self):
        g=self.fresh();g.observe('A',[{'position_m':[3.,0.,0.]}],.1);g.observe('B',[],.1)
        self.assertTrue(g.blocked([0,0,0],1.,1.,.1))
        self.assertEqual(g.event['reason'],'person_proximity_stop')
        for camera in ('A','B'):g.observe(camera,[],.2)
        self.assertTrue(g.blocked([0,0,0],0.,0.,.2))

    def test_higher_speed_expands_stop_radius(self):
        g=self.fresh()
        self.assertGreater(g.stopping_radius(1.),g.stopping_radius(.2)+.8)

    def test_one_stale_camera_stops(self):
        g=self.fresh();g.observe('A',[],.7)
        self.assertTrue(g.blocked([0,0,0],.8,1.,.7))
        self.assertEqual(g.event['camera_id'],'B')

    def test_unlocalized_detection_stops(self):
        g=self.fresh();g.observe('A',[],.1,1)
        self.assertTrue(g.blocked([0,0,0],0.,0.,.1))
        self.assertEqual(g.event['reason'],'person_position_unknown')

    def test_invalid_feedback_stops(self):
        g=self.fresh()
        self.assertTrue(g.blocked([float('nan'),0,0],0.,0.,0.))

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):PersonStopGate({'braking_deceleration_m_s2':0.},['A'])

    def test_foot_uses_bbox_bottom_not_centre(self):
        calibration={'id':'A','intrinsics':np.array([[100,0,100],[0,100,100],[0,0,1]]).tolist(),
            'transform':np.array([[1,0,0,0],[0,1,0,0],[0,0,1,10],[0,0,0,1]]).tolist()}
        people,unknown=localize_people([{'bbox_xyxy':[90,50,110,100],'confidence':.9}],calibration)
        self.assertEqual(unknown,0)
        np.testing.assert_allclose(people[0]['position_m'],[0,0,0])

    def test_future_and_out_of_order_frames_cannot_fake_freshness(self):
        g=self.fresh();g.observe('A',[],1.);g.observe('A',[],.2)
        self.assertTrue(g.blocked([0,0,0],0.,0.,.3))


if __name__=='__main__':unittest.main()
