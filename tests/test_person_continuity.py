"""Safety properties for fresh frames whose person detections disappear."""
import unittest

from person_safety import PersonPauseGate, PersonStopGate


def person(x, y=0., paired=None):
    result = dict(position_m=[x, y, 0.])
    if paired:
        result.update(localization_method='dual_view_head_triangulation',
                      paired_camera_id=paired)
    return result


def observe(gate, seconds, a, b):
    gate.observe('A', a, seconds)
    gate.observe('B', b, seconds)


class PersonContinuityTests(unittest.TestCase):
    def test_empty_fresh_frames_after_person_pause_never_resume(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(2.)], [])
        self.assertTrue(g.blocked([0, 0, 0], .2, .2, 0.))
        for t in (.15, .3, .6, 1., 1.5, 2., 2.5):
            observe(g, t, [], [])
            self.assertTrue(g.blocked([0, 0, 0], 0., 0., t))
        self.assertEqual(g.event['reason'], 'person_track_lost')
        self.assertEqual(g.resume_count, 0)

    def test_lost_distant_person_stops_a_moving_robot(self):
        for gate_type in (PersonStopGate, PersonPauseGate):
            with self.subTest(gate=gate_type):
                g = gate_type({}, ['A', 'B'])
                observe(g, 0., [person(20., paired='B')], [person(20., paired='A')])
                self.assertFalse(g.blocked([0, 0, 0], 1.4, 1.4, 0.))
                observe(g, .15, [], [])
                self.assertTrue(g.blocked([0, 0, 0], 1.4, 1.4, .15))
                self.assertEqual(g.event['reason'], 'person_track_lost')

    def test_one_camera_loss_with_unique_other_view_keeps_tracking(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(20., paired='B')], [person(20., paired='A')])
        self.assertFalse(g.blocked([0, 0, 0], 1.4, 1.4, 0.))
        for index in range(1, 20):
            t = index * .15
            observe(g, t, [], [person(20. + t)])
            self.assertFalse(g.blocked([0, 0, 0], 1.4, 1.4, t))

    def test_one_remaining_person_cannot_clear_two_previous_people(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(2.), person(20.)], [])
        self.assertTrue(g.blocked([0, 0, 0], .2, .2, 0.))
        observe(g, .15, [person(20.)], [])
        self.assertTrue(g.blocked([0, 0, 0], 0., 0., .15))
        self.assertEqual(g.event['reason'], 'person_track_lost')

    def test_unrelated_distant_person_cannot_replace_missing_person(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(2.)], [])
        g.blocked([0, 0, 0], 0., 0., 0.)
        observe(g, .15, [person(20.)], [])
        self.assertTrue(g.blocked([0, 0, 0], 0., 0., .15))
        self.assertEqual(g.event['reason'], 'person_track_lost')

    def test_ambiguous_association_requires_stop(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(20.), person(20.3)], [])
        self.assertFalse(g.blocked([0, 0, 0], 1.4, 1.4, 0.))
        observe(g, .15, [person(20.1), person(20.2)], [])
        self.assertTrue(g.blocked([0, 0, 0], 1.4, 1.4, .15))
        self.assertEqual(g.event['reason'], 'person_identity_ambiguous')

    def test_reappearance_does_not_reset_a_tracking_fault(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(2.)], [])
        g.blocked([0, 0, 0], 0., 0., 0.)
        observe(g, .15, [], [])
        g.blocked([0, 0, 0], 0., 0., .15)
        for t in (.3, .6, 1., 1.5, 2., 2.5):
            observe(g, t, [person(20.)], [])
            self.assertTrue(g.blocked([0, 0, 0], 0., 0., t))
        self.assertEqual(g.resume_count, 0)

    def test_continuously_observed_departure_can_resume(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(2., paired='B')], [person(2., paired='A')])
        g.blocked([0, 0, 0], 0., 0., 0.)
        resumed = False
        for index in range(1, 121):
            t = index * .1
            observe(g, t, [], [person(2. + 1.4*t)])
            resumed |= not g.blocked([0, 0, 0], 0., 0., t)
        self.assertTrue(resumed)
        self.assertEqual(g.resume_count, 1)

    def test_empty_scene_without_prior_detection_remains_an_explicit_limit(self):
        g = PersonPauseGate({}, ['A', 'B'])
        for t in (0., .15, .3):
            observe(g, t, [], [])
            self.assertFalse(g.blocked([0, 0, 0], 1.4, 1.4, t))

    def test_duplicate_capture_cannot_erase_a_person(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(2.)], [])
        observe(g, 0., [], [])
        self.assertTrue(g.blocked([0, 0, 0], 0., 0., 0.))
        self.assertEqual(g.event['reason'], 'person_proximity_stop')

    def test_cross_camera_batch_mismatch_cannot_claim_clearance(self):
        g = PersonPauseGate({}, ['A', 'B'])
        observe(g, 0., [person(20., paired='B')], [person(20., paired='A')])
        g.blocked([0, 0, 0], 1.4, 1.4, 0.)
        g.observe('A', [], .15)
        self.assertTrue(g.blocked([0, 0, 0], 1.4, 1.4, .15))
        self.assertEqual(g.event['reason'], 'person_frames_unsynchronized')


if __name__ == '__main__':
    unittest.main()
