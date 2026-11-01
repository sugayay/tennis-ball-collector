"""No blind navigation, no unrelated-target handoff, and temporary-area guards."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from handoff_control import FixedNavigator,HandoffVisualApproach,observations_for_target,trial_position_allowed


class HandoffControlTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'handoff_config.json').read_text())
        self.position = np.array([-13.6,-6.5,.205])
        self.rotation = np.eye(3)

    def test_missing_or_stale_fixed_detection_stops(self):
        navigator = FixedNavigator(self.config)
        for seconds in (0,2):
            wheels,ready,_ = navigator.command(self.position,self.rotation,seconds)
            np.testing.assert_array_equal(wheels,[0,0])
            self.assertFalse(ready)
        navigator.observe_fixed([{'position_m':[-13.6,-7.7,.0335]}],self.position,0)
        wheels,ready,_ = navigator.command(self.position,self.rotation,2)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertFalse(ready)

    def test_fixed_detection_outside_trial_is_not_a_target(self):
        navigator = FixedNavigator(self.config)
        navigator.observe_fixed([{'position_m':[6.8,6.8,.0335]}],self.position,0)
        self.assertIsNone(navigator.target)

    def test_sideways_target_causes_rotation_before_translation(self):
        navigator = FixedNavigator(self.config)
        navigator.observe_fixed([{'position_m':[-13.6,-7.7,.0335]}],self.position,0)
        wheels,ready,status = navigator.command(self.position,self.rotation,0)
        self.assertAlmostEqual(float(wheels.sum()),0)
        self.assertFalse(ready)
        self.assertEqual(status,'fixed_aligning')

    def test_unrelated_onboard_ball_cannot_trigger_handoff(self):
        target = np.array([-12.8,-6.5,.0335])
        unrelated = SimpleNamespace(position_in_robot_m=[.8,.5,-.1715])
        self.assertEqual(observations_for_target([unrelated],self.position,self.rotation,target,.25),[])

    def test_handoff_requires_two_onboard_matches(self):
        navigator = FixedNavigator(self.config)
        navigator.observe_fixed([{'position_m':[-12.8,-6.5,.0335]}],self.position,0)
        observed = SimpleNamespace(position_in_robot_m=[.8,0,-.1715])
        for expected in (False,True):
            navigator.observe_onboard([observed],self.position,self.rotation)
            wheels,ready,_ = navigator.command(self.position,self.rotation,0)
            np.testing.assert_array_equal(wheels,[0,0])
            self.assertEqual(ready,expected)

    def test_close_sideways_target_turns_to_find_onboard_ball(self):
        navigator = FixedNavigator(self.config)
        navigator.observe_fixed([{'position_m':[-13.,-6.,.0335]}],self.position,0)
        wheels,ready,status = navigator.command(self.position,self.rotation,0)
        self.assertFalse(ready)
        self.assertEqual(status,'handoff_aligning')
        self.assertAlmostEqual(float(wheels.sum()),0)
        self.assertGreater(wheels[1],wheels[0])

    def test_close_turn_never_bypasses_freshness_or_image_matches(self):
        navigator = FixedNavigator(self.config)
        navigator.observe_fixed([{'position_m':[-13.,-6.,.0335]}],self.position,0)
        yaw=np.arctan2(.5,.6)
        rotation=np.array([[np.cos(yaw),-np.sin(yaw),0],
                           [np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
        wheels,ready,status=navigator.command(self.position,rotation,.1)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertFalse(ready)
        self.assertEqual(status,'waiting_for_onboard_match')
        wheels,ready,status=navigator.command(self.position,self.rotation,3.)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertFalse(ready)
        self.assertEqual(status,'waiting_for_fresh_fixed_detection')

    def test_trial_guard_includes_robot_clearance(self):
        self.assertTrue(trial_position_allowed(self.position,self.config))
        self.assertFalse(trial_position_allowed(np.array([-13.6,-8.1,.205]),self.config))

    def test_close_lateral_drift_stops_translation_before_contact(self):
        controller = HandoffVisualApproach([23.77,10.97])
        observed = SimpleNamespace(position_in_robot_m=[.75,.04,-.1715])
        controller.observe([observed],self.position,self.rotation,1)
        controller.aligning = False
        controller.forward_start_time = 0
        wheels,ready,status = controller.command(self.position,self.rotation,1)
        self.assertAlmostEqual(float(wheels.sum()),0)
        self.assertFalse(ready)
        self.assertEqual(status,'realigning_near_ball')

    def test_close_yaw_drift_stops_translation_before_contact(self):
        controller = HandoffVisualApproach([23.77,10.97])
        observed = SimpleNamespace(position_in_robot_m=[.75,0,-.1715])
        controller.observe([observed],self.position,self.rotation,1)
        controller.aligning = False
        controller.forward_start_time = 0
        controller.yaw_rate = .2
        wheels,ready,status = controller.command(self.position,self.rotation,1)
        self.assertAlmostEqual(float(wheels.sum()),0)
        self.assertFalse(ready)
        self.assertEqual(status,'realigning_near_ball')


if __name__ == '__main__':
    unittest.main()
