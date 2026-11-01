"""Behavior contracts: absent/stale detections and explicit court permission."""
import unittest
from types import SimpleNamespace

import numpy as np

from vision_control import VisualApproach, court_position_allowed


class VisionControlTests(unittest.TestCase):
    def setUp(self):
        self.area = [23.77,10.97]
        self.position = np.array([-13.6,-6.5,0.205])
        self.rotation = np.eye(3)

    def observation(self, x, y):
        return SimpleNamespace(position_in_robot_m=[x,y,-0.1715])

    def test_absent_detections_do_not_move_or_trigger_pickup(self):
        controller = VisualApproach(self.area)
        controller.observe([], self.position, self.rotation, 0)
        wheels, ready, status = controller.command(self.position,self.rotation,1)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertFalse(ready)
        self.assertEqual(status,'waiting_for_ball')

    def test_stale_detection_stops_motion(self):
        controller = VisualApproach(self.area)
        controller.observe([self.observation(0.8,0)],self.position,self.rotation,0)
        wheels, ready, _ = controller.command(self.position,self.rotation,0.2)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertFalse(ready)

    def test_practice_rejects_inside_ball_but_break_can_accept_it(self):
        position = np.array([-12.8,0,0.205])
        observations = [self.observation(1.0,0)]
        for mode, accepted in [('practice',False),('break',True)]:
            controller = VisualApproach(self.area,mode)
            controller.observe(observations,position,self.rotation,1)
            self.assertEqual(controller.world_target is not None,accepted)

    def test_braking_lookahead_blocks_court_entry(self):
        position = np.array([-12.6,0,0.205])
        controller = VisualApproach(self.area,'practice')
        controller.observe([self.observation(0.6,0)],position,self.rotation,1)
        controller.command(position,self.rotation,1)
        controller.observe([self.observation(0.6,0)],position,self.rotation,1.5)
        wheels,ready,status = controller.command(position,self.rotation,1.5)
        self.assertFalse(ready)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertEqual(status,'court_entry_blocked')
        self.assertTrue(court_position_allowed([-12.45,0,0],self.area,'break'))

    def test_close_lateral_ball_is_aligned_before_advancing(self):
        controller = VisualApproach(self.area)
        controller.observe([self.observation(0.65,-0.1)],self.position,self.rotation,1)
        wheels,ready,status = controller.command(self.position,self.rotation,1)
        self.assertFalse(ready)
        self.assertEqual(status,'aligning')
        self.assertAlmostEqual(float(wheels.sum()),0)
        self.assertGreater(wheels[0],wheels[1])


if __name__ == '__main__':
    unittest.main()
