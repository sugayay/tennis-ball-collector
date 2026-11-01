import copy
import json
import math
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from mission_control import FleetCoordinator, MissionRobot, MissionVisualApproach


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
        self.config.pop('test_ball_positions_m')
        self.positions = {'A':np.array([-13.6,-6.5,.205]),'B':np.array([13.6,6.5,.205])}
        self.observations = [{'position_m':[x,y,.0335]} for x,y in [(-12.8,-6.5),(-11.5,-6.5),(12.8,6.5),(11.5,6.5)]]

    def test_disjoint_assignment_and_no_duplicate_reservation(self):
        fleet = FleetCoordinator(self.config,[23.77,10.97])
        for seconds in (0,.5):
            fleet.observe_fixed(self.observations,self.positions,seconds)
        self.assertEqual(len(fleet.assignments),2)
        self.assertLess(fleet.robots['A'].target[0],-12)
        self.assertGreater(fleet.robots['B'].target[0],12)

    def test_robot_basket_detection_is_not_new_ground_target(self):
        fleet = FleetCoordinator(self.config,[23.77,10.97])
        fleet.observe_fixed([{'position_m':[-13.6,-6.3,.0335]}],self.positions,0)
        self.assertEqual(fleet.assignments,[])

    def test_completed_target_not_reassigned(self):
        fleet = FleetCoordinator(self.config,[23.77,10.97])
        robot = fleet.robots['A']
        robot.completed_targets = [np.array(self.observations[0]['position_m'])]
        fleet.observe_fixed(self.observations,self.positions,0)
        self.assertAlmostEqual(robot.target[0],-11.5)

    def test_no_detection_keeps_both_robots_stopped(self):
        fleet = FleetCoordinator(self.config,[23.77,10.97])
        fleet.observe_fixed([],self.positions,0)
        for rid,robot in fleet.robots.items():
            wheels,_ = robot.command(self.positions[rid],np.eye(3),0)
            np.testing.assert_array_equal(wheels,[0,0])

    def test_stale_target_stops_then_bounded_retries_end_in_fault(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign(self.observations[0],self.positions['A'],0)
        for seconds in (1.2,1.7,3.4,5.1,7):
            wheels,_ = robot.command(self.positions['A'],np.eye(3),seconds)
            np.testing.assert_array_equal(wheels,[0,0])
        self.assertEqual(robot.stage,'fault')
        self.assertEqual(robot.retry_total,2)

    def test_retry_requires_a_fresh_matching_detection(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign(self.observations[0],self.positions['A'],0)
        robot.command(self.positions['A'],np.eye(3),1.7)
        robot.observe_fixed([self.observations[1]],self.positions['A'],2.5)
        robot.command(self.positions['A'],np.eye(3),2.5)
        self.assertEqual(robot.stage,'retry')
        robot.observe_fixed([self.observations[0]],self.positions['A'],2.6)
        robot.command(self.positions['A'],np.eye(3),2.6)
        self.assertEqual(robot.stage,'fixed')

    def test_separation_fault_stops_both(self):
        fleet = FleetCoordinator(self.config,[23.77,10.97])
        fleet.observe_fixed(self.observations,self.positions,0)
        fleet.guard_separation({'A':np.array([-13.6,-6.5,.205]),'B':np.array([-13.0,-6.5,.205])},1)
        self.assertTrue(all(r.stage=='fault' for r in fleet.robots.values()))

    def test_outside_trial_stops_even_with_fresh_detection(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign(self.observations[0],self.positions['A'],0)
        wheels,_ = robot.command(np.array([-13.6,-5.8,.205]),np.eye(3),0)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertEqual(robot.stage,'fault')

    def test_persistent_no_onboard_detection_navigation_timeout_is_bounded(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign(self.observations[0],self.positions['A'],0)
        robot.observe_fixed([self.observations[0]],self.positions['A'],33)
        wheels,_ = robot.command(self.positions['A'],np.eye(3),33)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertEqual(robot.stage,'retry')

    def test_unrelated_visual_detection_cannot_start_pickup(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign(self.observations[0],self.positions['A'],0)
        robot.transition('visual',0)
        from handoff_control import HandoffVisualApproach
        robot.visual = HandoffVisualApproach([23.77,10.97])
        robot.observe_onboard([SimpleNamespace(position_in_robot_m=[.5,0,-.1715])],self.positions['A'],np.eye(3),.1)
        wheels,_ = robot.command(self.positions['A'],np.eye(3),.1)
        np.testing.assert_array_equal(wheels,[0,0])
        self.assertEqual(robot.stage,'visual')

    def test_visual_to_scoop_has_no_intermediate_brake_pulse(self):
        from handoff_control import HandoffVisualApproach
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign({'position_m':[-13.05,-6.5,.0335]},self.positions['A'],0)
        robot.transition('visual',0)
        robot.visual = HandoffVisualApproach([23.77,10.97])
        robot.visual.aligning = False
        robot.visual.forward_start_time = 0
        for seconds in (.95,1.):
            robot.observe_onboard([SimpleNamespace(position_in_robot_m=[.55,0,-.1715])],self.positions['A'],np.eye(3),seconds)
        wheels,joints = robot.command(self.positions['A'],np.eye(3),1.)
        self.assertEqual(robot.stage,'scoop')
        self.assertAlmostEqual(float(wheels.mean()),6.)
        self.assertGreaterEqual(joints[0],-.448)

    def test_retry_does_not_resume_for_matching_ball_outside_trial(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign({'position_m':[-12.8,-5.55,.0335]},self.positions['A'],0)
        robot.fail_or_retry(1,'test_loss')
        robot.observe_fixed([{'position_m':[-12.8,-5.4,.0335]}],self.positions['A'],1.8)
        wheels,_ = robot.command(self.positions['A'],np.eye(3),1.8)
        self.assertEqual(robot.stage,'retry')
        np.testing.assert_array_equal(wheels,[0,0])

    def distance_controlled_scoop(self,insertion=0.):
        self.config['scoop_distance_m'] = .10
        self.config['scoop_preinsert_m'] = insertion
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign(self.observations[0],self.positions['A'],0)
        robot.transition('scoop',0)
        robot.heading_reference = 0.
        robot.scoop_start_position = self.positions['A'].copy()
        return robot

    def test_scoop_waits_for_measured_travel_instead_of_elapsed_time(self):
        robot = self.distance_controlled_scoop()
        wheels,joints = robot.command(self.positions['A']+np.array([.04,0,0]),np.eye(3),.4)
        self.assertEqual(robot.stage,'scoop')
        self.assertAlmostEqual(float(wheels.mean()),6.)
        self.assertLess(joints[0],-.20)
        robot.command(self.positions['A']+np.array([.10,0,0]),np.eye(3),.8)
        self.assertEqual(robot.stage,'secure')

    def test_stalled_scoop_stops_with_bounded_timeout(self):
        robot = self.distance_controlled_scoop()
        wheels,_ = robot.command(self.positions['A'],np.eye(3),1.51)
        self.assertEqual(robot.stage,'fault')
        self.assertEqual(robot.events[-1]['reason'],'scoop_progress_timeout')
        np.testing.assert_array_equal(wheels,[0,0])

    def test_scoop_rate_limit_prevents_an_abrupt_lift_at_high_speed(self):
        robot = self.distance_controlled_scoop()
        robot.config['minimum_scoop_seconds'] = .30
        robot.command(self.positions['A']+np.array([.20,0,0]),np.eye(3),.1)
        self.assertEqual(robot.stage,'scoop')
        robot.command(self.positions['A']+np.array([.20,0,0]),np.eye(3),.3)
        self.assertEqual(robot.stage,'secure')

    def test_invalid_scoop_travel_is_rejected(self):
        for distance in (0,-.1,.5,float('nan')):
            with self.subTest(distance=distance):
                self.config['scoop_distance_m'] = distance
                with self.assertRaises(ValueError):
                    MissionRobot('A',self.config,[23.77,10.97])

    def approaching_visual(self, forward, lateral=0.):
        visual = MissionVisualApproach([23.77,10.97])
        visual.world_target = self.positions['A']+np.array([forward,lateral,-.1715])
        visual.last_seen = 1.
        visual.detection_updates = 2
        visual.aligning = False
        visual.forward_start_time = 0.
        visual.yaw_rate = .20
        return visual

    def test_aligned_ball_starts_pickup_before_front_edge_contact(self):
        visual = self.approaching_visual(.60)
        wheels,ready,status = visual.command(self.positions['A'],np.eye(3),1.)
        self.assertTrue(ready)
        self.assertEqual(status,'at_shovel')

    def test_yaw_drift_is_damped_without_repeated_braking(self):
        visual = self.approaching_visual(.70)
        wheels,_,status = visual.command(self.positions['A'],np.eye(3),1.)
        self.assertEqual(status,'approaching')
        self.assertGreater(float(wheels.mean()),0.)
        self.assertGreater(wheels[0],wheels[1])

    def test_off_axis_ball_near_the_front_edge_still_requires_alignment(self):
        visual = self.approaching_visual(.60,.08)
        wheels,_,status = visual.command(self.positions['A'],np.eye(3),1.)
        self.assertEqual(status,'realigning_before_contact')
        self.assertAlmostEqual(float(wheels.mean()),0.)

    def test_inner_approach_uses_the_same_lateral_corridor_as_pickup_readiness(self):
        visual = self.approaching_visual(.60,.03)
        _,ready,status = visual.command(self.positions['A'],np.eye(3),1.)
        self.assertTrue(ready)
        self.assertEqual(status,'at_shovel')

    def test_ball_fitting_shovel_width_can_trigger_without_exact_centring(self):
        visual = self.approaching_visual(.59,.055)
        visual.yaw_rate = 0.
        _,ready,_ = visual.command(self.positions['A'],np.eye(3),1.)
        self.assertTrue(ready)

    def test_turning_ball_must_fit_the_predicted_lateral_envelope(self):
        visual = self.approaching_visual(.59,.060)
        visual.yaw_rate = .30
        _,ready,_ = visual.command(self.positions['A'],np.eye(3),1.)
        self.assertFalse(ready)

    def test_new_contact_trigger_still_requires_fresh_image_and_initial_alignment(self):
        for stale,aligning,updates in ((True,False,2),(False,True,2),(False,False,1)):
            visual = self.approaching_visual(.59,.04)
            visual.last_seen = .5 if stale else 1.
            visual.aligning = aligning
            visual.detection_updates = updates
            _,ready,_ = visual.command(self.positions['A'],np.eye(3),1.)
            self.assertFalse(ready)

    def test_rolling_target_tracks_adjacent_image_updates_but_rejects_a_distant_ball(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign({'position_m':[-12.8,-6.5,.0335]},self.positions['A'],0)
        robot.transition('visual',0)
        robot.visual = MissionVisualApproach([23.77,10.97])
        for seconds,forward in ((.05,.9),(.10,1.0),(.15,1.1),(.20,1.2)):
            robot.observe_onboard([SimpleNamespace(position_in_robot_m=[forward,0,-.1715])],self.positions['A'],np.eye(3),seconds)
        self.assertGreater(robot.target[0],-12.55)
        np.testing.assert_array_equal(robot.navigator.target,robot.target)
        self.assertEqual(robot.navigator.last_seen,0.)
        last = robot.target.copy()
        robot.observe_onboard([SimpleNamespace(position_in_robot_m=[1.8,0,-.1715])],self.positions['A'],np.eye(3),.25)
        np.testing.assert_array_equal(robot.target,last)
        self.assertEqual(robot.visual.last_seen,.20)

    def test_preinsertion_keeps_the_shovel_flat_until_measured_travel(self):
        robot = self.distance_controlled_scoop(insertion=.02)
        _,joints = robot.command(self.positions['A']+np.array([.01,0,0]),np.eye(3),.4)
        self.assertEqual(robot.stage,'scoop')
        self.assertAlmostEqual(joints[0],-.448)
        robot.command(self.positions['A']+np.array([.10,0,0]),np.eye(3),.5)
        self.assertEqual(robot.stage,'scoop')
        robot.command(self.positions['A']+np.array([.10,0,0]),np.eye(3),.8)
        self.assertEqual(robot.stage,'secure')

    def test_preinsertion_requires_a_valid_remaining_lift_distance(self):
        for insertion in (-.01,self.config['scoop_distance_m'],float('nan')):
            with self.subTest(insertion=insertion):
                self.config['scoop_preinsert_m'] = insertion
                with self.assertRaises(ValueError):
                    MissionRobot('A',self.config,[23.77,10.97])

    def test_pickup_velocity_feedback_reduces_drive_when_the_chassis_is_too_fast(self):
        robot = self.distance_controlled_scoop()
        robot.config['pickup_forward_speed_m_s'] = .30
        robot.filtered_forward_speed = .45
        wheels,_ = robot.command(self.positions['A'],np.eye(3),.1)
        self.assertAlmostEqual(float(wheels.mean()),0.)
        self.assertEqual(robot.stage,'scoop')

    def test_visual_to_scoop_preserves_yaw_rate_for_immediate_damping(self):
        robot = MissionRobot('A',self.config,[23.77,10.97])
        robot.assign(self.observations[0],self.positions['A'],0)
        robot.transition('visual',0)
        robot.visual = SimpleNamespace(yaw_rate=.20,command=lambda *args:(np.zeros(2),True,'at_shovel'))
        wheels,_ = robot.command(self.positions['A'],np.eye(3),1.)
        self.assertEqual(robot.stage,'scoop')
        self.assertAlmostEqual(robot.heading_rate,.20)
        self.assertGreater(wheels[0],wheels[1])



if __name__=='__main__':
    unittest.main()
