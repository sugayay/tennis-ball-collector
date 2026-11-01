"""Driving safety at speed and recovery after loss of vision."""
import copy
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from drive_speed import DriveSpeedProfile
from handoff_control import FixedNavigator


class DriveSpeedTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())

    def test_start_does_not_jump_to_full_speed(self):
        profile = DriveSpeedProfile(self.config)
        self.assertEqual(profile.command(3., 0., 0.), 0.)
        commands = [profile.command(3., 0., i/120) for i in range(1, 61)]
        self.assertLess(commands[0], .02)
        self.assertGreater(commands[-1], commands[0])
        self.assertTrue(all(0 <= c <= self.config['maximum_travel_command_m_s'] for c in commands))

    def test_full_speed_is_available_when_braking_distance_remains(self):
        profile = DriveSpeedProfile(self.config)
        for step in range(481):
            profile.command(6., self.config['cruise_speed_m_s'], step/120)
        self.assertAlmostEqual(profile.command(6., self.config['cruise_speed_m_s'], 4.01), self.config['cruise_speed_m_s'])
        # Acceleration and braking must both permit the configured cruise speed.

    def test_reaction_and_braking_fit_before_the_slow_zone(self):
        profile = DriveSpeedProfile(self.config)
        for step in range(481):
            profile.command(3., .55, step/120)
        for distance in (1.5, 1.3, 1.2, 1.1, 1.0):
            profile.command(distance, profile.desired, profile.previous_time+.01)
            v, near, a, reaction = profile.desired, profile.near_speed, profile.deceleration, profile.reaction
            required = max(0., (v-near)*reaction+(v*v-near*near)/(2*a))
            self.assertLessEqual(required, max(0., distance-profile.near_distance)+1e-9)
        self.assertAlmostEqual(profile.desired, self.config['near_handoff_speed_m_s'])

    def test_low_measured_speed_cannot_push_command_above_braking_envelope(self):
        profile = DriveSpeedProfile(self.config)
        for step in range(481):
            profile.command(3.5, .1, step/120)
        for distance in (2.7, 2.3, 1.8, 1.3):
            command = profile.command(distance, .1, profile.previous_time+.01)
            v, near, a, reaction = command, profile.near_speed, profile.deceleration, profile.reaction
            required = max(0., (v-near)*reaction+(v*v-near*near)/(2*a))
            self.assertLessEqual(required,max(0.,distance-profile.near_distance)+1e-9)

    def test_stale_image_stops_and_fresh_image_resumes_with_a_ramp(self):
        config = copy.deepcopy(self.config)
        config['trial_floor_bounds_m'] = config['trial_zones']['A']
        nav = FixedNavigator(config)
        p = np.array([-13.6, -6.5, .205])
        nav.target = p+np.array([3., 0., -.1715])
        nav.aligning = False; nav.forward_started = 0.; nav.last_seen = 2.
        for step in range(481):
            nav.command(p, np.eye(3), step/120)
        wheels, ready, status = nav.command(p, np.eye(3), 4.)
        np.testing.assert_array_equal(wheels, [0., 0.])
        self.assertFalse(ready)
        self.assertEqual(status, 'waiting_for_fresh_fixed_detection')
        nav.last_seen = 4.01
        wheels, _, _ = nav.command(p, np.eye(3), 4.01)
        self.assertAlmostEqual(float(wheels.mean()), 0.)

    def test_invalid_motion_feedback_stops(self):
        profile = DriveSpeedProfile(self.config)
        self.assertEqual(profile.command(2., float('nan'), 1.), 0.)

    def test_invalid_driving_parameters_are_rejected(self):
        for key, value in (('travel_acceleration_m_s2', 0.),
                           ('travel_brake_deceleration_m_s2', float('nan')),
                           ('travel_slow_speed_distance_m', .9),
                           ('near_handoff_speed_m_s', 1.6)):
            config = dict(self.config); config[key] = value
            with self.assertRaises(ValueError):
                DriveSpeedProfile(config)

    def navigator_at(self, distance):
        config=dict(self.config)
        config['trial_floor_bounds_m']=config['trial_zones']['A']
        nav=FixedNavigator(config)
        p=np.array([-13.6,-6.5,.205])
        nav.target=p+np.array([distance,0.,0.])
        nav.last_seen=3.;nav.aligning=False;nav.forward_started=0.;nav.forward_speed=.22
        return nav,p

    def test_short_route_keeps_the_proven_approach_speed(self):
        nav,p=self.navigator_at(1.8)
        wheels,_,status=nav.command(p,np.eye(3),3.)
        self.assertFalse(nav.fast_route)
        self.assertAlmostEqual(float(wheels.mean())*.11,.22)
        self.assertEqual(status,'fixed_approaching')

    def test_fast_route_stops_before_lowering_and_legacy_approach(self):
        nav,p=self.navigator_at(1.4)
        nav.fast_route=True
        wheels,_,_=nav.command(p,np.eye(3),3.)
        self.assertFalse(nav.fast_route)
        self.assertTrue(nav.fast_braking)
        self.assertLessEqual(abs(float(wheels.mean())*.11),.22)
        for step in range(1,250):
            nav.last_seen=3.+step/120
            nav.command(p,np.eye(3),3.+step/120)
        self.assertFalse(nav.fast_braking)
        self.assertGreater(nav.lowering_until,3.)
        nav.target=p+np.array([2.5,0.,0.])
        nav.previous_time=None
        nav.command(p,np.eye(3),3.01)
        self.assertFalse(nav.fast_route)

    def test_long_route_enables_the_faster_driving_controller(self):
        nav,p=self.navigator_at(3.)
        nav.command(p,np.eye(3),3.)
        self.assertTrue(nav.fast_route)

    def test_fast_wheel_commands_stay_inside_declared_14_rad_s_model_limit(self):
        nav,p=self.navigator_at(3.9)
        for step in range(601):
            nav.last_seen=step/120
            wheels,_,_=nav.command(p,np.eye(3),step/120)
            self.assertLessEqual(float(np.max(np.abs(wheels))),14.)
        self.assertGreater(float(wheels.mean())*.11,.66)

    def test_wheel_commands_cannot_exceed_the_urdf_limit(self):
        config=dict(self.config);config['maximum_wheel_speed_rad_s']=15.
        with self.assertRaises(ValueError):
            DriveSpeedProfile(config)

    def test_configured_wheel_speed_is_within_both_model_motor_limits(self):
        model = ET.parse(Path(__file__).resolve().parents[1]/'assets/collector_beveled.urdf').getroot()
        for name in ('left_wheel_joint', 'right_wheel_joint'):
            limit = float(model.find(f"joint[@name='{name}']/limit").attrib['velocity'])
            self.assertLessEqual(self.config['maximum_wheel_speed_rad_s'], limit)

    def test_fast_line_following_steers_back_after_a_lateral_displacement(self):
        nav,p=self.navigator_at(3.9)
        nav.fast_route=True;nav.route_origin=p.copy();nav.forward_speed=1.
        nav.speed_profile.desired=1.;nav.speed_profile.previous_time=2.99
        wheels,_,_=nav.command(p+np.array([.5,.1,0.]),np.eye(3),3.)
        self.assertGreater(wheels[0],wheels[1])
        self.assertGreater(float(wheels.mean()),0.)
        self.assertLessEqual(float(np.max(np.abs(wheels))),14.)


if __name__ == '__main__':
    unittest.main()
