"""Protect phase-specific travel speed, braking, and alignment before contact."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from handoff_control import FixedNavigator
from mission_control import MissionRobot, MissionVisualApproach

ROOT=Path(__file__).resolve().parents[1]


class TravelSpeedTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'mission_straight_config.json').read_text(encoding='utf-8'))
        # Preserve regression coverage for the preceding, opt-in legacy schedule.
        self.config.update(maximum_wheel_speed_rad_s=6., travel_speed_profile='legacy', cruise_speed_m_s=.44,
            maximum_travel_command_m_s=.6, travel_slowdown_distance_m=1.5,
            travel_full_cruise_distance_m=2.2, near_handoff_speed_m_s=.22,
            near_handoff_command_limit_m_s=.45)
        self.position=np.array([-13.6,-6.5,.205])

    def navigator(self,config):
        config=copy.deepcopy(config)
        config['trial_floor_bounds_m']=config['trial_zones']['A']
        nav=FixedNavigator(config)
        nav.observe_fixed([{'position_m':[-10.6,-6.5,.0335]}],self.position,1.)
        nav.aligning=False;nav.forward_started=0.;nav.forward_speed=nav.cruise_speed
        return nav

    def test_faster_travel_is_configured_without_changing_legacy_handoff(self):
        fast=self.navigator(self.config)
        legacy=dict(self.config)
        legacy.pop('cruise_speed_m_s');legacy.pop('maximum_travel_command_m_s')
        slow=self.navigator(legacy)
        fast_wheels,_,_=fast.command(self.position,np.eye(3),1.)
        slow_wheels,_,_=slow.command(self.position,np.eye(3),1.)
        self.assertAlmostEqual(float(fast_wheels.mean())*.11,.44)
        self.assertAlmostEqual(float(slow_wheels.mean())*.11,.22)
        self.assertLessEqual(float(np.max(np.abs(fast_wheels))),6.)

    def test_fast_travel_still_waits_for_initial_alignment(self):
        nav=self.navigator(self.config)
        nav.target[1]+=.8
        nav.aligning=True
        wheels,_,status=nav.command(self.position,np.eye(3),1.)
        self.assertAlmostEqual(float(wheels.mean()),0.)
        self.assertEqual(status,'fixed_aligning')

    def test_near_handoff_restores_legacy_speed_and_command_cap(self):
        for measured,expected in ((0.,.45),(.22,.22),(.4,0.)):
            nav=self.navigator(self.config)
            nav.target=self.position+np.array([1.3,0,-.1715])
            nav.forward_speed=measured
            wheels,ready,status=nav.command(self.position,np.eye(3),1.)
            self.assertAlmostEqual(float(wheels.mean())*.11,expected)
            self.assertFalse(ready)
            self.assertEqual(status,'fixed_approaching')

    def test_cruise_slows_continuously_with_rgb_estimated_distance(self):
        for distance,desired in ((1.49,.22),(1.85,.33),(2.21,.44)):
            nav=self.navigator(self.config)
            nav.target=self.position+np.array([distance,0,-.1715])
            nav.forward_speed=desired
            wheels,_,_=nav.command(self.position,np.eye(3),1.)
            self.assertAlmostEqual(float(wheels.mean())*.11,desired)

    def test_slowdown_configuration_must_leave_distance_before_handoff(self):
        for key,value in (('travel_slowdown_distance_m',.9),('travel_full_cruise_distance_m',1.4),
                          ('near_handoff_speed_m_s',.7),('near_handoff_command_limit_m_s',.7)):
            config=copy.deepcopy(self.config);config[key]=value
            with self.assertRaises(ValueError):self.navigator(config)

    def test_fast_cruise_has_more_braking_margin_than_the_old_15cm_guard(self):
        robot=MissionRobot('A',self.config,[23.77,10.97])
        robot.stage='fixed';robot.filtered_forward_speed=.5
        wheels=robot.guard_motion(np.full(2,.6/.11),np.array([-10.5,-6.5,.205]),np.eye(3),1.)
        np.testing.assert_array_equal(wheels,[0.,0.])
        self.assertEqual(robot.stage,'fault')

    def test_pickup_retains_the_15cm_guard_with_both_pickup_speeds(self):
        for speed in (.3,.4):
            with self.subTest(pickup_speed=speed):
                config=copy.deepcopy(self.config);config['pickup_forward_speed_m_s']=speed
                robot=MissionRobot('A',config,[23.77,10.97])
                robot.stage='scoop';robot.filtered_forward_speed=speed
                wheels=np.full(2,6.)
                np.testing.assert_array_equal(robot.guard_motion(wheels,np.array([-10.5,-6.5,.205]),np.eye(3),1.),wheels)

    def visual(self,lateral,yaw_rate):
        visual=MissionVisualApproach([23.77,10.97])
        visual.observe([SimpleNamespace(position_in_robot_m=[.8,lateral,-.1715])],self.position,np.eye(3),1.)
        visual.aligning=False;visual.forward_start_time=0.;visual.yaw_rate=yaw_rate
        return visual

    def test_invalid_speed_and_braking_values_are_rejected(self):
        for key,value in (('cruise_speed_m_s',float('nan')),('maximum_travel_command_m_s',.8)):
            config=copy.deepcopy(self.config);config[key]=value
            with self.assertRaises(ValueError):self.navigator(config)
        for value in (0.,-1.,float('nan')):
            config=copy.deepcopy(self.config);config['travel_brake_deceleration_m_s2']=value
            with self.assertRaises(ValueError):MissionRobot('A',config,[23.77,10.97])


if __name__=='__main__':unittest.main()
