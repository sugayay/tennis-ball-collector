"""Acceleration and braking limits based on own motion and route distance."""
from __future__ import annotations

import math


class DriveSpeedProfile:
    def __init__(self, config):
        self.cruise = config['cruise_speed_m_s']
        self.wheel_limit = config.get('maximum_wheel_speed_rad_s', 6.)
        self.command_limit = config['maximum_travel_command_m_s']
        self.near_speed = config['near_handoff_speed_m_s']
        self.near_distance = config['travel_slow_speed_distance_m']
        self.acceleration = config['travel_acceleration_m_s2']
        self.deceleration = config['travel_brake_deceleration_m_s2']
        self.reaction = config['travel_brake_reaction_seconds']
        self.feedback_gain = config.get('travel_speed_feedback_gain', 2.)
        values = (self.cruise, self.command_limit, self.near_speed, self.near_distance,
                  self.acceleration, self.deceleration, self.reaction, self.feedback_gain, self.wheel_limit)
        if (not all(math.isfinite(v) for v in values)
                or not 0 < self.near_speed <= self.cruise <= self.command_limit <= .11*self.wheel_limit <= 1.54
                or not config['handoff_distance_m'] < self.near_distance
                or not 0 < self.acceleration <= 1
                or not 0 < self.deceleration <= 1 or not 0 <= self.reaction <= 1
                or not 0 <= self.feedback_gain <= 2):
            raise ValueError('Invalid driving speed or braking envelope.')
        self.reset()

    def reset(self):
        self.previous_time = None
        self.desired = 0.

    def command(self, distance, measured_speed, seconds):
        if not all(math.isfinite(v) for v in (distance, measured_speed, seconds)):
            self.reset()
            return 0.
        # Distance permits reaction and deceleration to the existing handoff speed.
        remaining = max(0., distance-self.near_distance)
        ar = self.deceleration*self.reaction
        envelope = math.sqrt((self.near_speed+ar)**2 + 2*self.deceleration*remaining)-ar
        target = min(self.cruise, envelope)
        dt = 0. if self.previous_time is None else max(0., min(.05, seconds-self.previous_time))
        self.previous_time = seconds
        self.desired = min(target, self.desired+self.acceleration*dt)
        return max(0., min(self.command_limit, envelope, self.desired+self.feedback_gain*(self.desired-measured_speed)))
