"""YOLO-guided approach and an explicit practice/break court-entry rule.

This module has no Genesis import and cannot read the simulator's ball state.
Inputs are camera detections, elapsed simulation time, and the robot's own pose.
"""
from __future__ import annotations

import math
import numpy as np


def court_position_allowed(position, playing_area, mode, clearance=0.63):
    if mode == 'break':
        return True
    if mode != 'practice':
        raise ValueError('Unknown operation mode; use practice or break.')
    return (abs(position[0]) >= playing_area[0]/2+clearance
            or abs(position[1]) >= playing_area[1]/2+clearance)


class VisualApproach:
    def __init__(self, playing_area, mode='practice'):
        self.playing_area = playing_area
        self.mode = mode
        self.world_target = None
        self.last_seen = -math.inf
        self.detection_updates = 0
        self.aligning = True
        self.previous_yaw = None
        self.previous_time = None
        self.yaw_rate = 0.0
        self.previous_position = None
        self.forward_speed = 0.0
        self.forward_start_time = None

    def observe(self, observations, robot_position, robot_rotation, seconds):
        for observation in observations:
            local = np.array(observation.position_in_robot_m)
            world = robot_position+robot_rotation@local
            if not court_position_allowed(world, self.playing_area, self.mode, clearance=0.0):
                continue
            if self.world_target is not None and np.linalg.norm(world[:2]-self.world_target[:2]) < 0.25:
                world = 0.4*self.world_target+0.6*world
            self.world_target = world
            self.last_seen = seconds
            self.detection_updates += 1
            return

    def command(self, robot_position, robot_rotation, seconds):
        yaw = math.atan2(robot_rotation[1,0],robot_rotation[0,0])
        if self.previous_time is not None and seconds > self.previous_time:
            delta = math.atan2(math.sin(yaw-self.previous_yaw),math.cos(yaw-self.previous_yaw))
            measured = delta/(seconds-self.previous_time)
            self.yaw_rate = 0.8*self.yaw_rate+0.2*measured
            velocity = (robot_position-self.previous_position)/(seconds-self.previous_time)
            self.forward_speed = 0.8*self.forward_speed+0.2*float(velocity@robot_rotation[:,0])
        self.previous_yaw, self.previous_time = yaw, seconds
        self.previous_position = robot_position.copy()
        if self.world_target is None or seconds-self.last_seen > 0.15:
            return np.zeros(2), False, 'waiting_for_ball'
        local = robot_rotation.T @ (self.world_target-robot_position)
        bearing = math.atan2(local[1], local[0])
        angular = float(np.clip(3.0*bearing-0.8*self.yaw_rate, -0.8, 0.8))
        if self.aligning and abs(bearing) < 0.010 and abs(self.yaw_rate) < 0.05:
            self.aligning = False
            self.forward_start_time = seconds
        elif not self.aligning and local[0] > 0.75 and abs(bearing) > 0.20:
            self.aligning = True
        if self.aligning:
            forward = 0.0
        else:
            forward_ramp = min(1.0,max(0.0,(seconds-self.forward_start_time)/0.4))
            desired = 0.308*forward_ramp
            forward = float(np.clip(desired+2.0*(desired-self.forward_speed),0,0.60))
        # Check a conservative chassis footprint plus 0.15 m braking look-ahead.
        projected = robot_position+robot_rotation@np.array([0.15 if forward else 0.0, 0, 0])
        if not court_position_allowed(projected, self.playing_area, self.mode):
            return np.zeros(2), False, 'court_entry_blocked'
        if not self.aligning and local[0] <= 0.564 and abs(local[1]) < 0.035 and self.detection_updates >= 2:
            return np.zeros(2), True, 'at_shovel'
        ramp = min(1.0, seconds/0.5)
        wheels = np.array([forward-angular*0.45/2, forward+angular*0.45/2])/0.11
        return np.clip(wheels*ramp, -6.0, 6.0), False, 'aligning' if self.aligning else 'approaching'
