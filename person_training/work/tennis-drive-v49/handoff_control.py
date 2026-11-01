"""Fixed-camera navigation and onboard target locking, with no Genesis access."""
from __future__ import annotations

import math

import numpy as np

from vision_control import VisualApproach
from drive_speed import DriveSpeedProfile


def trial_position_allowed(position,config,*,clearance=None):
    margin = config['trial_clearance_m'] if clearance is None else clearance
    xmin,ymin,xmax,ymax = config['trial_floor_bounds_m']
    x,y = position[:2]
    return xmin+margin <= x <= xmax-margin and ymin+margin <= y <= ymax-margin


def observations_for_target(observations,position,rotation,target,gate):
    if target is None:
        return []
    return [item for item in observations
            if np.linalg.norm((position+rotation@np.array(item.position_in_robot_m)-target)[:2]) <= gate]


class FixedNavigator:
    """Drive one robot toward an image-derived target inside a temporary trial.

    No ball objects, simulator ball coordinates, or test-ball initial position
    enter this class. A stale fixed-camera frame cannot keep the robot moving.
    """
    def __init__(self,config):
        self.config = config
        profile = config.get('travel_speed_profile', 'legacy')
        if profile not in ('legacy', 'braking_distance'):
            raise ValueError('Unknown travel speed profile.')
        self.speed_profile = DriveSpeedProfile(config) if profile == 'braking_distance' else None
        self.fast_route = None
        self.long_route = False
        self.fast_braking = False
        self.brake_stable_since = None
        self.lowering_until = None
        self.last_travel_command = 0.
        self.brake_desired = 0.
        self.brake_previous_time = None
        self.route_origin = None
        self.line_tracking = config.get('fast_line_tracking', False)
        self.lookahead = config.get('fast_lookahead_m', .7)
        if not .4 <= self.lookahead <= 1.5:
            raise ValueError('Invalid fast transit lookahead.')
        self.fast_minimum_distance = config.get('fast_travel_minimum_distance_m', 2.2)
        self.fast_end_distance = config.get('fast_travel_end_distance_m', 1.5)
        if self.speed_profile and (not all(math.isfinite(v) for v in (self.fast_end_distance,self.fast_minimum_distance))
                or not config['handoff_distance_m'] < self.fast_end_distance < self.fast_minimum_distance):
            raise ValueError('Fast travel must end before the legacy approach.')
        self.wheel_limit = config.get('maximum_wheel_speed_rad_s', 6.)
        self.cruise_speed = config.get('cruise_speed_m_s', .22)
        self.command_speed_limit = config.get('maximum_travel_command_m_s', .45)
        if not 0 < self.cruise_speed <= self.command_speed_limit <= .11*self.wheel_limit <= 1.54:
            raise ValueError('Travel speeds must respect the configured wheel speed and 14 rad/s prototype URDF limit.')
        self.slowdown_distance = config.get('travel_slowdown_distance_m', 0.)
        self.full_cruise_distance = config.get('travel_full_cruise_distance_m', 0.)
        self.near_speed = config.get('near_handoff_speed_m_s', self.cruise_speed)
        self.near_command_limit = config.get('near_handoff_command_limit_m_s', self.command_speed_limit)
        if self.slowdown_distance and not (config['handoff_distance_m'] < self.slowdown_distance < self.full_cruise_distance
                and 0 < self.near_speed <= self.cruise_speed and self.near_speed <= self.near_command_limit <= self.command_speed_limit):
            raise ValueError('Cruise slowdown distances and near-handoff speeds are invalid.')
        self.target = None
        self.last_seen = -math.inf
        self.aligning = True
        self.previous_position = None
        self.previous_yaw = None
        self.previous_time = None
        self.forward_speed = 0.0
        self.yaw_rate = 0.0
        self.forward_started = None
        self.onboard_matches = 0

    def observe_fixed(self,observations,position,seconds):
        candidates = [item for item in observations
                      if trial_position_allowed(item['position_m'],self.config,clearance=0)]
        if self.target is not None:
            candidates = [item for item in candidates if np.linalg.norm(
                np.array(item['position_m'][:2])-self.target[:2]) <= self.config['target_association_distance_m']]
        if not candidates:
            return
        item = min(candidates,key=lambda value:np.linalg.norm(np.array(value['position_m'][:2])-position[:2]))
        measured = np.array(item['position_m'],dtype=float)
        self.target = measured if self.target is None else .4*self.target+.6*measured
        self.last_seen = seconds

    def observe_onboard(self,observations,position,rotation):
        matches = observations_for_target(observations,position,rotation,self.target,
                                          self.config['target_association_distance_m'])
        self.onboard_matches = self.onboard_matches+1 if matches else 0
        return matches

    def command(self,position,rotation,seconds):
        yaw = math.atan2(rotation[1,0],rotation[0,0])
        if self.previous_time is not None and seconds > self.previous_time:
            dt = seconds-self.previous_time
            delta = math.atan2(math.sin(yaw-self.previous_yaw),math.cos(yaw-self.previous_yaw))
            self.yaw_rate = .8*self.yaw_rate+.2*delta/dt
            measured = float(((position-self.previous_position)/dt)@rotation[:,0])
            self.forward_speed = .8*self.forward_speed+.2*measured
        self.previous_position,self.previous_yaw,self.previous_time = position.copy(),yaw,seconds
        if self.target is None or seconds-self.last_seen > self.config['fixed_observation_timeout_seconds']:
            if self.speed_profile:
                self.speed_profile.reset()
            return np.zeros(2),False,'waiting_for_fresh_fixed_detection'
        if not trial_position_allowed(position,self.config):
            return np.zeros(2),False,'trial_boundary_blocked'
        local = rotation.T@(self.target-position)
        distance = float(np.linalg.norm(local[:2]))
        bearing = math.atan2(local[1],local[0])
        if self.speed_profile:
            if self.fast_route is None:
                self.fast_route = distance > self.fast_minimum_distance
                self.long_route = self.fast_route
                self.route_origin = position.copy()
            elif self.fast_route and distance <= self.fast_end_distance:
                self.fast_route = False
                self.fast_braking = True
                self.brake_desired = self.last_travel_command
                self.brake_previous_time = seconds
                self.speed_profile.reset()
        if self.fast_braking:
            axis=self.target-(position if self.route_origin is None else self.route_origin)
            axis[2]=0.
            axis/=max(1e-9,float(np.linalg.norm(axis)))
            origin=position if self.route_origin is None else self.route_origin
            progress=float((position-origin)@axis)
            point=origin+axis*(progress+self.lookahead)
            aim=rotation.T@(point-position)
            angle=math.atan2(aim[1],aim[0])
            desired_yaw_rate=2*max(.2,self.forward_speed)*math.sin(angle)/self.lookahead
            angular=float(np.clip(desired_yaw_rate+.4*(desired_yaw_rate-self.yaw_rate),-.65,.65))
            dt=max(0.,min(.05,seconds-self.brake_previous_time))
            self.brake_previous_time=seconds
            self.brake_desired=max(0.,self.brake_desired-self.config['travel_brake_deceleration_m_s2']*dt)
            forward=float(np.clip(self.brake_desired+.5*(self.brake_desired-self.forward_speed),0.,self.command_speed_limit))
            braking_wheels=np.clip(np.array([forward-angular*.45/2,forward+angular*.45/2])/.11,-self.wheel_limit,self.wheel_limit)
            if abs(self.forward_speed)<.04 and abs(self.yaw_rate)<.08:
                if self.brake_stable_since is None:
                    self.brake_stable_since=seconds
                if seconds-self.brake_stable_since>=.25:
                    self.fast_braking=False
                    self.long_route=False
                    self.aligning=True
                    self.lowering_until=seconds+.65
            else:
                self.brake_stable_since=None
            return (braking_wheels if self.fast_braking else np.zeros(2)),False,'transit_braking_at_rest_before_lowering'
        if self.lowering_until is not None and seconds<self.lowering_until:
            return np.zeros(2),False,'lowering_at_rest_before_approach'
        if distance <= self.config['handoff_distance_m']:
            if self.speed_profile:
                self.speed_profile.reset()
            ready = self.onboard_matches >= 2 and abs(bearing) < .15
            if abs(bearing) >= .15:
                # A close target may start outside the onboard field of view.
                # Rotate in place before waiting for two matching RGB frames.
                angular = float(np.clip(2.4*bearing-.9*self.yaw_rate,-.65,.65))
                return np.array([-angular,angular])*.45/(2*.11),False,'handoff_aligning'
            return np.zeros(2),ready,'handoff_ready' if ready else 'waiting_for_onboard_match'
        if self.aligning and abs(bearing) < .035 and abs(self.yaw_rate) < .08:
            self.aligning = False
            self.forward_started = seconds
        elif not self.aligning and abs(bearing) > (.45 if self.long_route and self.line_tracking else .18):
            self.aligning = True
        angular = float(np.clip(2.4*bearing-.9*self.yaw_rate,-.65,.65))
        if self.long_route and self.line_tracking and not self.aligning:
            axis = self.target-self.route_origin
            axis[2] = 0.
            length = float(np.linalg.norm(axis))
            if length > 1e-6:
                axis /= length
                progress = float((position-self.route_origin)@axis)
                point = self.route_origin+axis*min(length,max(0.,progress)+self.lookahead)
                aim = rotation.T@(point-position)
                angle = math.atan2(aim[1],aim[0])
                desired_yaw_rate = 2*max(.2,self.forward_speed)*math.sin(angle)/self.lookahead
                angular = float(np.clip(desired_yaw_rate+.4*(desired_yaw_rate-self.yaw_rate),-.65,.65))
        if self.aligning:
            forward = 0.0
            if self.speed_profile:
                self.speed_profile.reset()
        elif self.speed_profile and self.fast_route:
            forward = self.speed_profile.command(distance, self.forward_speed, seconds)
        elif self.speed_profile:
            # Preserve v7 exactly on short routes and throughout the final approach.
            desired = .22*min(1.,max(0.,(seconds-self.forward_started)/.5))
            forward = float(np.clip(desired+2*(desired-self.forward_speed),0,.45))
        else:
            fraction = float(np.clip((distance-self.slowdown_distance)/(self.full_cruise_distance-self.slowdown_distance),0,1)) if self.slowdown_distance else 1.
            cruise = self.near_speed+(self.cruise_speed-self.near_speed)*fraction
            limit = self.near_command_limit+(self.command_speed_limit-self.near_command_limit)*fraction
            desired = cruise*min(1.,max(0.,(seconds-self.forward_started)/.5))
            forward = float(np.clip(desired+2*(desired-self.forward_speed),0,limit))
        forecast = position+rotation@np.array([.15 if forward else 0,0,0])
        if not trial_position_allowed(forecast,self.config):
            return np.zeros(2),False,'trial_boundary_blocked'
        if self.long_route and self.line_tracking:
            # Reserve motor speed for steering instead of clipping just one wheel.
            forward = min(forward,max(0.,self.wheel_limit*.11-abs(angular)*.45/2))
        self.last_travel_command = forward
        wheels = np.array([forward-angular*.45/2,forward+angular*.45/2])/.11
        return np.clip(wheels,-self.wheel_limit,self.wheel_limit),False,'fixed_aligning' if self.aligning else 'fixed_approaching'


class HandoffVisualApproach(VisualApproach):
    """Re-align near the shovel if a long preceding turn leaves yaw drift.

    The original one-robot controller is preserved. Only this integration trial
    adds a stop-and-align gate before the ball reaches the shovel's front edge.
    """
    def command(self,position,rotation,seconds):
        wheels,ready,status = super().command(position,rotation,seconds)
        if ready or status != 'approaching' or self.world_target is None:
            return wheels,ready,status
        local = rotation.T@(self.world_target-position)
        if local[0] < .85 and (abs(local[1]) > .020 or abs(self.yaw_rate) > .15):
            self.aligning = True
            self.forward_start_time = None
            bearing = math.atan2(local[1],local[0])
            angular = float(np.clip(3*bearing-.8*self.yaw_rate,-.65,.65))
            return np.array([-angular,angular])*.45/(2*.11),False,'realigning_near_ball'
        return wheels,ready,status
