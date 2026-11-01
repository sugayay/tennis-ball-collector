"""Two-robot image-only dispatch and bounded pickup state machines.

Inputs: RGB-derived ball observations, robot proprioception, simulation clock.
No Genesis imports, ball entities, ground-truth targets, or test positions.
Pickup cycles are attempts; physical collection is measured by a separate evaluator.
"""
from __future__ import annotations

import math
import numpy as np

from handoff_control import FixedNavigator, observations_for_target, trial_position_allowed
from vision_control import VisualApproach, court_position_allowed
from basket_confirmation import BasketConfirmation
from rolling_tracker import RollingBallTracker
from recollection import validate_recovery,select_recollection
from collection_service import HomeReturn,validate_service


def blend(a, b, fraction):
    t = float(np.clip(fraction, 0, 1))
    return a+(b-a)*(3*t*t-2*t*t*t)


PICKUP = -.448
CAPTURE, CARRY = math.radians(5), math.radians(15)
RAISED, TIPPED = math.radians(125), math.radians(-50)
# name, duration, lift start/end, absolute bucket start/end, wheel speed, brake
CYCLE = (
    ('scoop', .30, PICKUP, -.20, 0, CAPTURE, 6., False),
    ('secure', 1., -.20, -.20, CAPTURE, CARRY, 6., True),
    ('lift', 3., -.20, RAISED, CARRY, CARRY, 0., False),
    ('tip', 1.8, RAISED, RAISED, CARRY, TIPPED, 0., False),
    ('level', 1., RAISED, RAISED, TIPPED, CARRY, 0., False),
    ('lower', 3., RAISED, PICKUP, CARRY, 0., 0., False),
    ('retain', 3., PICKUP, PICKUP, 0., 0., 0., False),
)


class MissionVisualApproach(VisualApproach):
    """Damp steering continuously after the initial alignment.

    Repeated stop-and-turn corrections can roll a ball away. Steer while
    advancing within the shovel's capture width; stop for a larger lateral
    error. Keep the original pickup, freshness, and court guards.
    The original one-robot and handoff demonstrations remain unchanged.
    """
    def command(self, position, rotation, seconds):
        wheels, ready, status = super().command(position, rotation, seconds)
        if self.world_target is not None and status in ('approaching', 'at_shovel'):
            local = rotation.T@(self.world_target-position)
            # Stock shovel front is 0.551 m from the chassis origin.
            # The 0.60 m trigger leaves 49 mm ahead of it at rest; actual
            # contact depends on lip height and pose. Start before contact,
            # then use the configured measured insertion. The 70 mm corridor fits the 240 mm shovel
            # with a ball-radius and 16 mm lateral margin on each side.
            # Predict near-term yaw drift rather than demanding the exact
            # centre after the ball has already reached the shovel.
            lateral_envelope = abs(local[1])+.60*abs(self.yaw_rate)*.15
            if (not self.aligning and .35 < local[0] <= .60
                    and lateral_envelope <= .070 and self.detection_updates >= 2):
                return np.zeros(2), True, 'at_shovel'
        if ready or status != 'approaching' or self.world_target is None:
            return wheels, ready, status
        local = rotation.T@(self.world_target-position)
        if local[0] < .85:
            bearing = math.atan2(local[1], local[0])
            angular = float(np.clip(6*bearing-1.2*self.yaw_rate, -.65, .65))
            if abs(local[1]) <= .070:
                forward = float(wheels.mean())*.11
                return np.clip(np.array([forward-angular*.45/2,forward+angular*.45/2])/.11,-6,6),False,'approaching'
            self.aligning = True
            self.forward_start_time = None
            return np.array([-angular, angular])*.45/(2*.11), False, 'realigning_before_contact'
        return wheels, ready, status


class MissionRobot:
    def __init__(self, robot_id, config, playing_area):
        self.id, self.config, self.playing_area = robot_id, dict(config), playing_area
        distance = self.config.get('scoop_distance_m')
        if distance is not None and not 0 < distance <= .25:
            raise ValueError('scoop_distance_m must be greater than 0 and at most 0.25 m.')
        insertion = self.config.get('scoop_preinsert_m', 0.)
        if not 0 <= insertion < (distance or .25) or (insertion > 0 and distance is None):
            raise ValueError('scoop_preinsert_m requires distance feedback and must be below scoop_distance_m.')
        pickup_speed = self.config.get('pickup_forward_speed_m_s')
        if pickup_speed is not None and not 0 < pickup_speed <= .45:
            raise ValueError('pickup_forward_speed_m_s must be greater than 0 and at most 0.45 m/s.')
        reaction = self.config.get('travel_brake_reaction_seconds',.15)
        deceleration = self.config.get('travel_brake_deceleration_m_s2',.5)
        if not 0 <= reaction <= 1 or not 0 < deceleration <= 2:
            raise ValueError('Travel braking parameters must be finite and conservative.')
        if 'travel_shovel_lift_angle_rad' in self.config:
            angle = self.config['travel_shovel_lift_angle_rad']
            rate = self.config.get('travel_shovel_lift_rate_rad_s', .3)
            if not PICKUP <= angle <= -.20 or not .05 <= rate <= .5:
                raise ValueError('Invalid travel shovel pose or movement rate.')
        self.config['trial_floor_bounds_m'] = config['trial_zones'][robot_id]
        self.stage, self.stage_started = 'seek', 0.
        self.target, self.navigator, self.visual = None, None, None
        self.observations = []
        self.last_onboard_time = -math.inf
        self.last_fixed_time = -math.inf
        self.last_fixed_matches = []
        self.cycles, self.retries, self.retry_total = 0, 0, 0
        self.completed_targets, self.events = [], []
        self.joints = np.array([PICKUP, -PICKUP])
        self.cycle_index = 0
        self.heading_reference, self.previous_heading = None, None
        self.heading_rate = 0.
        self.previous_position = None
        self.forward_speed = 0.
        self.filtered_forward_speed = 0.
        self.scoop_start_position = None
        self.scoop_lift_started = None
        self.last_command_time = None
        self.last_status = 'seeking_image_target'
        basket = config.get('basket_confirmation_config')
        self.basket_confirmation = BasketConfirmation(basket) if basket else None
        self.basket_confirmation_required = bool(config.get('basket_confirmation_required', False))
        if self.basket_confirmation_required and not self.basket_confirmation:
            raise ValueError('Required basket confirmation needs sensor configuration.')
        self.confirmed_cycles = 0
        self.confirmed_balls = 0
        self.capacity,self.full_behavior = validate_service(config)
        self.service_enabled = bool(config.get('collection_service_enabled',False))
        if self.service_enabled and not self.basket_confirmation_required:
            raise ValueError('Collection service requires basket RGB confirmation')
        if self.service_enabled and self.full_behavior=='return_home':
            home=self.config['home_positions_m'].get(robot_id)
            if home is None or not trial_position_allowed(home,self.config) or not court_position_allowed(home,playing_area,'practice'):
                raise ValueError('Home must remain inside the approved temporary trial zone')
        self.home_return = None
        self.service_visits = 0
        self.paused_since = None
        self.paused_joints = None
        self.recovery_config = config.get('recovery_config')
        self.recovery_enabled = bool(self.recovery_config and self.recovery_config.get('enabled'))
        if self.recovery_enabled:
            validate_recovery(self.recovery_config)
            if not self.basket_confirmation_required:raise ValueError('Recollection requires basket confirmation')
        self.recollection_attempts = self.recollection_total = 0
        self.recovery_origin = self.recovery_baseline = None
        self.backoff_origin = self.backoff_heading = None
        self.recovery_track_id = None
        self.recovery_selection_reason = None
        self.test_empty_scoop_attempts = config.get('test_empty_scoop_attempts',0)

    def transition(self, stage, seconds, reason=None):
        if self.basket_confirmation is not None:
            if stage == 'scoop':
                self.basket_confirmation.begin_attempt(self.cycles+1, seconds)
            elif stage == 'retain':
                self.basket_confirmation.begin_confirmation(seconds)
        self.events.append({'seconds':seconds, 'robot':self.id, 'from':self.stage,
                            'to':stage, 'reason':reason, 'cycle':self.cycles if stage in ('confirm','recover_backoff','recover_wait','done','fault') else self.cycles+1,
                            'image_target_m':None if self.target is None else self.target.tolist()})
        self.stage, self.stage_started = stage, seconds

    def observe_basket(self, observed_count, seconds, valid):
        if self.basket_confirmation is not None:
            self.basket_confirmation.observe(observed_count, seconds, valid)

    def complete_target(self, seconds, reason):
        self.completed_targets.append(self.target.copy())
        self.target = self.navigator = self.visual = None
        completed = self.confirmed_cycles if self.basket_confirmation_required else self.cycles
        self.recovery_origin = self.recovery_baseline = self.recovery_track_id = None
        if self.service_enabled and self.basket_confirmation.stable_count(seconds)==self.capacity:
            self.service_visits += 1
            if self.full_behavior=='return_home':
                self.home_return=HomeReturn(self.config['home_positions_m'][self.id])
                self.transition('return_home',seconds,'basket_rgb_capacity_reached')
            else:self.transition('await_unload',seconds,'basket_rgb_capacity_reached')
        else:self.transition('done' if completed >= self.config['cycles_per_robot'] else 'seek', seconds, reason)

    def assign(self, observation, position, seconds, observed_seconds=None):
        if self.stage != 'seek':
            raise RuntimeError('A busy robot cannot receive another target.')
        self.target = np.array(observation['position_m'], dtype=float)
        self.retries = 0
        self.recollection_attempts = 0
        self.navigator = FixedNavigator(self.config)
        captured=seconds if observed_seconds is None else observed_seconds
        self.navigator.observe_fixed([observation], position, captured)
        self.observe_fixed([observation], position, captured)
        self.transition('fixed', seconds, 'reserved_image_detection')

    def observe_fixed(self, observations, position, seconds):
        self.last_fixed_matches = [] if self.target is None else [o for o in observations
            if np.linalg.norm(np.array(o['position_m'][:2])-self.target[:2]) <= self.config['target_association_distance_m']]
        self.last_fixed_time = seconds
        if self.navigator is not None and self.stage in ('fixed', 'retry', 'seek'):
            self.navigator.observe_fixed(self.last_fixed_matches, position, seconds)
            if self.navigator.target is not None:
                self.target = self.navigator.target.copy()

    def observe_onboard(self, observations, position, rotation, seconds):
        self.observations, self.last_onboard_time = observations, seconds
        if self.navigator is not None:
            self.navigator.observe_onboard(observations, position, rotation)
        if self.stage == 'visual' and self.visual is not None:
            matches = observations_for_target(observations, position, rotation, self.target,
                                               self.config['target_association_distance_m'])
            self.visual.observe(matches, position, rotation, seconds)
            if matches and self.visual.last_seen == seconds and self.visual.world_target is not None:
                # Keep association anchored to the last RGB observation instead
                # of an old fixed-camera point when a ball rolls during approach.
                # Each update still must match the previous target within 25 cm.
                self.target = self.visual.world_target.copy()
                if self.navigator is not None:
                    self.navigator.target = self.target.copy()

    def fail_or_retry(self, seconds, reason):
        if self.retries >= self.config['maximum_retries_per_target']:
            self.transition('fault', seconds, 'retry_limit:'+reason)
        else:
            self.retries += 1
            self.retry_total += 1
            self.transition('retry', seconds, reason)

    def stop(self, seconds, reason):
        if self.stage != 'fault':
            self.transition('fault', seconds, reason)

    def begin_recollection(self, position, rotation, seconds, result):
        if self.recollection_attempts >= self.recovery_config['maximum_attempts_per_target']:
            self.stop(seconds,'recollection_retry_limit');return
        self.recollection_attempts += 1;self.recollection_total += 1
        self.recovery_origin = self.target.copy()
        self.recovery_baseline = result['baseline_observed_count']
        self.navigator = self.visual = None
        self.backoff_origin = position.copy()
        self.backoff_heading = math.atan2(rotation[1,0],rotation[0,0])
        endpoint = position-rotation[:,0]*self.recovery_config['backoff_distance_m']
        if not trial_position_allowed(endpoint,self.config) or not court_position_allowed(endpoint,self.playing_area,'practice'):
            self.stop(seconds,'recollection_backoff_boundary');return
        self.transition('recover_backoff',seconds,'no_rgb_increase_make_room_for_fixed_camera')

    def recovery_basket_valid(self, seconds):
        count=self.basket_confirmation.stable_count(seconds)
        return count is not None and count==self.recovery_baseline

    def accept_recollection(self, track, position, seconds, observed_seconds=None):
        if self.stage!='recover_wait':raise RuntimeError('Recovery robot must be waiting')
        if not self.recovery_basket_valid(seconds):
            self.stop(seconds,'recollection_basket_unknown_or_changed');return
        # Geometric floor assumption is the same one used by fixed detection.
        observation=dict(position_m=[*track['position_m'],float(self.recovery_origin[2])])
        self.target=np.array(observation['position_m'])
        self.recovery_track_id=track['track_id']
        self.navigator=FixedNavigator(self.config)
        self.navigator.observe_fixed([observation],position,seconds if observed_seconds is None else observed_seconds)
        self.retries=0
        self.transition('fixed',seconds,'recollection_fresh_stationary_dual_view_candidate')

    def guard_motion(self, wheels, position, rotation, seconds):
        # Use proprioception and command velocity only. Trial bounds are unchanged.
        commanded = float(np.mean(wheels))*.11
        measured = self.filtered_forward_speed
        direction = np.sign(measured) if abs(measured)>.03 else np.sign(commanded)
        distance = .15
        if self.stage=='fixed':
            speed = max(abs(measured),abs(commanded))
            reaction = self.config.get('travel_brake_reaction_seconds',.15)
            deceleration = self.config.get('travel_brake_deceleration_m_s2',.5)
            distance = max(distance,speed*reaction+speed*speed/(2*deceleration))
        forecast = position+rotation@np.array([direction*distance,0.,0.])
        if (not trial_position_allowed(forecast,self.config)
                or not court_position_allowed(forecast,self.playing_area,'practice')):
            if np.any(np.abs(wheels)>1e-8):
                self.stop(seconds,'controller_braking_boundary_guard')
            return np.zeros(2)
        return wheels

    def pause(self,seconds,actual_joints):
        if self.paused_since is None:
            self.paused_since=seconds;self.paused_joints=np.asarray(actual_joints).copy()

    def resume(self,seconds):
        if self.paused_since is None:return
        duration=seconds-self.paused_since
        self.stage_started+=duration
        if self.scoop_lift_started is not None:self.scoop_lift_started+=duration
        self.paused_since=None;self.paused_joints=None
        # Re-enter navigation through alignment and acceleration ramp.
        if self.navigator is not None:
            self.navigator.aligning=True;self.navigator.forward_started=None
            if self.navigator.speed_profile:self.navigator.speed_profile.reset()
        if self.visual is not None:
            self.visual.aligning=True;self.visual.forward_start_time=None

    def command(self, position, rotation, seconds, dt=1/120):
        """Return wheel velocities and shovel joint targets; never alter world state."""
        zero = np.zeros(2)
        elapsed = seconds-self.stage_started
        yaw = math.atan2(rotation[1,0], rotation[0,0])
        if self.previous_heading is not None and self.last_command_time is not None and seconds>self.last_command_time:
            delta = math.atan2(math.sin(yaw-self.previous_heading), math.cos(yaw-self.previous_heading))
            self.heading_rate = .8*self.heading_rate+.2*delta/(seconds-self.last_command_time)
            if self.previous_position is not None:
                velocity = (position-self.previous_position)/(seconds-self.last_command_time)
                self.forward_speed = float(velocity@rotation[:,0])
                self.filtered_forward_speed = .8*getattr(self,'filtered_forward_speed',self.forward_speed)+.2*self.forward_speed
        self.previous_heading, self.last_command_time = yaw, seconds
        self.previous_position = position.copy()
        if not trial_position_allowed(position, self.config) or not court_position_allowed(position, self.playing_area, 'practice'):
            self.stop(seconds, 'position_guard')
        if self.paused_since is not None:
            return zero,self.paused_joints.copy()
        if self.stage=='return_home':
            count=self.basket_confirmation.stable_count(seconds)
            if count is None:
                self.last_status='waiting_for_fresh_stable_basket_before_return'
                return zero,self.joints.copy()
            if count is not None and count!=self.capacity:
                self.stop(seconds,'basket_count_changed_during_loaded_return')
                return zero,self.joints.copy()
            wheels,arrived=self.home_return.command(position,rotation,self.filtered_forward_speed,seconds)
            self.last_status='returning_with_full_basket'
            if arrived:self.transition('await_unload',seconds,'home_reached_wait_for_manual_unload')
            return self.guard_motion(wheels,position,rotation,seconds),self.joints.copy()
        if self.stage=='await_unload':
            self.last_status='waiting_for_stable_rgb_empty_basket'
            if self.basket_confirmation.stable_count(seconds)==0:
                # A new batch can contain balls landing at previously visited
                # positions. Keep the cumulative counter, release old reservations.
                self.completed_targets=[]
                self.transition('done' if self.confirmed_cycles>=self.config['cycles_per_robot'] else 'seek',seconds,'stable_rgb_empty_after_service')
            return zero,self.joints.copy()
        if self.stage in ('seek', 'done', 'fault'):
            return zero, self.joints.copy()
        if self.stage == 'confirm':
            result = self.basket_confirmation.outcome(seconds)
            self.last_status = 'basket_rgb:'+result['status']
            if result['status'] == 'confirmed_increase':
                self.basket_confirmation.finish(seconds)
                self.confirmed_cycles += 1
                self.confirmed_balls += result['final_observed_count']-result['baseline_observed_count']
                self.complete_target(seconds, 'basket_rgb_count_increased')
            elif elapsed >= 2.0:
                self.basket_confirmation.finish(seconds)
                if self.recovery_enabled and result['status']=='no_increase_observed':
                    self.begin_recollection(position,rotation,seconds,result)
                else:self.stop(seconds, 'basket_rgb:'+result['status'])
            return zero, self.joints.copy()
        if self.stage in ('recover_backoff','recover_wait'):
            if not self.recovery_basket_valid(seconds):
                self.stop(seconds,'recollection_basket_unknown_or_changed');return zero,self.joints.copy()
            if self.stage=='recover_wait':
                # This stage already commands no motion. A late ball frame can
                # be awaited within the existing bounded recovery timeout.
                # Dispatch still needs fresh dual-view evidence; the independent
                # person-camera deadline and moving-backoff fault remain intact.
                stale=seconds-self.last_fixed_time>self.recovery_config['fixed_max_age_seconds']
                self.last_status='waiting_for_fresh_fixed_camera' if stale else self.recovery_selection_reason or 'waiting_for_stationary_local_ball'
                if elapsed>=self.recovery_config['wait_timeout_seconds']:
                    self.stop(seconds,'recollection_no_safe_candidate:'+self.last_status)
                return zero,self.joints.copy()
            if seconds-self.last_fixed_time>self.recovery_config['fixed_max_age_seconds']:
                self.stop(seconds,'recollection_fixed_camera_stale');return zero,self.joints.copy()
            axis=np.array([math.cos(self.backoff_heading),math.sin(self.backoff_heading),0.])
            distance=float((self.backoff_origin-position)@axis)
            if distance>=self.recovery_config['backoff_distance_m']:
                self.transition('recover_wait',seconds,'backoff_complete_wait_for_rest_and_fresh_rgb')
                return zero,self.joints.copy()
            if elapsed>=self.recovery_config['backoff_timeout_seconds']:
                self.stop(seconds,'recollection_backoff_timeout');return zero,self.joints.copy()
            desired=-self.recovery_config['backoff_speed_m_s']
            forward=float(np.clip(desired+1.5*(desired-self.filtered_forward_speed),-.18,0.))
            error=math.atan2(math.sin(self.backoff_heading-yaw),math.cos(self.backoff_heading-yaw))
            angular=float(np.clip(3*error-.9*self.heading_rate,-.35,.35))
            wheels=np.array([forward-angular*.45/2,forward+angular*.45/2])/.11
            return self.guard_motion(wheels,position,rotation,seconds),self.joints.copy()
        if self.stage == 'retry':
            if elapsed >= self.config['retry_wait_seconds']:
                fresh = seconds-self.last_fixed_time <= self.config['fixed_observation_timeout_seconds']
                if fresh and self.last_fixed_matches:
                    self.navigator = FixedNavigator(self.config)
                    self.navigator.observe_fixed(self.last_fixed_matches, position, self.last_fixed_time)
                    if self.navigator.target is not None:
                        if seconds-self.last_onboard_time <= .15:
                            self.navigator.observe_onboard(self.observations, position, rotation)
                        self.transition('fixed', seconds, 'fresh_detection_reacquired')
                if self.stage=='retry' and elapsed > self.config['lost_detection_retry_seconds']:
                    self.fail_or_retry(seconds, 'target_not_reacquired')
            return zero, self.joints.copy()
        if self.stage == 'fixed':
            wheels, ready, status = self.navigator.command(position, rotation, seconds)
            if 'travel_shovel_lift_angle_rad' in self.config:
                target_angle = self.config['travel_shovel_lift_angle_rad'] if self.navigator.fast_route or self.navigator.fast_braking else PICKUP
                rate = self.config.get('travel_shovel_lift_rate_rad_s', .3)
                self.joints[0] += float(np.clip(target_angle-self.joints[0],-rate*dt,rate*dt))
                self.joints[1] = -self.joints[0]
            self.last_status = status
            if ready:
                self.transition('brake', seconds, 'two_matching_onboard_frames')
                return zero, self.joints.copy()
            if status == 'trial_boundary_blocked':
                self.stop(seconds, status)
                return zero, self.joints.copy()
            if elapsed > self.config['maximum_fixed_navigation_seconds']:
                self.fail_or_retry(seconds, 'fixed_navigation_timeout')
                return zero, self.joints.copy()
            if seconds-self.navigator.last_seen > self.config['lost_detection_retry_seconds']:
                self.fail_or_retry(seconds, 'fixed_target_lost')
                return zero, self.joints.copy()
            return self.guard_motion(wheels,position,rotation,seconds), self.joints.copy()
        if self.stage == 'brake':
            if elapsed >= .5:
                self.visual = MissionVisualApproach(self.playing_area, 'practice')
                if seconds-self.last_onboard_time <= .15:
                    self.observe_onboard(self.observations, position, rotation, self.last_onboard_time)
                self.transition('visual', seconds)
            return zero, self.joints.copy()
        if self.stage == 'visual':
            wheels, ready, status = self.visual.command(position, rotation, seconds)
            self.last_status = status
            if ready:
                if self.recovery_origin is not None and not self.recovery_basket_valid(seconds):
                    self.stop(seconds,'recollection_basket_changed_before_pickup');return zero,self.joints.copy()
                self.cycle_index = 0
                self.heading_reference, self.heading_rate = yaw, self.visual.yaw_rate
                self.scoop_start_position = position.copy()
                self.scoop_lift_started = None
                self.transition('scoop', seconds, 'ball_at_shovel_from_onboard_rgb')
                # Match the proven single-robot demo: immediately start the
                # scoop. A one-step zero-wheel pulse brakes the chassis just
                # as the timed lift begins and changes the contact timing.
                elapsed = 0.
            else:
                if status == 'court_entry_blocked':
                    self.stop(seconds, status)
                    return zero, self.joints.copy()
                last_seen = max(self.visual.last_seen, self.stage_started)
                if seconds-last_seen > self.config['lost_detection_retry_seconds'] or elapsed > self.config['maximum_visual_approach_seconds']:
                    self.fail_or_retry(seconds, 'onboard_target_lost_or_timeout')
                    return zero, self.joints.copy()
                return self.guard_motion(wheels,position,rotation,seconds), self.joints.copy()
        name, duration, a0, a1, b0, b1, speed, brake = CYCLE[self.cycle_index]
        fraction = (elapsed+dt)/duration
        distance_goal = self.config.get('scoop_distance_m')
        test_empty = self.cycles < self.test_empty_scoop_attempts
        if test_empty:distance_goal=None
        if name=='scoop' and distance_goal is not None:
            # Wheel slip and caster direction change actual travel. Synchronize
            # arm progress with proprioception, never with simulator ball state.
            direction = np.array([math.cos(self.heading_reference),math.sin(self.heading_reference),0.])
            travelled = float((position-self.scoop_start_position)@direction)
            predicted = max(0.,travelled)+max(0.,self.forward_speed)*dt
            insertion = self.config.get('scoop_preinsert_m',0.)
            fraction = max(0.,predicted-insertion)/(distance_goal-insertion)
            lift_elapsed = elapsed
            if insertion > 0:
                # Seat the thin front edge below the ball before raising it.
                # Insertion and lift both use only robot proprioception.
                if predicted >= insertion and self.scoop_lift_started is None:
                    self.scoop_lift_started = seconds
                lift_elapsed = 0. if self.scoop_lift_started is None else seconds-self.scoop_lift_started
            minimum_seconds = self.config.get('minimum_scoop_seconds',0.)
            if minimum_seconds > 0:
                fraction = min(fraction,(lift_elapsed+dt)/minimum_seconds)
            if elapsed > self.config.get('maximum_scoop_seconds',1.5):
                self.stop(seconds,'scoop_progress_timeout')
                return zero,self.joints.copy()
        arm, bucket = blend(a0, a1, fraction), blend(b0, b1, fraction)
        self.joints = np.array([arm, bucket-arm])
        speed *= max(0., 1-fraction) if brake else 1.
        pickup_speed = self.config.get('pickup_forward_speed_m_s')
        if name in ('scoop','secure') and pickup_speed is not None:
            desired = pickup_speed*(max(0.,1-fraction) if brake else 1.)
            measured = getattr(self,'filtered_forward_speed',self.forward_speed)
            speed = float(np.clip((desired+2*(desired-measured))/.11,0.,6.))
        error = math.atan2(math.sin(self.heading_reference-yaw), math.cos(self.heading_reference-yaw))
        angular = float(np.clip(6*error-1.2*self.heading_rate, -1.2, 1.2)) if speed else 0.
        correction = angular*.45/(2*.11)
        wheels = np.array([speed-correction, speed+correction])
        # Explicit test-only actuator omission: drive and shovel remain at rest
        # for the complete attempt. No ball pose/velocity is read or changed.
        if test_empty:
            wheels=zero
            self.joints=np.array([PICKUP,-PICKUP])
        if fraction >= 1.-1e-8:
            self.cycle_index += 1
            if self.cycle_index == len(CYCLE):
                self.cycles += 1
                if self.basket_confirmation_required:
                    self.transition('confirm', seconds+dt, 'mechanism_complete_wait_for_basket_rgb')
                else:
                    if self.basket_confirmation is not None:
                        self.basket_confirmation.finish(seconds+dt)
                    self.complete_target(seconds+dt, 'mechanism_cycle_completed_not_a_pickup_sensor')
            else:
                self.transition(CYCLE[self.cycle_index][0], seconds+dt)
        return self.guard_motion(wheels,position,rotation,seconds), self.joints.copy()


class FleetCoordinator:
    def __init__(self, config, playing_area):
        self.config = config
        self.robots = {rid:MissionRobot(rid, config, playing_area) for rid in config['trial_zones']}
        self.assignments = []
        self.recoveries = []
        self.tracking_frames = []
        self.tracker = RollingBallTracker(floor_size_m=config['perception_floor_size_m']) if config.get('recovery_config',{}).get('enabled') else None
        self.policy=dict(playing_area_m=playing_area,court_size_m=config.get('perception_floor_size_m'),trial_zones=config['trial_zones'])

    def observe_fixed(self, observations, positions, seconds, received_seconds=None, current_positions=None):
        now=seconds if received_seconds is None else received_seconds
        current_positions=positions if current_positions is None else current_positions
        tracked=self.tracker.update(observations,positions,seconds) if self.tracker else None
        if tracked is not None:self.tracking_frames.append(tracked)
        # Basket/arm balls must not be projected onto the floor as a new task.
        candidates = [o for o in observations if all(np.linalg.norm(
            np.array(o['position_m'][:2])-np.array(p[:2])) > self.config['robot_detection_exclusion_radius_m']
            for p in positions.values())]
        for rid, robot in self.robots.items():
            # Keep the locked target visible until the fixed->onboard transition.
            robot.observe_fixed(observations, positions[rid], seconds)
            if robot.stage=='recover_wait' and tracked is not None:
                if now-robot.stage_started < robot.recovery_config['settle_seconds'] or abs(robot.filtered_forward_speed)>.04:
                    continue
                if now-seconds>robot.recovery_config['fixed_max_age_seconds']:continue
                excluded=[r.target for key,r in self.robots.items() if key!=rid and r.target is not None]
                excluded += [p for r in self.robots.values() for p in r.completed_targets]
                chosen,reason=select_recollection(tracked['tracks'],robot.recovery_origin,rid,self.policy,
                    robot.recovery_config['search_radius_m'],excluded)
                robot.recovery_selection_reason=reason
                if chosen:
                    robot.accept_recollection(chosen,current_positions[rid],now,observed_seconds=seconds)
                    self.recoveries.append(dict(seconds=now,captured_seconds=seconds,robot=rid,attempt=robot.cycles+1,
                        candidate=chosen,reason=reason,accepted=robot.stage=='fixed'))
                continue
            if robot.stage != 'seek' or robot.paused_since is not None:
                continue
            reserved = [r.target for r in self.robots.values() if r.target is not None]
            completed = [target for r in self.robots.values() for target in r.completed_targets]
            available = [o for o in candidates
                if trial_position_allowed(o['position_m'], robot.config, clearance=0)
                and court_position_allowed(o['position_m'], robot.playing_area, 'practice', clearance=0)
                and all(np.linalg.norm(np.array(o['position_m'][:2])-p[:2]) > self.config['completed_target_exclusion_radius_m']
                        for p in reserved+completed)]
            if available:
                chosen = min(available, key=lambda o:(np.linalg.norm(np.array(o['position_m'][:2])-positions[rid][:2]), *o['position_m'][:2]))
                robot.assign(chosen, current_positions[rid], now, observed_seconds=seconds)
                self.assignments.append({'seconds':now, 'captured_seconds':seconds, 'robot':rid, 'cycle':robot.cycles+1,
                                         'image_target_m':robot.target.tolist()})

    def guard_separation(self, positions, seconds):
        names = list(self.robots)
        for index, first in enumerate(names):
            for second in names[index+1:]:
                if np.linalg.norm(np.array(positions[first][:2])-positions[second][:2]) < self.config['minimum_robot_separation_m']:
                    for robot in self.robots.values():
                        robot.stop(seconds, 'robot_separation_guard')

    @property
    def finished(self):
        terminal=('done','fault','await_unload') if self.config.get('end_trial_when_full',False) else ('done','fault')
        return all(r.stage in terminal for r in self.robots.values())
