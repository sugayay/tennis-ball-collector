"""Conservative RGB trajectory observer, deliberately disconnected from actuators.

Only image-derived ground-plane estimates and timestamped robot poses enter here.
An ID means a plausible image association, not a simulator identity. A missing ID
never means a collected ball. The floor assumption is not a height measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class TrackerConfig:
    maximum_gap_seconds: float = 1.1
    maximum_speed_m_s: float = 1.5
    innovation_margin_m: float = .18
    acceleration_bound_m_s2: float = 1.4
    robot_exclusion_m: float = .65
    minimum_confidence: float = .30
    confirmation_observations: int = 3
    confirmation_seconds: float = 1.
    stationary_speed_m_s: float = .08
    stationary_seconds: float = 1.
    dual_view_spread_m: float = .12

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.minimum_confidence > 1 or self.confirmation_observations < 2:
            raise ValueError('Invalid confidence or observation count')
        if not isinstance(self.confirmation_observations, int):
            raise ValueError('Observation count must be an integer')


@dataclass
class _Track:
    identity: int
    position: tuple
    velocity: tuple
    born: float
    last_seen: float
    observations: int = 1
    stable_since: float | None = None
    state: str = 'tentative'
    latest_sources: tuple = ()
    dual_view_agreement: bool = False
    last_reason: str = 'new_image_candidate'
    history: list = field(default_factory=list)


def _xy(value):
    if len(value) < 2:
        raise ValueError('Expected XY position')
    result = tuple(float(v) for v in value[:2])
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Position must be finite')
    return result


class RollingBallTracker:
    """One-to-one gated constant-velocity association with explicit uncertainty.

    Ambiguous graph components terminate every participating identity. We do not
    choose the nearest ball at a crossing, or resurrect an identity after a long
    occlusion. Unmatched detections start tentative IDs; three observations over
    one second are required for confirmation. This is a bounded development
    model, not a tracker for fast live tennis shots.
    """
    def __init__(self, config=None, *, floor_size_m=None):
        self.config = config or TrackerConfig()
        self.floor_size = _xy(floor_size_m) if floor_size_m is not None else None
        if self.floor_size is not None and any(v<=0 for v in self.floor_size):
            raise ValueError('Floor dimensions must be positive')
        self.tracks = {}
        self.retired = []
        self.last_time = None
        self.next_identity = 1

    def update(self, detections, robot_positions, seconds):
        now = float(seconds)
        if not math.isfinite(now) or now < 0 or (self.last_time is not None and now <= self.last_time):
            raise ValueError('Frame timestamps must be finite and strictly increasing')
        # Pose age/expected robot inventory is checked by the streaming adapter.
        robots = [_xy(p) for p in robot_positions.values()]
        if not robots:
            raise ValueError('Robot poses are required to mask arm/basket projections')
        self.last_time = now
        stats = dict(invalid=0, low_confidence=0, near_robot=0, outside_floor=0, ambiguous=0)
        obs = []
        for detection in detections:
            try:
                p = _xy(detection['position_m'])
                confidence = float(detection['confidence'])
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError('Invalid confidence')
            except (KeyError, ValueError, TypeError, OverflowError):
                stats['invalid'] += 1
                continue
            if confidence < self.config.minimum_confidence:
                stats['low_confidence'] += 1
                continue
            if self.floor_size and any(abs(v)>size/2 for v,size in zip(p,self.floor_size)):
                # Ray-plane projection is undefined as a ground target beyond
                # the known floor; do not invent continuation on another plane.
                stats['outside_floor'] += 1
                continue
            if any(math.dist(p, robot) <= self.config.robot_exclusion_m for robot in robots):
                stats['near_robot'] += 1
                continue
            obs.append((p, detection))
        # Sorting makes IDs independent of detector confidence/list order.
        obs.sort(key=lambda item: (item[0], -item[1]['confidence']))
        for identity, track in list(self.tracks.items()):
            if now-track.last_seen > self.config.maximum_gap_seconds+1e-9:
                self._retire(identity, now, 'expired_identity_unknown_collection')

        edges = {}
        incoming = {i: [] for i in range(len(obs))}
        for identity, track in self.tracks.items():
            dt = now-track.last_seen
            prediction = tuple(p+v*dt for p,v in zip(track.position, track.velocity))
            innovation = self.config.innovation_margin_m + .5*self.config.acceleration_bound_m_s2*dt*dt
            physical = self.config.maximum_speed_m_s*dt + self.config.innovation_margin_m
            edges[identity] = [i for i,(p,_) in enumerate(obs)
                               if math.dist(p, prediction) <= innovation
                               and math.dist(p, track.position) <= physical]
            for i in edges[identity]:
                incoming[i].append(identity)

        # Reject whole connected components containing competing assignments.
        bad_tracks = {identity for identity, matches in edges.items() if len(matches)>1}
        bad_obs = {i for i, matches in incoming.items() if len(matches)>1}
        while True:
            expanded_obs = bad_obs | {i for identity in bad_tracks for i in edges[identity]}
            expanded_tracks = bad_tracks | {identity for i in expanded_obs for identity in incoming[i]}
            if expanded_obs == bad_obs and expanded_tracks == bad_tracks:
                break
            bad_obs, bad_tracks = expanded_obs, expanded_tracks
        stats['ambiguous'] = len(bad_obs)
        for identity in sorted(bad_tracks):
            self._retire(identity, now, 'ambiguous_association_identity_withheld')

        used = set(bad_obs)
        for identity, track in self.tracks.items():
            matches = edges[identity]
            if not matches:
                track.state = 'unseen'
                track.stable_since = None
                track.last_reason = 'no_fresh_image_unknown_collection'
                continue
            i = matches[0]
            used.add(i)
            p, detection = obs[i]
            dt = now-track.last_seen
            velocity = tuple((a-b)/dt for a,b in zip(p, track.position))
            # Measured interval speed, not smoothed into apparent stationarity.
            speed = math.hypot(*velocity)
            track.stable_since = (track.stable_since if track.stable_since is not None else now) if speed <= self.config.stationary_speed_m_s else None
            track.position, track.velocity, track.last_seen = p, velocity, now
            track.observations += 1
            confirmed = (track.observations >= self.config.confirmation_observations
                         and now-track.born >= self.config.confirmation_seconds-1e-9)
            track.state = 'observed' if confirmed else 'tentative'
            track.last_reason = 'unique_motion_association'
            self._record(track, detection, now)

        for i,(p,detection) in enumerate(obs):
            if i in used:
                continue
            track = _Track(self.next_identity, p, (0.,0.), now, now)
            self.next_identity += 1
            self.tracks[track.identity] = track
            self._record(track, detection, now)
        return {'seconds': now, 'tracks': self.snapshot(now), 'filtered': stats}

    def _record(self, track, detection, now):
        # Require actual independent view provenance, not just a sources label.
        by_camera = {}
        for item in detection.get('observations', []):
            try:
                camera = str(item['camera_id'])
                p = _xy(item['position_m'])
            except (KeyError, TypeError, ValueError):
                continue
            if camera:
                by_camera.setdefault(camera, p)
        points = list(by_camera.values())
        track.latest_sources = tuple(sorted(by_camera))
        track.dual_view_agreement = (len(points)>=2 and
            max(math.dist(a,b) for a in points for b in points) <= self.config.dual_view_spread_m)
        track.history.append({'seconds':now, 'position_m':list(track.position),
                              'speed_m_s':math.hypot(*track.velocity),
                              'dual_view_agreement':track.dual_view_agreement})

    def _retire(self, identity, now, reason):
        track = self.tracks.pop(identity)
        self.retired.append({'track_id':identity, 'seconds':now, 'reason':reason,
                             'history':track.history})

    def snapshot(self, seconds):
        now = float(seconds)
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError('Cannot query a past/nonfinite tracker state')
        rows = []
        for identity, track in sorted(self.tracks.items()):
            fresh = abs(now-track.last_seen) < 1e-8
            stationary = (fresh and track.stable_since is not None and
                          now-track.stable_since >= self.config.stationary_seconds-1e-9)
            rows.append(dict(track_id=identity, position_m=list(track.position),
                velocity_m_s=list(track.velocity), speed_m_s=math.hypot(*track.velocity),
                first_seen_seconds=track.born, last_seen_seconds=track.last_seen,
                observations=track.observations, state=track.state if fresh else 'unseen',
                fresh_observation=fresh, stationary=stationary,
                camera_ids=list(track.latest_sources), dual_view_agreement=track.dual_view_agreement,
                collection_status='unknown', reason=track.last_reason))
        return rows


def inspect_candidate(track, policy):
    """Read-only candidate screen, NOT navigation/path/person-safety authorization.

    Even an eligible point needs fresh perception and the existing navigation and
    person-stop guards before any future integration can command wheel motion.
    """
    x,y = track['position_m']
    reasons = []
    if not track['fresh_observation']:
        reasons.append('no_fresh_image')
    if track['state'] != 'observed':
        reasons.append('identity_not_confirmed')
    if not track['stationary']:
        reasons.append('not_observed_stationary')
    if not track['dual_view_agreement']:
        reasons.append('ground_plane_not_cross_checked')
    lx, ly = policy['playing_area_m']
    if abs(x) < lx/2 and abs(y) < ly/2:
        reasons.append('practice_court_entry_forbidden')
    fx,fy = policy['court_size_m']
    if abs(x) > fx/2 or abs(y) > fy/2:
        reasons.append('outside_simulated_floor')
    eligible_robots = [rid for rid,(xmin,ymin,xmax,ymax) in policy['trial_zones'].items()
                       if xmin <= x <= xmax and ymin <= y <= ymax]
    if not eligible_robots:
        reasons.append('outside_trial_zones')
    return dict(candidate_eligible=not reasons, eligible_robots=eligible_robots if not reasons else [],
                withheld_reasons=reasons, actuator_command_authorized=False)
