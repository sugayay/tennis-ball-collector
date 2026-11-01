"""Conservative continuity checks on synchronized RGB person observations.

Tracks never expire into free space. Only reciprocal, triangulated camera
pairs are collapsed; proximity alone cannot merge two people. Association is
geometric, not biometric: never-seen people and undetectable identity swaps
remain limitations. A lost/ambiguous track requires a latched safety stop.
"""
import math


class PersonContinuity:
    def __init__(self, speed_bound, position_margin):
        self.speed_bound = speed_bound
        self.position_margin = position_margin
        self.stamp = None
        self.tracks = []
        self.next_id = 0

    @staticmethod
    def observations(frames):
        entries = [(camera, p) for camera, frame in frames.items()
                   for p in frame['people']]
        used = set()
        result = []
        for index, (camera, person) in enumerate(entries):
            if index in used:
                continue
            used.add(index)
            peers = []
            if person.get('localization_method') == 'dual_view_head_triangulation':
                peers = [j for j, (other_camera, other) in enumerate(entries)
                         if j not in used and other_camera == person.get('paired_camera_id')
                         and other.get('paired_camera_id') == camera
                         and other.get('localization_method') == 'dual_view_head_triangulation'
                         and math.dist(person['position_m'], other['position_m']) < 1e-8]
            # A malformed or ambiguous pair remains separate observations.
            if len(peers) == 1:
                used.add(peers[0])
            result.append(list(person['position_m']))
        return result

    def update(self, frames):
        stamps = [frame['seconds'] for frame in frames.values()]
        stamp = min(stamps)
        if max(stamps) - stamp > 1e-9:
            return dict(reason='person_frames_unsynchronized', capture_seconds=stamps)
        if self.stamp is not None and stamp <= self.stamp:
            return None
        points = self.observations(frames)
        if self.stamp is not None:
            reach = self.speed_bound * (stamp - self.stamp) + 2 * self.position_margin
            candidates = [[j for j, point in enumerate(points)
                           if math.dist(track['position'][:2], point[:2]) <= reach + 1e-9]
                          for track in self.tracks]
            missing = [track for track, choices in zip(self.tracks, candidates) if not choices]
            if missing:
                return dict(reason='person_track_lost', capture_seconds=stamp,
                            last_seen_seconds=self.stamp,
                            lost_track_ids=[t['id'] for t in missing],
                            last_positions_m=[t['position'] for t in missing])
            if any(len(choices) != 1 for choices in candidates) or any(
                    sum(j in choices for choices in candidates) > 1 for j in range(len(points))):
                return dict(reason='person_identity_ambiguous', capture_seconds=stamp,
                            previous_track_count=len(self.tracks), observation_count=len(points))
            matched = {choices[0]: track['id'] for track, choices in zip(self.tracks, candidates)}
        else:
            matched = {}
        tracks = []
        for index, point in enumerate(points):
            track_id = matched.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            tracks.append(dict(id=track_id, position=point))
        self.tracks = tracks
        self.stamp = stamp
        return None
