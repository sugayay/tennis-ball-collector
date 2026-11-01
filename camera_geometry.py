"""RGB pinhole calibration helpers; no Genesis entities or target-state access."""
from __future__ import annotations

import numpy as np


def world_to_pixel(point, intrinsic, transform):
    """Project a world point using OpenGL camera axes (right, up, back)."""
    local = np.linalg.solve(np.asarray(transform), np.r_[point, 1.0])[:3]
    depth = -local[2]
    if depth <= 0:
        return None
    return np.array([intrinsic[0, 0]*local[0]/depth+intrinsic[0, 2],
                     intrinsic[1, 2]-intrinsic[1, 1]*local[1]/depth])


def pixel_to_plane(pixel, intrinsic, transform, plane_z):
    """Estimate a detected ball center on the known, flat ball-center plane."""
    u, v = pixel
    ray = np.array([(u-intrinsic[0, 2])/intrinsic[0, 0],
                    -(v-intrinsic[1, 2])/intrinsic[1, 1], -1.0])
    origin = transform[:3, 3]
    direction = transform[:3, :3] @ ray
    if direction[2] >= -1e-8:
        return None
    distance = (plane_z-origin[2])/direction[2]
    if distance <= 0:
        return None
    return origin+distance*direction


def merge_ball_estimates(observations, distance_m):
    """Merge overlapping-view estimates, retaining their camera provenance.

    This has no robot assignment or driving policy. Empty observations remain
    empty: no simulator ground truth or fallback targets are consulted.
    """
    groups = []
    for item in sorted(observations, key=lambda value:value['confidence'], reverse=True):
        p = np.array(item['position_m'], dtype=float)
        match = next((group for group in groups
                      if np.linalg.norm(p[:2]-np.array(group['position_m'][:2])) <= distance_m), None)
        if match is None:
            groups.append({'position_m':p.tolist(), 'confidence':item['confidence'],
                           'sources':[item['camera_id']], 'observations':[item]})
        else:
            match['observations'].append(item)
            match['sources'] = sorted(set(match['sources']+[item['camera_id']]))
            weights = np.array([value['confidence'] for value in match['observations']])
            points = np.array([value['position_m'] for value in match['observations']])
            match['position_m'] = np.average(points,axis=0,weights=weights).tolist()
    return groups
