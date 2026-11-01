"""Bounded selection of a fresh RGB floor candidate after unconfirmed pickup.

Association after occlusion is explicitly a new local candidate, not a claim of
physical identity. No nearest-neighbour tie breaking or simulator truth input.
"""
import copy
import math
from rolling_tracker import inspect_candidate


def corroborate_fixed_with_onboard(fixed, onboard, maximum_spread_m=.12):
    """Attach same-instant independent RGB provenance without moving a target.

    The caller captures both views before physics advances. Only a unique
    one-to-one match is retained; ego-only detections never create floor tasks.
    """
    result=copy.deepcopy(fixed)
    for camera_id, observations in onboard.items():
        edges=[[i for i,f in enumerate(fixed)
                if math.dist(o['position_m'][:2],f['position_m'][:2])<=maximum_spread_m]
               for o in observations]
        for o,matches in zip(observations,edges):
            if len(matches)!=1:continue
            index=matches[0]
            if sum(index in e for e in edges)!=1:continue
            item={**copy.deepcopy(o),'camera_id':camera_id}
            result[index]['observations'].append(item)
            result[index]['sources']=sorted(set(result[index]['sources']+[camera_id]))
    return result


def validate_recovery(config):
    limits={'search_radius_m':(0.,1.),'wait_timeout_seconds':(1.,30.),
        'backoff_distance_m':(0.,.5),'backoff_speed_m_s':(0.,.15),
        'backoff_timeout_seconds':(1.,10.),'settle_seconds':(0.,2.),
        'fixed_max_age_seconds':(0.,1.)}
    for name,(lower,upper) in limits.items():
        value=config[name]
        if not math.isfinite(value) or not lower<value<=upper:raise ValueError(name)
    count=config['maximum_attempts_per_target']
    if not isinstance(count,int) or not 1<=count<=2:raise ValueError('recovery retry bound')


def select_recollection(tracks, origin, robot_id, policy, radius, excluded):
    """Two plausible local balls block selection even if only one is stationary."""
    local=[t for t in tracks if t['fresh_observation']
           and math.dist(t['position_m'],origin[:2])<=radius
           and all(math.dist(t['position_m'],p[:2])>.30 for p in excluded)]
    if len(local)!=1:
        return None,'ambiguous_local_candidates' if local else 'no_local_candidate'
    candidate=local[0]
    result=inspect_candidate(candidate,policy)
    if not result['candidate_eligible'] or robot_id not in result['eligible_robots']:
        return None,','.join(result['withheld_reasons']) or 'not_in_this_robot_zone'
    return candidate,'new_local_stationary_rgb_candidate_identity_not_proven'
