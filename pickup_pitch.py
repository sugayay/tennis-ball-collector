"""Own-attitude flat-court pickup targets;pure math,no simulator/ball inputs."""
import math
import numpy as np


def flat_targets(current,rotation,dt):
    q=np.asarray(current,float);r=np.asarray(rotation,float)
    if q.shape!=(2,) or r.shape!=(3,3) or not np.isfinite(q).all() or not np.isfinite(r).all() or not 0<=dt<=.1:
        raise ValueError('Finite fresh own-pose feedback required')
    pitch=math.atan2(float(r[2,0]),math.hypot(float(r[0,0]),float(r[1,0])))
    compensation=float(np.clip(pitch,-.10,.10))
    arm=float(np.clip(-.448-compensation,-.48,-.20))
    tilt=float(np.clip(-compensation-arm,-3.10,.85))
    target=np.array([arm,tilt]);command=q+np.clip(target-q,-.30*dt,.30*dt)
    return command,{'own_pitch_rad':pitch,'pitch_compensation_rad':compensation,
                    'desired_flat_targets':target.tolist(),'command_targets':command.tolist(),
                    'arm_lower_stroke_saturated':bool(arm<=-.48),'ball_state_input':False}
