"""Measured storage interlock for the lift-equipped collection chassis."""
import math
import numpy as np

JOINTS=('refill_lift_joint','refill_slide_joint','refill_tip_joint','refill_gate_joint')
TOLERANCE=(.02,.02,.04,.02)
VELOCITY_LIMIT=(.16,.14,.45,.14)  # Declared mechanical joint velocity limits.

def stored(q,v):
    return (len(q)==len(v)==4 and all(math.isfinite(float(x)) for x in (*q,*v))
            and all(abs(float(x))<=limit for x,limit in zip(q,TOLERANCE))
            and all(abs(float(x))<.06 for x in v))

class StorageInterlock:
    """At-rest arming; continuous position and bounded vibration in transit.

    A .1s velocity burst at the strictest .14m/s slide limit is bounded by
    14mm; the independent20mm position limit always acts immediately.
    This never authorizes extension or clears an existing storage fault.
    """
    def __init__(self):
        self.armed=False;self.fault=None;self.previous=None;self.burst_since=None
    def observe(self,q,v,seconds):
        if self.fault:return self.result(q,v,seconds)
        if not math.isfinite(seconds) or self.previous is not None and seconds<self.previous:
            self.fault='invalid_storage_clock';return self.result(q,v,seconds)
        self.previous=seconds
        finite=len(q)==len(v)==4 and all(math.isfinite(float(x)) for x in (*q,*v))
        if not finite:self.fault='invalid_storage_measurement'
        elif any(abs(float(x))>lim for x,lim in zip(q,TOLERANCE)):
            self.fault='lift_position_outside_storage'
        elif any(abs(float(x))>lim for x,lim in zip(v,VELOCITY_LIMIT)):
            self.fault='lift_storage_joint_velocity_limit'
        elif not self.armed:
            if stored(q,v):self.armed=True
            else:self.fault='lift_not_settled_before_travel'
        else:
            if any(abs(float(x))>=.06 for x in v):
                if self.burst_since is None:self.burst_since=seconds
                if seconds-self.burst_since>=.1-1e-9:self.fault='persistent_lift_storage_motion'
            else:self.burst_since=None
        return self.result(q,v,seconds)
    def result(self,q,v,seconds):
        duration=0. if self.burst_since is None else max(0.,seconds-self.burst_since)
        return dict(home=self.armed and self.fault is None,settled=bool(stored(q,v)),
                    armed=self.armed,velocity_burst_seconds=duration,fault=self.fault)

class LiftStorage:
    def __init__(self,robot):
        self.robot=robot;self.fault=None;self.shovel_hold=None
        self.interlock=StorageInterlock();self.clock=-1/120
        self.brake_engaged=False
        self.dofs=[robot.get_joint(name).dofs_idx_local[0] for name in JOINTS]
        robot.set_dofs_position(np.zeros(4),dofs_idx_local=self.dofs)
        robot.set_dofs_kp(np.array([1200.,900.,100.,500.]),dofs_idx_local=self.dofs)
        robot.set_dofs_kv(np.array([80.,65.,12.,30.]),dofs_idx_local=self.dofs)
        maximum=np.array([120.,60.,18.,25.])
        robot.set_dofs_force_range(-maximum,maximum,dofs_idx_local=self.dofs)
        robot.control_dofs_position(np.zeros(4),dofs_idx_local=self.dofs)
        robot.set_dofs_frictionloss(np.array([0.,30.,0.,0.]),dofs_idx_local=self.dofs)
        self.brake_engaged=True
    def _change_brake(self,engaged,base_speed,safety_clear):
        o=self.observation()
        if self.fault is not None or not safety_clear or not math.isfinite(base_speed) or abs(base_speed)>=.025 or not o['settled']:
            return False
        self.robot.set_dofs_frictionloss(np.array([0.,30. if engaged else 0.,0.,0.]),dofs_idx_local=self.dofs)
        self.brake_engaged=engaged
        return True
    def release_for_service(self,base_speed,*,safety_clear):
        return self._change_brake(False,base_speed,safety_clear)
    def engage_for_travel(self,base_speed,*,safety_clear):
        return self._change_brake(True,base_speed,safety_clear)
    def observation(self):
        q=self.robot.get_dofs_position(dofs_idx_local=self.dofs).detach().cpu().numpy().astype(float)
        v=self.robot.get_dofs_velocity(dofs_idx_local=self.dofs).detach().cpu().numpy().astype(float)
        position_home=bool(len(q)==4 and np.isfinite(q).all() and all(abs(x)<=lim for x,lim in zip(q,TOLERANCE)))
        return dict(q=q.tolist(),v=v.tolist(),home=position_home and self.fault is None and self.brake_engaged,settled=bool(stored(q,v)),brake_engaged=self.brake_engaged)
    def check(self,seconds=None):
        o=self.observation()
        self.clock=self.clock+1/120 if seconds is None else seconds
        state=self.interlock.observe(o['q'],o['v'],self.clock);self.fault=state['fault']
        if not self.brake_engaged and self.fault is None:
            self.fault=self.interlock.fault='lift_transport_brake_not_engaged';state=self.interlock.result(o['q'],o['v'],self.clock)
        return {**o,**state}
