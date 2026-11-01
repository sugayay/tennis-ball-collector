"""One physical wheel-servo configuration shared by mission and qualifiers."""
from pathlib import Path
import json,math
import numpy as np

def settings(config=None):
    if config is None:config=json.loads((Path(__file__).resolve().parent/'drive_actuator_config.json').read_text(encoding='utf8'))
    gain=config['wheel_velocity_gain'];torque=config['maximum_wheel_torque_nm']
    if type(gain) not in (int,float) or not math.isfinite(gain) or not 0<gain<=12:raise ValueError('Wheel gain must be finite in0..12')
    if type(torque) not in (int,float) or not math.isfinite(torque) or not 0<torque<=6:raise ValueError('Wheel torque must be finite in0..6Nm')
    return float(gain),float(torque)

def configure_wheels(robot,config=None):
    gain,torque=settings(config)
    dofs=[robot.get_joint(name).dofs_idx_local[0] for name in ('left_wheel_joint','right_wheel_joint')]
    robot.set_dofs_kp(np.zeros(2),dofs_idx_local=dofs)
    robot.set_dofs_kv(np.full(2,gain),dofs_idx_local=dofs)
    robot.set_dofs_force_range(np.full(2,-torque),np.full(2,torque),dofs_idx_local=dofs)
    robot.control_dofs_velocity(np.zeros(2),dofs_idx_local=dofs)
    return dofs
