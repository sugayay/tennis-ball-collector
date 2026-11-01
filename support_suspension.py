"""Fixed-reference, force-limited spring/damper model for two caster sliders.

Call once at scene initialization. There is no running ride-height controller,
task-state input, or later position correction. Genesis applies the constant
PD reference as F = k(rest - q) - c*qdot within the declared force limit.
"""
from pathlib import Path
import json
import math
import numpy as np

JOINTS = ('front_caster_suspension_joint', 'rear_caster_suspension_joint')
CONFIG = Path(__file__).with_name('support_suspension_config.json')


def settings(config=None):
    config = json.loads(CONFIG.read_text(encoding='utf8')) if config is None else config
    keys = ('stiffness_N_m', 'damping_Ns_m', 'rest_position_m', 'force_limit_N')
    values = [config[key] for key in keys]
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in values):
        raise ValueError('Suspension parameters must be finite numbers.')
    k, c, rest, force = map(float, values)
    if k <= 0 or c <= 0 or not 0 < force <= 120:
        raise ValueError('Positive spring/damper gains and force <= URDF120N required.')
    stroke = config.get('stroke_m')
    if (not isinstance(stroke, (list, tuple)) or len(stroke) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in stroke)
            or list(stroke) != [-.025, .025] or not stroke[0] < rest < stroke[1]):
        raise ValueError('Rest must be inside the unchanged URDF +/-25mm stroke.')
    return k, c, rest, force


def configure_if_present(robot, config=None):
    names = {joint.name for joint in robot.joints}
    present = [name in names for name in JOINTS]
    if not any(present):
        return None
    if not all(present):
        raise ValueError('Both support suspension joints are required.')
    k, c, rest, force = settings(config)
    dofs = [robot.get_joint(name).dofs_idx_local[0] for name in JOINTS]
    if len(set(dofs)) != 2 or any(not isinstance(x, (int, np.integer)) or x < 0 for x in dofs):
        raise ValueError('Distinct support suspension DOFs required.')
    robot.set_dofs_position(np.zeros(2), dofs_idx_local=dofs)
    robot.set_dofs_kp(np.full(2, k), dofs_idx_local=dofs)
    robot.set_dofs_kv(np.full(2, c), dofs_idx_local=dofs)
    robot.set_dofs_force_range(np.full(2, -force), np.full(2, force), dofs_idx_local=dofs)
    robot.control_dofs_position(np.full(2, rest), dofs_idx_local=dofs)
    return {'joint_names': list(JOINTS), 'dofs': list(map(int, dofs)),
            'fixed_rest_position_m': rest, 'stiffness_N_m': k,
            'damping_Ns_m': c, 'force_limit_N': force}
