"""Exercise the real fleet safety branch with pure controllers and joint reads."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from mission_control import FleetCoordinator
from person_safety import PersonPauseGate

ROOT = Path(__file__).resolve().parents[1]


def runtime_safety_branch():
    tree = ast.parse((ROOT / 'fleet_mission.py').read_text(encoding='utf-8'))
    # Extract production statements by semantic boundaries, avoiding a duplicate
    # implementation or importing Genesis just for the integration boundary.
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and ast.unparse(node.target) == '(rid, robot)'
                and ast.unparse(node.iter) == 'control.robots.items()'
                and any(isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == '(wheels, joints)'
                        for item in node.body))
    start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == '(wheels, joints)')
    end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
               and ast.unparse(node.targets[0]) == 'forecast')
    return compile(ast.Module(body=loop.body[start:end], type_ignores=[]),
                   str(ROOT / 'fleet_mission.py'), 'exec')


class PersonFaultIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.branch = runtime_safety_branch()

    def setUp(self):
        config = json.loads((ROOT / 'mission_straight_config.json').read_text())
        config['automatic_person_resume'] = True
        self.fleet = FleetCoordinator(config, [23.77, 10.97])
        self.robot = self.fleet.robots['A']
        self.fleet.robots['B'].stop(0., 'test_inactive_peer')
        self.gate = PersonPauseGate(config['person_safety'], ['A', 'B'])
        self.pose = np.array([-13.6, -6.5, .206])
        self.actual_joints = np.array([-.32, .32])
        self.env = dict(robot=self.robot, p=self.pose, rotation=np.eye(3), rid='A',lift={},lift_trace=[],
                        mission=config, safety={'A': self.gate}, safety_joints={},
                        np=np, as_array=np.asarray, shovel_dofs={'A': [0, 1]},
                        robots={'A': SimpleNamespace(get_dofs_position=lambda **kw: self.actual_joints.copy())})

    def step(self, seconds):
        self.env['now'] = seconds
        exec(self.branch, self.env)
        return self.env['wheels'], self.env['joints']

    def observe(self, seconds, nearby=False, unknown=False):
        people = [{'position_m': (self.pose + [2., 0., 0.]).tolist()}] if nearby else []
        self.gate.observe('A', people, seconds, int(unknown))
        self.gate.observe('B', [], seconds)

    def assert_held(self):
        np.testing.assert_array_equal(self.env['wheels'], [0., 0.])
        np.testing.assert_array_equal(self.env['joints'], self.actual_joints)

    def test_camera_loss_during_pause_finishes_as_fault_and_stays_stopped(self):
        self.observe(0., nearby=True); self.step(0.)
        self.assertIsNotNone(self.robot.paused_since)
        self.step(1.)
        self.assertEqual(self.robot.stage, 'fault')
        self.assertEqual(self.robot.events[-1]['reason'], 'safety_vision_stale')
        self.assertIsNone(self.robot.paused_since)
        self.assertTrue(self.fleet.finished)
        self.assert_held()
        for seconds in (1.5, 2., 2.5, 3., 3.5):
            self.observe(seconds); self.step(seconds)
            self.assertEqual(self.robot.stage, 'fault')
            self.assert_held()
        faults = [event for event in self.robot.events if event['to'] == 'fault']
        self.assertEqual(len(faults), 1)
        self.assertEqual(self.gate.resume_count, 0)

    def test_unknown_position_during_pause_also_escalates_to_fault(self):
        self.observe(0., nearby=True); self.step(0.)
        self.observe(.2, unknown=True); self.step(.2)
        self.assertEqual(self.robot.stage, 'fault')
        self.assertEqual(self.robot.events[-1]['reason'], 'person_position_unknown')
        self.assertIsNone(self.robot.paused_since)
        self.assert_held()

    def test_normal_person_clearance_still_resumes(self):
        self.observe(0., nearby=True); self.step(0.)
        for index in range(1, 25):
            seconds = index*.5
            self.gate.observe('A', [{'position_m': (self.pose+[2.+1.4*seconds,0,0]).tolist()}], seconds)
            self.gate.observe('B', [], seconds); self.step(seconds)
            if seconds < 7.:
                self.assertIsNotNone(self.robot.paused_since)
                self.assert_held()
        self.assertEqual(self.robot.stage, 'seek')
        self.assertIsNone(self.robot.paused_since)
        self.assertEqual(self.gate.resume_count, 1)
        self.assertNotIn('A', self.env['safety_joints'])

    def test_fresh_empty_detections_during_pause_finish_as_tracking_fault(self):
        self.observe(0., nearby=True); self.step(0.)
        self.observe(.15); self.step(.15)
        self.assertEqual(self.robot.stage, 'fault')
        self.assertEqual(self.robot.events[-1]['reason'], 'person_track_lost')
        self.assertIsNone(self.robot.paused_since)
        for seconds in (.3, .6, 1., 1.5, 2.):
            self.observe(seconds, nearby=True); self.step(seconds)
            self.assertEqual(self.robot.stage, 'fault')
            self.assert_held()
        self.assertEqual(self.gate.resume_count, 0)

    def test_guard_fault_during_pause_does_not_restart_after_clearance(self):
        self.observe(0., nearby=True); self.step(0.)
        # Simulate a guard fault while the robot is held at its measured pose.
        self.robot.stop(.1, 'robot_separation_guard')
        self.assertIsNone(self.robot.paused_since)
        for seconds in (.5, 1., 1.5, 2., 2.1):
            self.observe(seconds); self.step(seconds)
            self.assertEqual(self.robot.stage, 'fault')
            self.assert_held()

    def test_pickup_fault_keeps_measured_joint_pose_after_person_leaves(self):
        self.robot.stage = 'secure'
        self.robot.cycle_index = 1
        self.robot.stage_started = 0.
        self.robot.heading_reference = 0.
        self.observe(.05, nearby=True); self.step(.05)
        self.assertEqual(self.robot.stage, 'fault')
        self.assertEqual(self.robot.events[-1]['reason'], 'person_proximity_stop')
        for seconds in (.5, 1., 1.5, 2., 2.1):
            self.observe(seconds); self.step(seconds)
            self.assertEqual(self.robot.stage, 'fault')
            self.assert_held()

    def test_lift_storage_fault_stops_and_keeps_measured_shovel_pose(self):
        holder=SimpleNamespace(shovel_hold=None,check=lambda seconds:dict(q=[.1,0,0,0],v=[0.]*4,home=False,fault='lift_not_stored_during_collection'))
        self.env['lift']['A']=holder
        self.observe(0.);self.step(0.)
        self.assertEqual(self.robot.stage,'fault');self.assert_held()
        for t in (.25,.5,1.):
            self.observe(t);self.step(t)
            self.assertEqual(self.robot.stage,'fault');self.assert_held()


if __name__ == '__main__':
    unittest.main()
