import copy
import json
from pathlib import Path
import unittest

from mission_suite import prepare_case, result_row, suite_summary

ROOT = Path(__file__).resolve().parents[1]


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT/'mission_straight_config.json').read_text(encoding='utf-8'))
        self.cases = json.loads((ROOT/'mission_cases.json').read_text(encoding='utf-8'))['cases']

    def test_five_original_cases_and_long_drive_preserve_one_controller(self):
        before = copy.deepcopy(self.base)
        self.assertEqual([c['id'] for c in self.cases],
            ['01_straight','02_diagonal','03_side_turn','04_far','05_close_spacing','06_long_drive','07_one_metre_drive','08_baseline_long_rgb'])
        for case in self.cases:
            config = prepare_case(self.base,case)
            for key in self.base.keys()-{'test_ball_positions_m','trial_zones'}:
                self.assertEqual(config[key],self.base[key])
        self.assertEqual(self.base,before)

    def test_unsafe_or_overlapping_initialization_is_rejected(self):
        for points in ([[0,0]]*4,[[-12.8,-6.5]]*4):
            case = {**self.cases[0],'ball_positions_m':points}
            with self.assertRaises(ValueError):
                prepare_case(self.base,case)

    def test_execution_failure_is_unknown_not_a_zero_ball_measurement(self):
        case = self.cases[0]
        row = result_row(case,{'status':'running','verified':False},1,3.)
        self.assertEqual(row['status'],'execution_error')
        self.assertIsNone(row['held_ball_count'])
        summary = suite_summary([case],{case['id']:row},{},True,1)
        self.assertEqual(summary['evaluated_ball_count'],0)
        self.assertEqual(summary['expected_ball_count'],4)
        self.assertFalse(summary['all_cases_collected_all_balls'])

    def test_failed_physics_report_is_a_completed_measurement(self):
        case = self.cases[0]
        report = {'verified':False,'checks':{'retention':False},
                  'physics':{'A':{'last_3_seconds_held_ball_indices':[0]},'B':{'last_3_seconds_held_ball_indices':[2,3]}},
                  'maximum_ball_heights_m':[.62,.1,.62,.62],'events':[],
                  'retries':{'A':0,'B':0},'simulated_seconds':45.,
                  'controller_uses_simulator_ball_state':False,'ball_state_modified_during_episode':False,
                  'model_sha256':'same-model'}
        row = result_row(case,report,1,100.)
        self.assertEqual(row['status'],'complete')
        self.assertEqual(row['held_ball_count'],3)
        self.assertEqual(row['uncollected_ball_indices'],[1])
        self.assertFalse(row['all_physical_checks_passed'])


if __name__=='__main__':
    unittest.main()
