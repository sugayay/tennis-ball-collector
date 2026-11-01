import json,unittest
from pathlib import Path
from basket_confirmation import BasketConfirmation

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'basket_camera_config.json').read_text())

class BasketConfirmationTests(unittest.TestCase):
    def monitor(self,baseline=0):
        m=BasketConfirmation(CONFIG)
        for i in range(6):m.observe(baseline,i*.25)
        m.begin_attempt(1,1.3);m.begin_confirmation(10.)
        return m

    def samples(self,m,count,start=10.,valid=True):
        for i in range(6):m.observe(count,start+i*.25,valid)

    def test_zero_to_one_requires_stable_new_count(self):
        m=self.monitor();self.samples(m,1)
        self.assertEqual(m.outcome(11.3)['status'],'confirmed_increase')

    def test_one_to_two_counts_new_ball_not_old_ball(self):
        m=self.monitor(1);self.samples(m,2)
        self.assertEqual(m.finish(11.3)['status'],'confirmed_increase')
        self.assertIsNone(m.attempt)

    def test_existing_ball_is_not_new_success(self):
        m=self.monitor(1);self.samples(m,1)
        self.assertEqual(m.outcome(11.3)['status'],'no_increase_observed')

    def test_empty_observation_is_not_success_or_proven_failure(self):
        m=self.monitor();self.samples(m,0)
        row=m.outcome(11.3)
        self.assertEqual(row['status'],'no_increase_observed')
        self.assertIn('not_proof',row['reason'])

    def test_single_high_outlier_does_not_confirm(self):
        m=self.monitor();m.observe(1,10.)
        self.assertEqual(m.outcome(10.)['status'],'unknown')
        self.samples(m,0,10.25)
        self.assertEqual(m.outcome(11.5)['status'],'no_increase_observed')

    def test_fast_burst_cannot_replace_elapsed_observation(self):
        m=self.monitor()
        for i in range(20):m.observe(1,10+i*.01)
        self.assertEqual(m.outcome(10.2)['status'],'unknown')

    def test_stale_confirmation_is_unknown(self):
        m=self.monitor();self.samples(m,1)
        self.assertEqual(m.outcome(12.)['status'],'unknown')

    def test_missing_baseline_cannot_be_invented(self):
        m=BasketConfirmation(CONFIG);m.begin_attempt(1,0);m.begin_confirmation(10)
        self.samples(m,1)
        self.assertEqual(m.outcome(11.3)['status'],'unknown')

    def test_stale_baseline_cannot_confirm(self):
        m=BasketConfirmation(CONFIG)
        for i in range(6):m.observe(0,i*.25)
        m.begin_attempt(1,2);m.begin_confirmation(10);self.samples(m,1)
        self.assertEqual(m.outcome(11.3)['status'],'unknown')

    def test_old_frames_cannot_confirm_after_lowering(self):
        m=BasketConfirmation(CONFIG)
        for i in range(6):m.observe(0,i*.25)
        m.begin_attempt(1,1.3);m.begin_confirmation(1.4)
        self.assertEqual(m.outcome(1.4)['status'],'unknown')

    def test_occlusion_breaks_confirmation_window(self):
        m=self.monitor();self.samples(m,1);m.observe(1,11.5,False)
        self.assertEqual(m.outcome(11.5)['status'],'unknown')
        m.observe(1,11.75)
        self.assertEqual(m.outcome(11.75)['status'],'unknown')

    def test_frame_gap_breaks_confirmation_window(self):
        m=self.monitor();self.samples(m,1);m.observe(1,12.)
        self.assertEqual(m.outcome(12.)['status'],'unknown')

    def test_lost_ball_is_unknown(self):
        m=self.monitor(1);self.samples(m,0)
        self.assertEqual(m.outcome(11.3)['status'],'unknown')

    def test_two_new_balls_is_ambiguous_for_single_pickup(self):
        m=self.monitor();self.samples(m,2)
        self.assertEqual(m.outcome(11.3)['status'],'unknown')

    def test_invalid_counts_are_unknown(self):
        for count in [None,3,-1,float('nan'),1.5]:
            m=self.monitor();self.samples(m,count)
            self.assertEqual(m.outcome(11.3)['status'],'unknown')

    def test_timestamp_regression_rejected(self):
        m=self.monitor();m.observe(0,10)
        for time in [10,9,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):m.observe(1,time)

if __name__=='__main__':unittest.main()
