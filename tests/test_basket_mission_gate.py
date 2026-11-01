import copy,json,unittest
from pathlib import Path
import numpy as np
from mission_control import MissionRobot,CYCLE

ROOT=Path(__file__).resolve().parents[1]

class BasketMissionGateTests(unittest.TestCase):
    def robot(self,count=1,baseline=True):
        cfg=json.loads((ROOT/'mission_straight_config.json').read_text());cfg.pop('test_ball_positions_m')
        cfg['basket_confirmation_config']=json.loads((ROOT/'basket_camera_config.json').read_text())
        cfg['basket_confirmation_required']=True
        r=MissionRobot('A',cfg,[23.77,10.97]);r.target=np.array([-12.,-6.5,.0335])
        m=r.basket_confirmation
        if baseline:
            for i in range(6):r.observe_basket(0,i*.25,True)
        m.begin_attempt(1,1.3);m.begin_confirmation(10)
        for i in range(6):r.observe_basket(count,10+i*.25,True)
        r.stage='retain';r.stage_started=8.3;r.cycle_index=len(CYCLE)-1;r.heading_reference=0.
        return r

    def command(self,r,t):return r.command(np.array([-13.,-6.5,.205]),np.eye(3),t)

    def test_mechanism_completion_does_not_mark_target_complete(self):
        r=self.robot();w,_=self.command(r,11.3)
        self.assertEqual(r.stage,'confirm');self.assertEqual(r.cycles,1)
        self.assertEqual(r.confirmed_cycles,0);self.assertEqual(r.completed_targets,[])
        np.testing.assert_array_equal(w,[0,0])

    def test_rgb_increase_unlocks_next_target(self):
        r=self.robot();self.command(r,11.3);self.command(r,11.32)
        self.assertEqual(r.stage,'seek');self.assertEqual(r.confirmed_cycles,1)
        self.assertEqual(len(r.completed_targets),1)
        self.assertEqual(r.events[-1]['reason'],'basket_rgb_count_increased')

    def test_no_increase_stops_without_completing_target(self):
        r=self.robot(0);self.command(r,11.3)
        for i in range(1,10):r.observe_basket(0,11.25+i*.25,True)
        self.command(r,13.51)
        self.assertEqual(r.stage,'fault');self.assertEqual(r.completed_targets,[])
        self.assertEqual(r.events[-1]['reason'],'basket_rgb:no_increase_observed')

    def test_lost_camera_stops_as_unknown(self):
        r=self.robot(0);self.command(r,11.3);self.command(r,13.51)
        self.assertEqual(r.stage,'fault');self.assertEqual(r.confirmed_cycles,0)
        self.assertEqual(r.events[-1]['reason'],'basket_rgb:unknown')

    def test_missing_baseline_never_confirms_visible_ball(self):
        r=self.robot(1,baseline=False);self.command(r,11.3);self.command(r,11.32)
        self.assertEqual(r.stage,'confirm');self.assertEqual(r.completed_targets,[])

    def test_second_cycle_needs_second_confirmed_increase(self):
        r=self.robot();r.cycles=1;r.confirmed_cycles=1
        r.basket_confirmation.attempt['cycle']=2
        r.basket_confirmation.attempt['baseline_observed_count']=1
        self.command(r,11.3);self.command(r,11.32)
        self.assertEqual(r.stage,'confirm');self.assertEqual(r.confirmed_cycles,1)
        for i in range(6):r.observe_basket(2,11.5+i*.25,True)
        self.command(r,12.8)
        self.assertEqual(r.stage,'done');self.assertEqual(r.confirmed_cycles,2)

if __name__=='__main__':unittest.main()
