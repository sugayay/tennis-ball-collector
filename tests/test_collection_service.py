import json,unittest
from pathlib import Path
import numpy as np
from mission_control import MissionRobot,FleetCoordinator
from collection_service import HomeReturn
ROOT=Path(__file__).resolve().parents[1]

class ServiceTests(unittest.TestCase):
    def robot(self,behavior='return_home'):
        c=json.loads((ROOT/'mission_straight_config.json').read_text())
        b=json.loads((ROOT/'basket_camera_config.json').read_text());b['maximum_count']=3
        c.update(collection_service_enabled=True,basket_capacity=3,cycles_per_robot=6,
            full_basket_behavior=behavior,home_positions_m={'A':[-13.6,-6.5,.206],'B':[13.6,6.5,.206]},
            basket_confirmation_config=b,basket_confirmation_required=True)
        r=MissionRobot('A',c,[23.77,10.97]);r.target=np.array([-12.,-6.5,.0335]);r.confirmed_cycles=3
        for i in range(6):r.observe_basket(3,i*.25,True)
        return r

    def test_full_basket_returns_without_claiming_remaining_goal(self):
        r=self.robot();r.complete_target(1.25,'confirmed')
        self.assertEqual(r.stage,'return_home');self.assertEqual(r.confirmed_cycles,3)
        self.assertIsNone(r.target);self.assertEqual(r.service_visits,1)

    def test_partial_count_drop_is_not_unloading_completion(self):
        r=self.robot('stop');r.complete_target(1.25,'confirmed')
        for i in range(6):r.observe_basket(1,2.+i*.25,True)
        w,_=r.command(np.array([-13.,-6.5,.206]),np.eye(3),3.25)
        self.assertEqual(r.stage,'await_unload');np.testing.assert_array_equal(w,[0,0])

    def test_fresh_stable_empty_image_allows_next_collection(self):
        r=self.robot('stop');r.complete_target(1.25,'confirmed')
        for i in range(6):r.observe_basket(0,2.+i*.25,True)
        r.command(np.array([-13.,-6.5,.206]),np.eye(3),3.25)
        self.assertEqual(r.stage,'seek');self.assertEqual(r.confirmed_cycles,3)
        self.assertEqual(r.completed_targets,[])

    def test_home_arrival_requires_low_measured_speed_and_hold(self):
        nav=HomeReturn([0,0,0]);p=np.array([.05,0,0]);r=np.eye(3)
        self.assertFalse(nav.command(p,r,.2,0)[1])
        self.assertFalse(nav.command(p,r,0.,.5)[1])
        self.assertTrue(nav.command(p,r,0.,1.)[1])

    def test_visible_ball_loss_during_return_stops_service(self):
        r=self.robot();r.complete_target(1.25,'confirmed')
        for i in range(6):r.observe_basket(2,2.+i*.25,True)
        w,_=r.command(np.array([-13.,-6.5,.206]),np.eye(3),3.25)
        self.assertEqual(r.stage,'fault');np.testing.assert_array_equal(w,[0,0])

    def test_stale_basket_image_prevents_loaded_return(self):
        r=self.robot();r.complete_target(1.25,'confirmed')
        w,_=r.command(np.array([-13.,-6.5,.206]),np.eye(3),2.)
        self.assertEqual(r.stage,'return_home');np.testing.assert_array_equal(w,[0,0])

    def test_pause_freezes_action_and_stale_camera_remains_stale_after_resume(self):
        r=self.robot();r.stage='visual';r.stage_started=1.;r.last_onboard_time=1.
        r.pause(2.,[-.3,.3])
        w,j=r.command(np.array([-13.,-6.5,.206]),np.eye(3),10.)
        np.testing.assert_array_equal(w,[0,0]);np.testing.assert_allclose(j,[-.3,.3])
        r.resume(12.)
        self.assertEqual(r.stage_started,11.);self.assertEqual(r.last_onboard_time,1.)
