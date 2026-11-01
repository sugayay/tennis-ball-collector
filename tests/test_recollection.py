import copy,json,unittest
from pathlib import Path
import numpy as np
from mission_control import MissionRobot,FleetCoordinator,CYCLE,PICKUP
from recollection import select_recollection,validate_recovery,corroborate_fixed_with_onboard
from rolling_tracker import RollingBallTracker

ROOT=Path(__file__).resolve().parents[1]
read=lambda name:json.loads((ROOT/name).read_text())
POLICY=dict(playing_area_m=[23.77,10.97],court_size_m=[34.73,17.07],trial_zones=read('mission_straight_config.json')['trial_zones'])

def detection(x=-12.8,y=-6.5,dual=True):
    p=[x,y,.0335]
    return dict(position_m=p,confidence=.9,observations=[dict(camera_id=k,position_m=p) for k in ('A','B')[:2 if dual else 1]])

def track(x=-12.8,y=-6.5,**changes):
    row=dict(track_id=1,position_m=[x,y],fresh_observation=True,state='observed',stationary=True,dual_view_agreement=True)
    row.update(changes);return row

class SelectionTests(unittest.TestCase):
    def choose(self,rows,excluded=()):return select_recollection(rows,[-12.8,-6.5,.0335],'A',POLICY,.75,excluded)
    def test_unique_stationary_dual_view_can_be_selected(self):self.assertEqual(self.choose([track()])[0]['track_id'],1)
    def test_moving_candidate_is_withheld(self):self.assertIsNone(self.choose([track(stationary=False)])[0])
    def test_single_view_is_withheld(self):self.assertIsNone(self.choose([track(dual_view_agreement=False)])[0])
    def test_stale_candidate_is_withheld(self):self.assertIsNone(self.choose([track(fresh_observation=False)])[0])
    def test_tentative_candidate_is_withheld(self):self.assertIsNone(self.choose([track(state='tentative')])[0])
    def test_second_moving_candidate_still_makes_identity_ambiguous(self):
        chosen,reason=self.choose([track(),track(x=-12.5,track_id=2,stationary=False)])
        self.assertIsNone(chosen);self.assertEqual(reason,'ambiguous_local_candidates')
    def test_completed_or_other_reserved_target_excluded(self):self.assertIsNone(self.choose([track()],[[-12.8,-6.5]])[0])
    def test_far_new_ball_is_not_substituted(self):self.assertIsNone(self.choose([track(x=-11.5)])[0])
    def test_court_interior_is_rejected(self):
        self.assertIsNone(select_recollection([track(-11.,-5.)],[-11.,-5.,.0335],'A',POLICY,.75,[])[0])
    def test_other_robot_zone_is_rejected(self):
        self.assertIsNone(select_recollection([track(12.8,6.5)],[12.8,6.5,.0335],'A',POLICY,.75,[])[0])
    def test_unbounded_retry_configuration_rejected(self):
        cfg=read('recovery_config.json');cfg['maximum_attempts_per_target']=3
        with self.assertRaises(ValueError):validate_recovery(cfg)

class IndependentViewTests(unittest.TestCase):
    def fixed(self,x=-12.8):
        d=detection(x,dual=False);d['sources']=['A'];return d
    def ego(self,x=-12.81):return dict(position_m=[x,-6.5,.0335],confidence=.8)
    def test_fixed_and_ego_corroborate_without_changing_fixed_position(self):
        fixed=[self.fixed()];before=copy.deepcopy(fixed)
        merged=corroborate_fixed_with_onboard(fixed,{'onboard-A':[self.ego()]})
        self.assertEqual(fixed,before);self.assertEqual(merged[0]['position_m'],fixed[0]['position_m'])
        t=RollingBallTracker().update(merged,{'A':[-13.8,-6.5]},0)['tracks'][0]
        self.assertTrue(t['dual_view_agreement']);self.assertEqual(t['camera_ids'],['A','onboard-A'])
    def test_far_disagreement_never_corroborates(self):
        merged=corroborate_fixed_with_onboard([self.fixed()],{'onboard-A':[self.ego(-12.5)]})
        self.assertEqual(merged[0]['sources'],['A'])
    def test_ego_only_does_not_create_candidate(self):
        self.assertEqual(corroborate_fixed_with_onboard([],{'onboard-A':[self.ego()]}),[])
    def test_two_ego_detections_cannot_choose_one_arbitrarily(self):
        merged=corroborate_fixed_with_onboard([self.fixed()],{'onboard-A':[self.ego(),self.ego(-12.79)]})
        self.assertEqual(merged[0]['sources'],['A'])
    def test_one_ego_cannot_corroborate_two_fixed_candidates(self):
        merged=corroborate_fixed_with_onboard([self.fixed(),self.fixed(-12.82)],{'onboard-A':[self.ego()]})
        self.assertTrue(all(m['sources']==['A'] for m in merged))

class RecoveryControlTests(unittest.TestCase):
    def config(self):
        cfg=read('mission_straight_config.json');cfg.pop('test_ball_positions_m')
        cfg.update(basket_confirmation_config=read('basket_camera_config.json'),basket_confirmation_required=True,
            recovery_config=read('recovery_config.json'),perception_floor_size_m=POLICY['court_size_m'])
        return cfg
    def waiting(self):
        r=MissionRobot('A',self.config(),POLICY['playing_area_m'])
        r.target=np.array([-12.8,-6.5,.0335]);r.cycles=1;r.stage='confirm';r.stage_started=10.
        for i in range(6):r.observe_basket(0,i*.25,True)
        r.basket_confirmation.begin_attempt(1,1.3);r.basket_confirmation.begin_confirmation(9.)
        for i in range(13):r.observe_basket(0,9+i*.25,True)
        r.observe_fixed([],np.array([-13.1,-6.5,.205]),12.)
        return r
    def command(self,r,seconds,x=-13.1):return r.command(np.array([x,-6.5,.205]),np.eye(3),seconds)
    def test_no_increase_starts_bounded_recollection_and_does_not_complete(self):
        r=self.waiting();w,_=self.command(r,12.01)
        self.assertEqual(r.stage,'recover_backoff');self.assertEqual(r.confirmed_cycles,0)
        self.assertEqual(r.completed_targets,[]);self.assertEqual(r.recollection_total,1)
        np.testing.assert_array_equal(w,[0,0])
    def test_backoff_commands_reverse_only_while_fresh(self):
        r=self.waiting();self.command(r,12.01);w,_=self.command(r,12.02)
        self.assertTrue(all(w<0));self.assertTrue(all(abs(w*.11)<=.18))
    def test_backoff_stale_camera_latches_stop(self):
        r=self.waiting();self.command(r,12.01)
        for t in [12.25,12.5,12.75]:r.observe_basket(0,t,True)
        w,_=self.command(r,12.8)
        self.assertEqual(r.stage,'fault');self.assertEqual(r.events[-1]['reason'],'recollection_fixed_camera_stale')
        np.testing.assert_array_equal(w,[0,0])
    def test_stationary_wait_can_await_late_ball_frame_without_moving(self):
        r=self.waiting();self.command(r,12.01);r.stage='recover_wait';r.stage_started=12.1
        for t in (12.25,12.5,12.75):r.observe_basket(0,t,True)
        w,_=self.command(r,12.8)
        self.assertEqual(r.stage,'recover_wait');self.assertEqual(r.last_status,'waiting_for_fresh_fixed_camera')
        np.testing.assert_array_equal(w,[0,0]);self.assertEqual(r.confirmed_cycles,0)
    def test_stationary_wait_still_times_out_when_ball_camera_never_recovers(self):
        r=self.waiting();self.command(r,12.01);r.stage='recover_wait';r.stage_started=12.1
        for t in np.arange(22.5,24.51,.25):r.observe_basket(0,float(t),True)
        w,_=self.command(r,24.5)
        self.assertEqual(r.stage,'fault');np.testing.assert_array_equal(w,[0,0])
        self.assertEqual(r.events[-1]['reason'],'recollection_no_safe_candidate:waiting_for_fresh_fixed_camera')
    def test_unknown_basket_never_starts_recollection(self):
        r=self.waiting();r.observe_basket(0,12.25,False);self.command(r,12.3)
        self.assertEqual(r.stage,'fault');self.assertEqual(r.recollection_total,0)
    def test_existing_ball_is_not_reinterpreted_after_recovery(self):
        r=self.waiting();self.command(r,12.01);r.observe_basket(1,12.25,True)
        self.command(r,12.3);self.assertEqual(r.stage,'fault');self.assertEqual(r.confirmed_cycles,0)
    def test_retry_limit_cannot_loop_forever(self):
        r=self.waiting();r.recollection_attempts=2;self.command(r,12.01)
        self.assertEqual(r.stage,'fault');self.assertEqual(r.events[-1]['reason'],'recollection_retry_limit')
    def test_reversing_outside_trial_zone_is_rejected_before_moving(self):
        r=self.waiting();w,_=self.command(r,12.01,x=-14.5)
        self.assertEqual(r.stage,'fault');np.testing.assert_array_equal(w,[0,0])
    def test_recovery_wait_times_out_at_zero_command(self):
        r=self.waiting();self.command(r,12.01);r.stage='recover_wait';r.stage_started=0.
        w,_=self.command(r,12.02);self.assertEqual(r.stage,'fault');np.testing.assert_array_equal(w,[0,0])
    def test_fault_is_not_resumed_by_fresh_observations(self):
        r=self.waiting();r.stop(12,'person_proximity_stop');r.observe_fixed([detection()],np.array([-13.1,-6.5,.205]),12.01)
        w,_=self.command(r,12.02);self.assertEqual(r.stage,'fault');np.testing.assert_array_equal(w,[0,0])
    def test_failed_attempt_does_not_count_towards_goal(self):
        r=self.waiting();r.cycles=2;r.confirmed_cycles=1
        r.complete_target(12.1,'confirmed_increase');self.assertEqual(r.stage,'seek')
    def test_three_attempts_with_two_confirmations_can_finish(self):
        r=self.waiting();r.cycles=3;r.confirmed_cycles=2
        r.complete_target(12.1,'confirmed_increase');self.assertEqual(r.stage,'done')
    def test_explicit_fault_fixture_holds_all_actuators_and_expires_after_attempt(self):
        r=self.waiting();r.cycles=0;r.test_empty_scoop_attempts=1;r.heading_reference=0.;r.stage_started=12.
        r.scoop_start_position=np.array([-13.1,-6.5,.205])
        for index,(name,*_) in enumerate(CYCLE):
            r.stage=name;r.cycle_index=index
            wheels,joints=self.command(r,12.01)
            np.testing.assert_array_equal(wheels,[0,0]);np.testing.assert_array_equal(joints,[PICKUP,-PICKUP])
        r.cycles=1;r.stage='scoop';r.cycle_index=0
        wheels,_=self.command(r,12.02);self.assertGreater(wheels.mean(),0.)
    def test_streaming_tracker_dispatches_only_after_observed_rest(self):
        fleet=FleetCoordinator(self.config(),POLICY['playing_area_m'])
        a=fleet.robots['A'];fleet.robots['B'].stop(0,'test_other_robot_inactive')
        a.target=np.array([-12.8,-6.5,.0335]);a.recovery_origin=a.target.copy();a.recovery_baseline=0
        a.stage='recover_wait';a.stage_started=0.;a.cycles=1
        poses={'A':np.array([-13.6,-6.5,.205]),'B':np.array([13.6,6.5,.205])}
        for i in range(7):
            t=i*.25;a.observe_basket(0,t,True)
            if i%2==0:fleet.observe_fixed([detection()],poses,t)
            if t<1.5:self.assertEqual(a.stage,'recover_wait')
        self.assertEqual(a.stage,'fixed');self.assertEqual(len(fleet.recoveries),1)
        self.assertEqual(a.confirmed_cycles,0);self.assertEqual(a.cycles,1)

if __name__=='__main__':unittest.main()
