"""Temporal RGB count evidence, not simulator success/ball-state feedback."""
from __future__ import annotations
import math


class BasketConfirmation:
    def __init__(self,config):
        self.config=dict(config)
        for name in ('stable_seconds','maximum_frame_gap_seconds','frame_max_age_seconds'):
            if not math.isfinite(config[name]) or config[name]<=0:raise ValueError(name)
        for name in ('minimum_samples','maximum_count'):
            if not isinstance(config[name],int) or config[name]<1:raise ValueError(name)
        self.samples=[];self.last_seen=None;self.last_invalid_reason=None
        self.attempt=None;self.records=[]

    def observe(self,count,seconds,valid=True):
        if not math.isfinite(seconds) or seconds<0 or (self.last_seen is not None and seconds<=self.last_seen):
            raise ValueError('Basket frame time must increase')
        gap=self.last_seen is not None and seconds-self.last_seen>self.config['maximum_frame_gap_seconds']+1e-9
        self.last_seen=seconds
        if not valid or not isinstance(count,int) or not 0<=count<=self.config['maximum_count']:
            self.samples=[];self.last_invalid_reason='invalid_or_occluded_image';return
        if gap or (self.samples and self.samples[-1][1]!=count):self.samples=[]
        self.last_invalid_reason=None
        self.samples.append((seconds,count))
        # Bounded buffer retains more than one complete confirmation window.
        horizon=max(3.,2*self.config['stable_seconds'])
        self.samples=[sample for sample in self.samples if seconds-sample[0]<=horizon+1e-9]

    def stable_count(self,seconds):
        if self.last_seen is None or seconds<self.last_seen or seconds-self.last_seen>self.config['frame_max_age_seconds']+1e-9:
            return None
        if len(self.samples)<self.config['minimum_samples']:return None
        if self.samples[-1][0]-self.samples[0][0]<self.config['stable_seconds']-1e-9:return None
        return self.samples[-1][1]

    def begin_attempt(self,cycle,seconds):
        self.attempt=dict(cycle=cycle,started_seconds=seconds,
            baseline_observed_count=self.stable_count(seconds),
            baseline_last_seen_seconds=self.last_seen,confirmation_started_seconds=None)

    def begin_confirmation(self,seconds):
        if self.attempt is None:raise RuntimeError('No active basket attempt')
        self.attempt['confirmation_started_seconds']=seconds
        self.samples=[]

    def outcome(self,seconds):
        baseline=self.attempt['baseline_observed_count'] if self.attempt else None
        count=self.stable_count(seconds)
        stage_started=self.attempt and self.attempt['confirmation_started_seconds'] is not None
        if not stage_started or baseline is None or count is None:
            status,reason='unknown','baseline_or_fresh_stable_confirmation_missing'
        elif count==baseline+1:
            status,reason='confirmed_increase','one_more_ball_seen_in_lowered_basket_view'
        elif count==baseline:
            # A detector miss/occlusion can look like no increase. This is NOT
            # proof of an empty basket or permission to chase another target.
            status,reason='no_increase_observed','same_visible_count_not_proof_of_physical_failure'
        else:
            status,reason='unknown','unexpected_count_change'
        return dict(status=status,reason=reason,baseline_observed_count=baseline,
            final_observed_count=count,seconds=seconds,
            confirmation_samples=len(self.samples),
            confirmation_span_seconds=self.samples[-1][0]-self.samples[0][0] if self.samples else 0.)

    def finish(self,seconds):
        result={**(self.attempt or {}),**self.outcome(seconds)}
        self.records.append(result);self.attempt=None
        return result
