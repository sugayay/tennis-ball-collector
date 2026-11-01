"""Latched proximity stop using fresh, RGB-derived person foot positions only."""
from __future__ import annotations

import math

import numpy as np
from person_continuity import PersonContinuity


class PersonStopGate:
    def __init__(self,config,camera_ids):
        self.config=dict(config)
        self.camera_ids=tuple(camera_ids)
        if not self.camera_ids or len(set(self.camera_ids))!=len(self.camera_ids):
            raise ValueError('Distinct safety camera IDs are required.')
        defaults=dict(frame_max_age_seconds=.65,robot_radius_m=.63,person_radius_m=.65,
                      position_margin_m=.25,person_speed_bound_m_s=1.5,reaction_seconds=.15,
                      braking_deceleration_m_s2=.5)
        self.parameters={k:self.config.get(k,v) for k,v in defaults.items()}
        if not all(math.isfinite(v) and v>=0 for v in self.parameters.values()):
            raise ValueError('Invalid person stop parameters.')
        if self.parameters['frame_max_age_seconds']<=0 or self.parameters['braking_deceleration_m_s2']<=0:
            raise ValueError('Freshness and braking limits must be positive.')
        self.frames={rid:None for rid in self.camera_ids}
        self.latched=False
        self.event=None
        self.continuity=PersonContinuity(self.parameters['person_speed_bound_m_s'],
                                         self.parameters['position_margin_m'])

    def observe(self,camera_id,people,seconds,unlocalized_count=0):
        if camera_id not in self.frames:raise ValueError('Unknown safety camera.')
        if not math.isfinite(seconds):return
        previous=self.frames[camera_id]
        if previous is not None and seconds<=previous['seconds']:return
        positions=[];observations=[]
        for person in people:
            p=np.asarray(person['position_m'],dtype=float)
            if p.shape==(3,) and np.isfinite(p).all():
                positions.append(p.copy())
                observations.append({**person,'position_m':p.tolist()})
            else:unlocalized_count+=1
        self.frames[camera_id]=dict(seconds=float(seconds),positions=positions,
                                    unlocalized_count=int(unlocalized_count),people=observations)

    def stopping_radius(self,speed):
        p=self.parameters
        return (p['robot_radius_m']+p['person_radius_m']+p['position_margin_m']
            +p['person_speed_bound_m_s']*(p['frame_max_age_seconds']+p['reaction_seconds']+speed/p['braking_deceleration_m_s2'])
            +speed*p['reaction_seconds']+speed*speed/(2*p['braking_deceleration_m_s2']))

    def _stop(self,seconds,reason,**evidence):
        self.latched=True
        self.event=dict(seconds=float(seconds),reason=reason,**evidence)
        return True

    def blocked(self,position,measured_speed,commanded_speed,seconds):
        if self.latched:return True
        position=np.asarray(position,dtype=float)
        if (position.shape!=(3,) or not np.isfinite(position).all()
                or not all(math.isfinite(x) for x in (measured_speed,commanded_speed,seconds))):
            return self._stop(seconds if math.isfinite(seconds) else 0.,'invalid_safety_feedback')
        for rid in self.camera_ids:
            frame=self.frames[rid]
            if frame is None or not 0<=seconds-frame['seconds']<=self.parameters['frame_max_age_seconds']:
                return self._stop(seconds,'safety_vision_stale',camera_id=rid)
            if frame['unlocalized_count']:
                return self._stop(seconds,'person_position_unknown',camera_id=rid)
        fault=self.continuity.update(self.frames)
        if fault:return self._stop(seconds,**fault)
        speed=max(abs(measured_speed),abs(commanded_speed))
        radius=self.stopping_radius(speed)
        for rid,frame in self.frames.items():
            for person in frame['positions']:
                distance=float(np.linalg.norm(person[:2]-position[:2]))
                if distance<=radius:
                    return self._stop(seconds,'person_proximity_stop',camera_id=rid,
                        image_person_position_m=person.tolist(),image_distance_m=distance,
                        stopping_radius_m=radius,own_speed_m_s=float(measured_speed),
                        commanded_speed_m_s=float(commanded_speed))
        return False


class PersonPauseGate(PersonStopGate):
    """Opt-in proximity pause; resume only after repeated fresh clear views.

    Unknown locations, lost/ambiguous people, invalid feedback and camera
    failures remain latched. Empty detections cannot clear a known person.
    Clearance is evaluated at the configured future cruise speed, so stopping
    cannot shrink the clearance radius and cause repeated stop/start chatter.
    """
    def __init__(self,config,camera_ids):
        super().__init__(config,camera_ids)
        self.clear_since=None;self.clear_frames=0;self.last_clear_stamp=None
        self.events=[];self.resume_count=0
        self.resume_speed=config.get('resume_speed_m_s',1.5)
        self.clear_duration=config.get('resume_clear_seconds',1.5)
        if not 0<self.resume_speed<=1.5 or self.clear_duration<1.:
            raise ValueError('Resume needs a positive design speed and sustained clearance.')

    def blocked(self,position,measured_speed,commanded_speed,seconds):
        if not self.latched:
            blocked=super().blocked(position,measured_speed,commanded_speed,seconds)
            if blocked:self.events.append(dict(self.event))
            return blocked
        if self.event['reason']!='person_proximity_stop':return True
        # Use the original gate to validate every frame and the full-speed radius.
        probe=PersonStopGate(self.config,self.camera_ids);probe.frames=self.frames
        probe.continuity=self.continuity
        clear=not probe.blocked(position,measured_speed,self.resume_speed,seconds)
        if not clear and probe.event['reason']!='person_proximity_stop':
            self.event=dict(probe.event);self.events.append(dict(self.event))
        if not clear or not math.isfinite(measured_speed) or abs(measured_speed)>.03:
            self.clear_since=None;self.clear_frames=0;self.last_clear_stamp=None
            return True
        stamp=min(self.frames[rid]['seconds'] for rid in self.camera_ids)
        if self.clear_since is None:self.clear_since=stamp
        if self.last_clear_stamp is None or stamp>self.last_clear_stamp:
            self.clear_frames+=1;self.last_clear_stamp=stamp
        if stamp-self.clear_since>=self.clear_duration and self.clear_frames>=4:
            self.latched=False;self.resume_count+=1
            self.events.append(dict(seconds=seconds,reason='fresh_clear_views_resume',
                                    clear_span_seconds=stamp-self.clear_since,clear_frames=self.clear_frames))
            self.clear_since=None;self.clear_frames=0;self.last_clear_stamp=None
            return False
        return True
