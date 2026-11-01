"""Bounded inference workers and immutable capture-time camera calibration.

Workers receive RGB and calibration only. All Genesis access remains on the
simulation thread. No physics step waits for a running inference future.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
import threading
import numpy as np


@dataclass(frozen=True)
class FrozenCamera:
    intrinsics: np.ndarray
    transform: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    basket_plane: float = .086

    def ray(self,u,v):
        k=self.intrinsics
        ray=np.array([(u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-1.])
        return self.transform[:3,3],self.transform[:3,:3]@ray

    def point_on_ball_center_plane(self,u,v,radius):
        origin,direction=self.ray(u,v)
        if direction[2]>=-1e-5:return None
        distance=(radius-origin[2])/direction[2]
        if distance<=0:return None
        local=self.rotation.T@(origin+distance*direction-self.position)
        return local if .20<local[0]<8 else None

    def point_on_basket_plane(self,u,v):
        origin,direction=self.ray(u,v)
        origin=self.rotation.T@(origin-self.position);direction=self.rotation.T@direction
        if direction[2]>=-1e-8:return None
        distance=(self.basket_plane-origin[2])/direction[2]
        return (origin+distance*direction).tolist() if distance>0 else None


def freeze_camera(camera,position,rotation,basket_plane=.086):
    arrays=[np.array(x,dtype=float,copy=True) for x in (camera.intrinsics,camera.transform,position,rotation)]
    for array in arrays:array.flags.writeable=False
    return FrozenCamera(*arrays,basket_plane)


class SameStepCapture:
    """Share immutable RGB only within the exact same physical scene step.

    Used on the scene thread, never by workers. There is no cross-step cache,
    observation rescheduling or invented fresh timestamp for an older image.
    """
    def __init__(self):
        self.owner=threading.get_ident()
        self.step=None;self.views=None;self.started_wall=None;self.records=[]

    def get(self,step,capture):
        if threading.get_ident()!=self.owner:
            raise RuntimeError('Camera capture belongs to the scene thread')
        if self.step is not None and step<self.step:
            raise ValueError('Scene steps must not move backwards')
        if self.step==step:
            self.records.append(dict(step=step,reused=True,capture_wall_seconds=0.))
            return self.views
        started=time.perf_counter()
        # Commit only a complete pair. A camera exception must not publish a
        # partial image set or relabel a previous pair as the new scene step.
        views=tuple((np.array(rgb,dtype=np.uint8,copy=True),calibration) for rgb,calibration in capture())
        for rgb,_ in views:rgb.flags.writeable=False
        self.step=step;self.views=views;self.started_wall=started
        self.records.append(dict(step=step,reused=False,capture_wall_seconds=time.perf_counter()-started))
        return views


class InferenceWorkers:
    def __init__(self,channels=('fixed','onboard','basket')):
        self.pools={key:ThreadPoolExecutor(max_workers=1,thread_name_prefix=key) for key in channels}
        self.pending={};self.records=[];self.started=False
        self.wall_timing={}

    def warm(self,key,function,*args):
        """Initialize on its eventual worker thread before physics starts."""
        if self.started:raise RuntimeError('Blocking warmup is forbidden after runtime inference starts')
        return self.pools[key].submit(function,*args).result()

    def idle(self,key):return key not in self.pending

    def submit(self,key,captured_seconds,function,*args,capture_started_wall=None):
        self.started=True
        if not self.idle(key):return False
        submitted=time.perf_counter()
        timing=dict(capture_started=submitted if capture_started_wall is None else capture_started_wall,
                    submitted=submitted)
        def execute():
            start=time.perf_counter();timing['started']=start
            value=function(*args);finished=time.perf_counter();timing['finished']=finished
            return value,finished-start
        self.wall_timing[key]=timing
        self.pending[key]=(float(captured_seconds),self.pools[key].submit(execute))
        return True

    def poll(self,seconds):
        ready=[]
        for key,(captured,future) in list(self.pending.items()):
            if not future.done():continue
            del self.pending[key]
            timing=self.wall_timing.pop(key)
            value,wall=future.result()
            applied_wall=time.perf_counter()
            row=dict(channel=key,captured_seconds=captured,applied_seconds=float(seconds),
                     simulation_age_seconds=float(seconds-captured),inference_wall_seconds=wall,
                     capture_wall_seconds=timing['submitted']-timing['capture_started'],
                     dispatch_wait_wall_seconds=timing['started']-timing['submitted'],
                     delivery_wait_wall_seconds=applied_wall-timing['finished'],
                     capture_to_poll_wall_seconds=applied_wall-timing['capture_started'])
            self.records.append(row);ready.append((key,captured,value))
        return ready

    def close(self):
        for pool in self.pools.values():pool.shutdown(wait=True,cancel_futures=True)
