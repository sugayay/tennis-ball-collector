"""Fresh RGB seating gate before lifting; no simulator entities or ball truth."""
import math
import numpy as np


class ImageSeatingGate:
    # Declared front lip is approximately0.55m from the chassis origin.
    # Keep the image-estimated centre20mm behind it and inside the side walls.
    minimum_x = .39
    maximum_x = .530
    # After a strict in-region frame, tolerate up to5mm of longitudinal
    # observation variation when counting the next frame. The lift decision
    # still requires the latest estimate inside the original strict region.
    maximum_keep_x = .535
    maximum_y = .070
    maximum_age = .15

    def __init__(self):
        self.seconds = -math.inf
        self.valid_seconds = -math.inf
        self.matched_current_frame = False
        self.point_world = None
        self.frames = 0

    def seed_motion(self, point_world, captured_seconds, now):
        """Carry valid visual motion evidence; never count it as a seating frame."""
        if point_world is None:return
        point=np.asarray(point_world,dtype=float)
        if (point.shape!=(3,) or not np.isfinite(point).all()
                or not math.isfinite(captured_seconds) or not math.isfinite(now)
                or not 0<=now-captured_seconds<=self.maximum_age):return
        self.point_world=point.copy()
        self.seconds=self.valid_seconds=captured_seconds
        self.frames=0
        self.matched_current_frame=False

    def inside(self, local):
        return (np.isfinite(local).all() and self.minimum_x<=local[0]<=self.maximum_x
                and abs(local[1])<=self.maximum_y)

    def observe(self, observations, position, rotation, target, seconds, not_before):
        if not math.isfinite(seconds) or seconds<not_before or seconds<=self.seconds:
            return
        if seconds-self.seconds>self.maximum_age:self.frames=0
        self.seconds=seconds
        # A missed image breaks lift confirmation, but does not refresh or erase
        # the last valid motion estimate inside its existing150ms lifetime.
        self.matched_current_frame=False
        if target is not None:
            for item in observations:
                local=np.asarray(item.position_in_robot_m,dtype=float)
                if local.shape!=(3,) or not np.isfinite(local).all():continue
                world=position+rotation@local
                if np.linalg.norm(world[:2]-target[:2])<=.25:
                    self.point_world=world
                    self.valid_seconds=seconds
                    self.matched_current_frame=True
                    retained=(self.frames>0 and self.minimum_x<=local[0]<=self.maximum_keep_x
                              and abs(local[1])<=self.maximum_y)
                    self.frames=self.frames+1 if self.inside(local) or retained else 0
                    break
        if not self.matched_current_frame:self.frames=0
        if target is None:self.point_world=None

    def fresh(self, seconds):
        return self.point_world is not None and 0<=seconds-self.valid_seconds<=self.maximum_age

    def ready(self, position, rotation, seconds):
        return (self.frames>=2 and self.fresh(seconds)
                and self.inside(rotation.T@(self.point_world-position)))
