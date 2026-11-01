"""Capacity and home-return policies from basket RGB and own pose only."""
import math
import numpy as np


class HomeReturn:
    def __init__(self,home):
        self.home=np.asarray(home,dtype=float)
        if self.home.shape!=(3,) or not np.isfinite(self.home).all():raise ValueError('Invalid home pose')
        self.settled_since=None

    def command(self,position,rotation,measured_speed,seconds):
        local=rotation.T@(self.home-position);distance=float(np.linalg.norm(local[:2]))
        if distance<.12:
            if abs(measured_speed)<.03:
                if self.settled_since is None:self.settled_since=seconds
                return np.zeros(2),seconds-self.settled_since>=.5
            self.settled_since=None
            return np.zeros(2),False
        self.settled_since=None
        bearing=math.atan2(local[1],local[0]);angular=float(np.clip(2.4*bearing,-.5,.5))
        # Conservative loaded return until high-speed loaded runs are verified.
        forward=min(.22,.4*distance) if abs(bearing)<.12 else 0.
        return np.array([forward-angular*.45/2,forward+angular*.45/2])/.11,False


def validate_service(config):
    capacity=config.get('basket_capacity',2)
    if type(capacity) is not int or not 1<=capacity<=6:raise ValueError('Prototype basket capacity must be 1..6')
    behavior=config.get('full_basket_behavior','stop')
    if behavior not in ('stop','return_home'):raise ValueError('Unsupported full basket behavior')
    if behavior=='return_home' and not config.get('home_positions_m'):raise ValueError('Home positions required')
    return capacity,behavior
