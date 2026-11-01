"""Conservative optical clearance from own encoders and declared robot geometry.

The complete moving arm and bucket must be below the basket floor (plus a
5 mm margin). Thus none can lie between the overhead camera and basket balls.
No ball state, world pose, depth, rendering mask or detected count enters this
gate. The enclosing boxes and joint axes are checked against the URDF in tests.
"""
import math

FLOOR_TOP = .0525  # basket_floor: origin Z .04, thickness .025
MARGIN = .005


def basket_view_clearance(angles):
    if len(angles) != 2 or not all(math.isfinite(float(a)) for a in angles):
        return dict(clear=False, maximum_shovel_z_in_robot_m=None,
                    basket_floor_top_m=FLOOR_TOP, margin_m=MARGIN)
    lift, tilt = map(float, angles)
    if not -.6 <= lift <= 2.4 or not -3.2 <= tilt <= 1.0:
        return dict(clear=False, maximum_shovel_z_in_robot_m=None,
                    basket_floor_top_m=FLOOR_TOP, margin_m=MARGIN)
    # The joint axis is -Y, so a local +X point rises for positive lift.
    def highest(xrange, zrange, angle, origin_z):
        return max(origin_z+x*math.sin(angle)+z*math.cos(angle)
                   for x in xrange for z in zrange)
    arm = highest((0., .4), (-.0125, .0125), lift, -.005)
    shaft = highest((.39, .41), (-.01, .01), lift, -.005)
    bucket = highest((-.008, .25), (-.026, .065), lift+tilt,
                     -.005+.4*math.sin(lift))
    top = max(arm, shaft, bucket)
    return dict(clear=top <= FLOOR_TOP-MARGIN,
                maximum_shovel_z_in_robot_m=top,
                basket_floor_top_m=FLOOR_TOP, margin_m=MARGIN)
