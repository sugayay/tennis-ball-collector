"""Actual basket frame evaluation only; no motor inputs."""
import math

def nominal_basket_base_from_link(ball_world,basket_position,basket_rotation):
    import numpy as np
    return np.array([-.08,0.,.04])+basket_rotation.T@(np.asarray(ball_world)-basket_position)


def basket_link_fk(base_position,base_rotation,q):
    import numpy as np
    lift,slide,tip,gate=q
    c,s=math.cos(tip),math.sin(tip)
    relative=np.array([[c,0.,-s],[0.,1.,0.],[s,0.,c]])
    return (base_position+base_rotation@np.array([-.08-slide,0.,.04+lift]),
            base_rotation@relative)


def validate_basket_geometry(urdf):
    import xml.etree.ElementTree as ET
    tree=ET.parse(urdf).getroot()
    joints={j.get('name'):j for j in tree.findall('joint')}
    expected={'refill_lift_joint':('0 0 1',[-.08,0.,.04]),
              'refill_slide_joint':('-1 0 0',[0.,0.,0.]),
              'refill_tip_joint':('0 -1 0',[0.,0.,0.])}
    for name,(axis,origin) in expected.items():
        j=joints[name]
        assert [float(x) for x in j.find('axis').get('xyz').split()]==[float(x) for x in axis.split()]
        assert [float(x) for x in j.find('origin').get('xyz').split()]==origin
        assert all(float(x)==0. for x in j.find('origin').get('rpy','0 0 0').split())
    links={l.get('name'):l for l in tree.findall('link')}
    collisions={c.get('name'):c for c in links['refill_basket'].findall('collision')}
    for name,(size,origin) in {'basket_floor_collision':([.34,.30,.025],[0.,0.,0.]),
                      'basket_left_collision':([.34,.025,.15],[0.,.15,.07]),
                      'basket_right_collision':([.34,.025,.15],[0.,-.15,.07]),
                      'basket_front_collision':([.025,.32,.15],[.16,0.,.07])}.items():
        assert [float(x) for x in collisions[name].find('geometry/box').get('size').split()]==size
        assert [float(x) for x in collisions[name].find('origin').get('xyz').split()]==origin
    gate=joints['refill_gate_joint']
    assert [float(x) for x in gate.find('origin').get('xyz').split()]==[-.16,0.,.07]
    assert [float(x) for x in gate.find('axis').get('xyz').split()]==[0.,0.,1.]
    return True


BALL_CENTER_BOUNDS=((-0.194,.034),(-.104,.104),(.080,.1465))
WALL_TOLERANCE=.0005

def sphere_is_contained(local):
    if len(local) != 3 or not all(math.isfinite(value) for value in local):
        return False
    x, y, z = local
    return (BALL_CENTER_BOUNDS[0][0] - WALL_TOLERANCE <= x <= BALL_CENTER_BOUNDS[0][1] + WALL_TOLERANCE
            and BALL_CENTER_BOUNDS[1][0] - WALL_TOLERANCE <= y <= BALL_CENTER_BOUNDS[1][1] + WALL_TOLERANCE
            and BALL_CENTER_BOUNDS[2][0] <= z <= BALL_CENTER_BOUNDS[2][1] + WALL_TOLERANCE)

