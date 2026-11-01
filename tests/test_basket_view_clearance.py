import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from basket_view_clearance import basket_view_clearance


class BasketViewClearanceTests(unittest.TestCase):
    def test_ground_supported_shovel_remains_visibly_below_basket(self):
        for angles in [(-.448,.448),(-.39,.448),(-.35,.40),(-.48,.50),(-.30,.45)]:
            self.assertTrue(basket_view_clearance(angles)['clear'],angles)
        self.assertGreater(abs(-.39-(-.448)),.04)

    def test_raised_transfer_and_tipped_poses_are_not_usable(self):
        for lift,absolute in [(math.radians(125),math.radians(15)),
                              (math.radians(125),math.radians(-50)),(1.,0.),(.5,0.)]:
            self.assertFalse(basket_view_clearance((lift,absolute-lift))['clear'])

    def test_invalid_encoder_values_fail_closed(self):
        for angles in [[],[0.],[0.,0.,0.],[math.nan,0.],[0.,math.inf],[10.,0.],[0.,-10.]]:
            self.assertFalse(basket_view_clearance(angles)['clear'])

    def test_flat_shovel_at_floor_height_is_not_blindly_accepted(self):
        self.assertFalse(basket_view_clearance((0.,0.))['clear'])

    def test_geometry_matches_model_joint_origins_axes_and_enclosing_box(self):
        assets=Path(__file__).resolve().parents[1]/'assets'
        model=ET.parse(assets/'collector_beveled.urdf').getroot()
        for name,origin in [('shovel_lift_joint','-0.06 0 -0.005'),('shovel_tilt_joint','0.40 0 0')]:
            joint=model.find(f"joint[@name='{name}']")
            self.assertEqual(joint.find('origin').attrib['xyz'],origin)
            self.assertEqual(joint.find('axis').attrib['xyz'],'0 -1 0')
        floor=model.find("link[@name='base_link']/visual[@name='basket_floor']")
        self.assertEqual(floor.find('origin').attrib['xyz'],'-0.08 0 0.04')
        self.assertEqual(floor.find('geometry/box').attrib['size'],'0.34 0.30 0.025')
        # Every modeled arm/bucket visual lies inside the analytic envelopes.
        for link in ['shovel_arm','shovel_bucket']:
            for visual in model.find(f"link[@name='{link}']").findall('visual'):
                origin=[float(v) for v in visual.find('origin').attrib.get('xyz','0 0 0').split()]
                self.assertNotIn('rpy',visual.find('origin').attrib)
                box=visual.find('geometry/box')
                if box is None:
                    vertices=[[float(v) for v in l.split()[1:]] for l in (assets/'collector_beveled_floor.obj').read_text().splitlines() if l.startswith('v ')]
                    self.assertTrue(all(-.008<=v[0]<=.25 and -.026<=v[2]<=.065 for v in vertices))
                    continue
                size=[float(v) for v in box.attrib['size'].split()]
                xrange=(origin[0]-size[0]/2,origin[0]+size[0]/2)
                zrange=(origin[2]-size[2]/2,origin[2]+size[2]/2)
                bounds= ((.39,.41),(-.01,.01)) if visual.attrib['name']=='bucket_pivot_shaft' else ((0.,.4),(-.0125,.0125)) if link=='shovel_arm' else ((-.008,.25),(-.026,.065))
                for actual,expected in zip([xrange,zrange],bounds):
                    self.assertGreaterEqual(actual[0],expected[0]-1e-9)
                    self.assertLessEqual(actual[1],expected[1]+1e-9)

    def test_lift_motion_cannot_keep_old_clearance(self):
        self.assertTrue(basket_view_clearance((-.448,.448))['clear'])
        self.assertFalse(basket_view_clearance((.8,-.8))['clear'])
        self.assertTrue(basket_view_clearance((-.448,.448))['clear'])
