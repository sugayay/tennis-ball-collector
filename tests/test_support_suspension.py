import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from support_suspension import JOINTS, CONFIG, settings, configure_if_present


class Robot:
    def __init__(self, names=JOINTS):
        self.joints = [SimpleNamespace(name=n, dofs_idx_local=[i + 12]) for i, n in enumerate(names)]
        self.calls = []
    def get_joint(self, name):
        return next(j for j in self.joints if j.name == name)
    def set_dofs_position(self, x, **kw): self.calls.append(('initial', x, kw))
    def set_dofs_kp(self, x, **kw): self.calls.append(('kp', x, kw))
    def set_dofs_kv(self, x, **kw): self.calls.append(('kv', x, kw))
    def set_dofs_force_range(self, a, b, **kw): self.calls.append(('force', a, b, kw))
    def control_dofs_position(self, x, **kw): self.calls.append(('reference', x, kw))


class SupportSuspensionTests(unittest.TestCase):
    def test_fixed_reference_and_physical_force_limit(self):
        r = Robot(); result = configure_if_present(r)
        self.assertEqual(result['dofs'], [12, 13])
        self.assertEqual([c[0] for c in r.calls], ['initial', 'kp', 'kv', 'force', 'reference'])
        for call, expected in zip(r.calls, ([0, 0], [2500, 2500], [120, 120], [-120, -120], [0., 0.])):
            np.testing.assert_array_equal(call[1], expected)
            self.assertEqual(call[-1]['dofs_idx_local'], [12, 13])
        np.testing.assert_array_equal(r.calls[3][2], [120, 120])

    def test_legacy_rigid_variant_is_untouched(self):
        r = Robot(('left_wheel_joint', 'right_wheel_joint'))
        self.assertIsNone(configure_if_present(r)); self.assertEqual(r.calls, [])

    def test_partial_suspension_rejected_before_actuation(self):
        r = Robot(JOINTS[:1])
        with self.assertRaises(ValueError): configure_if_present(r)
        self.assertEqual(r.calls, [])

    def test_duplicate_dof_rejected_before_actuation(self):
        r = Robot(); r.joints[1].dofs_idx_local = [12]
        with self.assertRaises(ValueError): configure_if_present(r)
        self.assertEqual(r.calls, [])

    def test_invalid_settings_never_mutate_robot(self):
        base = json.loads(CONFIG.read_text())
        bad = [('stiffness_N_m', 0), ('damping_Ns_m', -1), ('stiffness_N_m', True),
               ('force_limit_N', 120.001), ('rest_position_m', .025), ('stroke_m', [-.03, .03]),
               ('damping_Ns_m', float('nan')), ('rest_position_m', float('inf'))]
        for key, value in bad:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(base); config[key] = value; r = Robot()
                with self.assertRaises(ValueError): configure_if_present(r, config)
                self.assertEqual(r.calls, [])

    def test_authored_neutral_caster_frames_and_all_original_links_preserved(self):
        source = CONFIG.parent
        old = ET.parse(Path(__file__).resolve().parent/'fixtures/collector_rigid_v78.urdf').getroot()
        new = ET.parse(source/'assets/collector_basket_lift.urdf').getroot()
        # XML serialization whitespace differs; compare the full recursive elements.
        def signature(e):
            return e.tag, e.attrib, (e.text or '').strip(), tuple(signature(c) for c in e)
        new_links = {x.get('name'): x for x in new.findall('link')}
        for link in old.findall('link'):
            self.assertEqual(signature(link), signature(new_links[link.get('name')]))
        for side in ('front', 'rear'):
            before = old.find(f"joint[@name='{side}_caster_yaw']")
            yaw = new.find(f"joint[@name='{side}_caster_yaw']")
            spring = new.find(f"joint[@name='{side}_caster_suspension_joint']")
            self.assertEqual(spring.find('origin').attrib, before.find('origin').attrib)
            self.assertEqual(yaw.find('origin').get('xyz'), '0 0 0')
            self.assertEqual(spring.find('axis').get('xyz'), '0 0 1')
            self.assertEqual(spring.find('parent').get('link'), before.find('parent').get('link'))
            self.assertEqual(yaw.find('child').attrib, before.find('child').attrib)
            self.assertEqual(spring.find('child').get('link'), yaw.find('parent').get('link'))
            self.assertEqual(float(spring.find('limit').get('lower')), -.025)
            self.assertEqual(float(spring.find('limit').get('upper')), .025)

    def test_only_suspension_joints_change_and_added_mass_is_81_grams(self):
        source = CONFIG.parent
        old = ET.parse(Path(__file__).resolve().parent/'fixtures/collector_rigid_v78.urdf').getroot()
        new = ET.parse(source/'assets/collector_basket_lift.urdf').getroot()
        def signature(e): return e.tag, e.attrib, tuple(signature(c) for c in e)
        originals = {j.get('name'): j for j in old.findall('joint')}
        for joint in new.findall('joint'):
            name = joint.get('name')
            if name not in JOINTS and name not in ('front_caster_yaw', 'rear_caster_yaw'):
                self.assertEqual(signature(joint), signature(originals[name]))
        carriers = [new.find(f"link[@name='{side}_caster_suspension_carrier']") for side in ('front', 'rear')]
        self.assertAlmostEqual(sum(float(l.find('inertial/mass').get('value')) for l in carriers), .081)
        self.assertTrue(all(not l.findall('collision') for l in carriers))


if __name__ == '__main__': unittest.main()
