"""Protect the tapered floor and the explicit 2 mm rear support trial."""
from collections import Counter
import copy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def structure(element):
    return (element.tag, element.attrib, (element.text or '').strip(),
            [structure(child) for child in element])


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return sum(x*y for x,y in zip(a,b))


class BeveledGeometryTests(unittest.TestCase):
    def test_only_declared_floor_support_speed_and_wheel_representation_changes(self):
        stock = ET.parse(ASSETS/'collector.urdf').getroot()
        trial = ET.parse(ASSETS/'collector_beveled.urdf').getroot()
        trial.attrib['name'] = stock.attrib['name']
        for wheel in ('left_wheel','right_wheel'):
            old=stock.find(f"link[@name='{wheel}']/collision/geometry")
            new=trial.find(f"link[@name='{wheel}']/collision/geometry")
            self.assertEqual(new.find('mesh').attrib,{'filename':'wheel_collision_128.obj','scale':'1 1 1'})
            new.remove(new.find('mesh'));new.append(copy.deepcopy(old.find('cylinder')))
        for joint in ('left_wheel_joint', 'right_wheel_joint'):
            old = stock.find(f"joint[@name='{joint}']/limit")
            new = trial.find(f"joint[@name='{joint}']/limit")
            self.assertEqual(old.attrib, {'effort':'12', 'velocity':'12'})
            self.assertEqual(new.attrib, {'effort':'12', 'velocity':'14'})
            new.attrib['velocity'] = old.attrib['velocity']
        rear = trial.find("joint[@name='rear_caster_yaw']/origin")
        original = stock.find("joint[@name='rear_caster_yaw']/origin")
        self.assertEqual(rear.attrib['xyz'], '-0.21 0 -0.141')
        self.assertEqual(original.attrib['xyz'], '-0.21 0 -0.143')
        rear.attrib['xyz'] = original.attrib['xyz']
        old_bucket = stock.find("link[@name='shovel_bucket']")
        new_bucket = trial.find("link[@name='shovel_bucket']")
        for tag, name in (('visual','shovel_floor'), ('collision','shovel_floor_collision')):
            old = old_bucket.find(f"{tag}[@name='{name}']")
            new = new_bucket.find(f"{tag}[@name='{name}']")
            self.assertEqual(new.find('origin').attrib['xyz'], '0 0 0')
            self.assertEqual(new.find('geometry/mesh').attrib,
                             {'filename':'collector_beveled_floor.obj','scale':'1 1 1'})
            index = list(new_bucket).index(new)
            new_bucket.remove(new)
            new_bucket.insert(index, copy.deepcopy(old))
        self.assertEqual(structure(stock), structure(trial))

    def test_floor_is_closed_convex_and_thinner_without_extending_its_reach(self):
        vertices, faces = [], []
        for line in (ASSETS/'collector_beveled_floor.obj').read_text(encoding='utf-8').splitlines():
            fields = line.split()
            if fields[0]=='v': vertices.append(tuple(map(float,fields[1:])))
            if fields[0]=='f': faces.append(tuple(int(v)-1 for v in fields[1:]))
        self.assertEqual(len(vertices),10)
        self.assertEqual(len(faces),16)
        self.assertEqual([(min(v[i] for v in vertices),max(v[i] for v in vertices)) for i in range(3)],
                         [(0.,.25),(-.12,.12),(-.026,-.024)])
        for x, thickness in ((0.,.002),(.20,.002),(.25,.00025)):
            heights = [v[2] for v in vertices if v[0]==x]
            self.assertAlmostEqual(max(heights)-min(v[2] for v in vertices),thickness)
        edges = Counter()
        volume = 0.
        for indices in faces:
            self.assertEqual(len(indices),3)
            a,b,c = [vertices[i] for i in indices]
            normal = cross(tuple(y-x for x,y in zip(a,b)),tuple(y-x for x,y in zip(a,c)))
            self.assertGreater(dot(normal,normal),0.)
            for p in vertices:
                self.assertLessEqual(dot(normal,tuple(y-x for x,y in zip(a,p))),1e-12)
            volume += dot(a,cross(b,c))/6
            for i,j in zip(indices,indices[1:]+indices[:1]): edges[(i,j)] += 1
        self.assertTrue(all(n==edges[(j,i)]==1 for (i,j),n in edges.items()))
        self.assertAlmostEqual(volume,.0001095,places=12)


if __name__=='__main__':
    unittest.main()
