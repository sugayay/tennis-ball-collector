from pathlib import Path
import collections,math,unittest,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

class WheelMeshTests(unittest.TestCase):
    def setUp(self):
        lines=(ROOT/'assets/wheel_collision_128.obj').read_text().splitlines()
        self.verts=np.array([[float(v) for v in line.split()[1:]] for line in lines if line.startswith('v ')])
        self.faces=np.array([[int(v)-1 for v in line.split()[1:]] for line in lines if line.startswith('f ')])
    def test_closed_cylinder_has_declared_dimensions_and_outward_faces(self):
        np.testing.assert_allclose(np.ptp(self.verts,axis=0),[.22,.22,.065],atol=1e-10)
        edges=collections.Counter()
        volume=0.
        for face in self.faces:
            a,b,c=self.verts[face]
            normal=np.cross(b-a,c-a)
            self.assertGreater(float(normal@((a+b+c)/3)),0.)
            volume+=float(a@np.cross(b,c))/6
            for i,j in zip(face,np.roll(face,-1)):edges[tuple(sorted((i,j)))]+=1
        self.assertTrue(all(count==2 for count in edges.values()))
        exact=math.pi*.11**2*.065
        self.assertLess(abs(volume/exact-1),.0005)
    def test_collision_mesh_keeps_visual_cylinders_and_mass(self):
        root=ET.parse(ROOT/'assets/collector_beveled.urdf').getroot()
        for name in ('left_wheel','right_wheel'):
            link=root.find(f"link[@name='{name}']")
            self.assertEqual(link.find('collision/geometry/mesh').get('filename'),'wheel_collision_128.obj')
            self.assertIsNotNone(link.find('visual/geometry/cylinder'))
            self.assertEqual(float(link.find('inertial/mass').get('value')),.35)
    def test_arm_wheel_gap_and_radial_error_bound(self):
        self.assertAlmostEqual(.285-.01-(.225+.065/2),.0175)
        self.assertLess(.11*(1-math.cos(math.pi/128)),.000034)

if __name__=='__main__':unittest.main()
