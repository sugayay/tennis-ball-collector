"""Image-coordinate handoff contracts, independent of simulator target state."""
import unittest

import numpy as np

from camera_geometry import merge_ball_estimates, pixel_to_plane, world_to_pixel


class CameraGeometryTests(unittest.TestCase):
    def test_known_pixel_offset_maps_to_floor_distance(self):
        # Camera at 10 m, looking straight down, focal length 1000 pixels.
        intrinsic = np.array([[1000.,0,1000],[0,1000,600],[0,0,1]])
        transform = np.eye(4)
        transform[2,3] = 10
        point = pixel_to_plane([1100,400],intrinsic,transform,0)
        np.testing.assert_allclose(point,[1,2,0])
        np.testing.assert_allclose(world_to_pixel(point,intrinsic,transform),[1100,400])

    def test_upward_ray_does_not_create_a_floor_target(self):
        transform = np.diag([1.,-1.,-1.,1.])
        transform[2,3] = 10
        self.assertIsNone(pixel_to_plane([0,0],np.eye(3),transform,0))

    def test_no_detections_means_no_targets(self):
        self.assertEqual(merge_ball_estimates([],0.18),[])

    def test_overlapping_cameras_merge_one_ball_and_preserve_another(self):
        observations = [
            {'position_m':[1.,2.,0.0335],'confidence':.9,'camera_id':'A'},
            {'position_m':[1.05,2.03,0.0335],'confidence':.7,'camera_id':'B'},
            {'position_m':[1.5,2.,0.0335],'confidence':.8,'camera_id':'B'},
        ]
        merged = merge_ball_estimates(observations,0.18)
        self.assertEqual(len(merged),2)
        self.assertEqual(merged[0]['sources'],['A','B'])
        self.assertEqual(merged[1]['sources'],['B'])


if __name__ == '__main__':
    unittest.main()
