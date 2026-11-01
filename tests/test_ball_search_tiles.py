import unittest
import numpy as np
from fixed_detection import tile_starts,ball_search_tiles
from camera_geometry import world_to_pixel

class BallSearchTilesTests(unittest.TestCase):
    def test_perspective_zone_keeps_edges_and_interior_without_empty_bbox_corners(self):
        eye=np.array([-19.,0.,10.]);look=np.array([-8.,0.,.0335])
        back=eye-look;back/=np.linalg.norm(back)
        right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right)
        up=np.cross(back,right)
        t=np.eye(4);t[:3,:3]=np.column_stack((right,up,back));t[:3,3]=eye
        focal=2160/(2*np.tan(np.radians(82)/2))
        k=np.array([[focal,0,1920],[0,focal,1080],[0,0,1]])
        tiles=[(x,y,min(3840,x+512),min(2160,y+512)) for y in tile_starts(2160,512,.2) for x in tile_starts(3840,512,.2)]
        zones=[[-17.365,-8.535,-11.885,8.535],[11.885,-8.535,17.365,8.535]]
        kept=ball_search_tiles(tiles,zones,dict(intrinsics=k,transform=t),.0335)
        bbox=[]
        for x1,y1,x2,y2 in zones:
            corners=np.array([world_to_pixel([x,y,.0335],k,t) for x in (x1-.75,x2+.75) for y in (y1-.75,y2+.75)])
            low,high=corners.min(axis=0)-20,corners.max(axis=0)+20
            bbox.extend(tile for tile in tiles if tile[0]<=high[0] and tile[2]>=low[0] and tile[1]<=high[1] and tile[3]>=low[1])
            for x in np.linspace(x1-.75,x2+.75,31):
                for y in np.linspace(y1-.75,y2+.75,41):
                    u,v=world_to_pixel([x,y,.0335],k,t)
                    if 0<=u<3840 and 0<=v<2160:
                        self.assertTrue(any(a<=u<=c and b<=v<=d for a,b,c,d in kept))
        self.assertLess(len(kept),len(set(bbox)))

    def test_every_point_in_configured_zone_remains_covered(self):
        k=np.array([[100,0,500],[0,100,500],[0,0,1]])
        t=np.eye(4);t[2,3]=10
        calibration=dict(intrinsics=k.tolist(),transform=t.tolist())
        tiles=[(x,y,x+128,y+128) for y in tile_starts(1000,128,.2) for x in tile_starts(1000,128,.2)]
        kept=ball_search_tiles(tiles,[[-3,-2,4,3]],calibration,.0335)
        self.assertLess(len(kept),len(tiles))
        for x in np.linspace(-3.75,4.75,19):
            for y in np.linspace(-2.75,3.75,17):
                u,v=world_to_pixel([x,y,.0335],k,t)
                self.assertTrue(any(a<=u<=c and b<=v<=d for a,b,c,d in kept))

    def test_zone_crossing_camera_plane_falls_back_to_all_tiles(self):
        tiles=[(0,0,100,100)]
        self.assertEqual(ball_search_tiles(tiles,[[0,0,1,1]],dict(intrinsics=np.eye(3),transform=np.eye(4)),.0335),tiles)
