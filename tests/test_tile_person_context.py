"""Crop seams cannot create blind image strips or truncate a retained person."""
import unittest
from types import SimpleNamespace
import numpy as np
from fixed_detection import tile_starts,tile_owns_pixel,FixedBallDetector


class TilePersonContextTests(unittest.TestCase):
    def test_every_pixel_is_owned_even_at_final_irregular_overlap(self):
        size=512;overlap=.2;margin=(size-round(size*(1-overlap)))/2
        for length in (300,512,513,2160,3840):
            starts=tile_starts(length,size,overlap)
            tiles=[(x,0,min(length,x+size),300) for x in starts]
            for pixel in range(length):
                self.assertTrue(any(tile_owns_pixel(t,(pixel+.5,150),(length,300),margin) for t in tiles))

    def test_recorded_fragment_uses_the_neighbouring_complete_context(self):
        point=(1664.24,498.79)
        self.assertFalse(tile_owns_pixel((1640,0,2152,512),point,(3840,2160),51))
        self.assertTrue(tile_owns_pixel((1230,410,1742,922),point,(3840,2160),51))

    def test_real_image_edges_are_not_removed(self):
        self.assertTrue(tile_owns_pixel((0,0,512,512),(1,1),(3840,2160),51))
        self.assertTrue(tile_owns_pixel((3328,1648,3840,2160),(3839,2159),(3840,2160),51))

    def test_broader_person_context_preserves_small_ball_tile_resolution(self):
        calls=[]
        class Tensor:
            def __init__(self,array):self.array=np.asarray(array)
            def cpu(self):return self
            def numpy(self):return self.array
        class Model:
            def predict(self,images,**kw):
                calls.append((len(images),images[0].size,kw))
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(np.empty((0,4))),conf=Tensor([]),cls=Tensor([]))) for _ in images]
        detector=FixedBallDetector.__new__(FixedBallDetector)
        detector.model=Model();detector.classes=[32];detector.person_classes=[0]
        detector.person_model=detector.model
        detector.vision={'device':'cpu','confidence':.2,'image_size':640}
        detector.config={'tile_size_pixels':512,'tile_overlap_fraction':.2,'tile_inference_size_pixels':1280,
            'inference_batch_size':4,'box_merge_iou':.3,'person_confidence':.75,
            'person_tile_size_pixels':1024,'person_tile_inference_size_pixels':1280}
        calibration={'id':'A','intrinsics':np.eye(3).tolist(),'transform':np.eye(4).tolist()}
        result=detector.detect(np.zeros((2160,3840,3),dtype=np.uint8),calibration,.0335,tiled=True)
        balls=[c for c in calls if c[2]['classes']==[32]]
        people=[c for c in calls if c[2]['classes']==[0]]
        self.assertEqual(sum(c[0] for c in balls),60)
        self.assertEqual(sum(c[0] for c in people),15)
        self.assertTrue(all(c[1]==(512,512) and c[2]['imgsz']==1280 for c in balls))
        self.assertTrue(all(c[1]==(1024,1024) and c[2]['conf']==.75 for c in people))
        self.assertTrue(result['person_context_separate_from_balls'])
