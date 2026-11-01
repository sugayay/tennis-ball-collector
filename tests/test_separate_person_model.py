"""Person inference must cover the whole image despite scoped ball search."""
import unittest
from types import SimpleNamespace
import numpy as np
from fixed_detection import FixedBallDetector

class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def cpu(self):return self
    def numpy(self):return self.value

class RecordingModel:
    def __init__(self,category,boxes):self.category=category;self.boxes=boxes;self.calls=[]
    def predict(self,images,**kwargs):
        self.calls.append((len(images),[im.size for im in images],kwargs))
        return [
            SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(np.array(self.boxes).reshape(-1,4)),
                conf=Tensor([.9]*len(self.boxes)),cls=Tensor([self.category]*len(self.boxes)))) for _ in images]

class SeparatePersonTests(unittest.TestCase):
    def test_person_only_worker_does_not_run_any_ball_search(self):
        d,cal=self.detector();d.person_model=d.model;d.person_classes=[0]
        result=d.detect(np.zeros((2160,3840,3),dtype=np.uint8),cal,.0335,tiled=True,include_balls=False)
        self.assertEqual(result['tile_count'],0);self.assertEqual(result['person_tile_count'],60)
        self.assertTrue(all(c[2]['classes']==[0] for c in d.model.calls))

    def test_ball_only_worker_cannot_publish_safety_results(self):
        d,cal=self.detector()
        result=d.detect(np.zeros((2160,3840,3),dtype=np.uint8),cal,.0335,tiled=True,include_people=False)
        self.assertEqual(d.person_model.calls,[])
        self.assertEqual(result['person_tile_count'],0);self.assertEqual(result['persons'],[])

    def detector(self):
        d=FixedBallDetector.__new__(FixedBallDetector)
        d.model=RecordingModel(32,[]);d.person_model=RecordingModel(7,[])
        d.classes=[32];d.person_classes=[7]
        d.vision=dict(device='cpu',confidence=.2,image_size=640)
        d.config=dict(tile_size_pixels=512,tile_overlap_fraction=.2,tile_inference_size_pixels=1280,
            inference_batch_size=4,box_merge_iou=.3,person_confidence=.75,person_tile_size_pixels=512,
            person_tile_inference_size_pixels=1280,ball_search_zones_m=[[-1,-1,1,1]])
        k=np.array([[100,0,1920],[0,100,1080],[0,0,1]])
        t=np.eye(4);t[2,3]=10
        return d,dict(id='A',intrinsics=k.tolist(),transform=t.tolist())

    def test_ball_roi_does_not_remove_person_tiles_even_at_same_tile_size(self):
        d,cal=self.detector();result=d.detect(np.zeros((2160,3840,3),dtype=np.uint8),cal,.0335,tiled=True)
        self.assertLess(result['tile_count'],60);self.assertEqual(result['person_tile_count'],60)
        self.assertEqual(sum(c[0] for c in d.person_model.calls),60)
        self.assertTrue(all(c[2]['classes']==[32] for c in d.model.calls))
        self.assertTrue(all(c[2]['classes']==[7] and c[2]['conf']==.75 for c in d.person_model.calls))

    def test_whole_frame_uses_person_model_class_mapping_and_localizes_feet(self):
        d,cal=self.detector();d.person_model.boxes=[[1900,1000,1940,1080]]
        result=d.detect(np.zeros((2160,3840,3),dtype=np.uint8),cal,.0335,tiled=False)
        self.assertEqual(result['person_tile_count'],1)
        self.assertEqual(result['observations'],[])
        self.assertEqual(len(result['persons']),1)
        np.testing.assert_allclose(result['persons'][0]['position_m'],[0,0,0])
        self.assertEqual(d.person_model.calls[0][1],[(3840,2160)])

if __name__=='__main__':unittest.main()
