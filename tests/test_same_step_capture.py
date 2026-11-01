from concurrent.futures import ThreadPoolExecutor
import time,unittest
import numpy as np
from async_perception import SameStepCapture,InferenceWorkers


class SameStepCaptureTests(unittest.TestCase):
    def test_same_step_shares_one_immutable_pair_and_next_step_recaptures(self):
        cache=SameStepCapture();rgb=np.ones((2,3,3),dtype=np.uint8);calls=[]
        def capture():calls.append(1);return [(rgb,{'id':'A'}),(rgb,{'id':'B'})]
        first=cache.get(4,capture);stamp=cache.started_wall
        self.assertIs(cache.get(4,capture),first)
        self.assertEqual(cache.started_wall,stamp);self.assertEqual(len(calls),1)
        rgb[:]=2
        self.assertTrue(np.all(first[0][0]==1))
        with self.assertRaises(ValueError):first[0][0][:]=3
        second=cache.get(5,capture)
        self.assertTrue(np.all(second[0][0]==2));self.assertEqual(len(calls),2)
        self.assertEqual([r['reused'] for r in cache.records],[False,True,False])

    def test_failed_second_camera_does_not_refresh_previous_pair(self):
        cache=SameStepCapture();rgb=np.zeros((2,3,3),dtype=np.uint8)
        previous=cache.get(4,lambda:[(rgb,{'id':'A'}),(rgb,{'id':'B'})]);stamp=cache.started_wall
        def broken():
            yield rgb,{'id':'A'}
            raise RuntimeError('camera B lost')
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError,'camera B lost'):cache.get(5,broken)
            self.assertEqual(cache.step,4);self.assertEqual(cache.started_wall,stamp)
        self.assertIs(cache.views,previous)
        with self.assertRaises(ValueError):cache.get(3,lambda:[])

    def test_workers_cannot_capture_from_or_mutate_the_scene(self):
        cache=SameStepCapture()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaisesRegex(RuntimeError,'scene thread'):pool.submit(cache.get,0,lambda:[]).result(2)

    def test_shared_image_uses_original_wall_time_and_preserves_simulation_age(self):
        workers=InferenceWorkers(('person',));cache=SameStepCapture()
        try:
            views=cache.get(12,lambda:[(np.zeros((2,3,3),dtype=np.uint8),{})])
            workers.submit('person',.1,lambda value:len(value),views,capture_started_wall=cache.started_wall)
            workers.pending['person'][1].result(2)
            self.assertEqual(workers.poll(.2),[('person',.1,1)])
            row=workers.records[0]
            self.assertAlmostEqual(row['simulation_age_seconds'],.1)
            fields=['capture_wall_seconds','dispatch_wait_wall_seconds','inference_wall_seconds','delivery_wait_wall_seconds']
            self.assertTrue(all(row[k]>=0 for k in fields))
            self.assertAlmostEqual(sum(row[k] for k in fields),row['capture_to_poll_wall_seconds'])
        finally:workers.close()
