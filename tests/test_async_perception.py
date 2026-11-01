import threading,unittest
import numpy as np
from async_perception import freeze_camera,InferenceWorkers
from types import SimpleNamespace

class AsyncPerceptionTests(unittest.TestCase):
    def test_warmup_uses_runtime_thread_and_cannot_block_after_motion_starts(self):
        worker=InferenceWorkers(('person',))
        try:
            warmed=worker.warm('person',threading.get_ident)
            worker.submit('person',.1,threading.get_ident)
            thread_id,_=worker.pending['person'][1].result(timeout=3)
            self.assertEqual(thread_id,warmed)
            with self.assertRaises(RuntimeError):worker.warm('person',lambda:None)
        finally:worker.close()

    def test_slow_ball_search_does_not_hold_completed_safety_result(self):
        release=threading.Event();started=threading.Event();worker=InferenceWorkers(('person','fixed'))
        def slow_ball():started.set();release.wait(3);return 'balls'
        try:
            worker.submit('fixed',0.,slow_ball);self.assertTrue(started.wait(1))
            worker.submit('person',.1,lambda:'person frame')
            worker.pending['person'][1].result(timeout=3)
            self.assertEqual(worker.poll(.2),[('person',.1,'person frame')])
            self.assertFalse(worker.idle('fixed'))
            self.assertAlmostEqual(worker.records[0]['simulation_age_seconds'],.1)
        finally:release.set();worker.close()

    def test_capture_calibration_does_not_follow_later_robot_motion(self):
        camera=SimpleNamespace(intrinsics=np.array([[100.,0,100],[0,100,100],[0,0,1]]),transform=np.eye(4))
        camera.transform[2,3]=10
        p=np.array([0.,0.,0.]);r=np.eye(3)
        frozen=freeze_camera(camera,p,r)
        p[0]=4;camera.transform[0,3]=4
        np.testing.assert_allclose(frozen.point_on_basket_plane(100,100),[0,0,.086])
        with self.assertRaises(ValueError):frozen.position[0]=5

    def test_busy_inference_neither_blocks_physics_loop_nor_accumulates_jobs(self):
        release=threading.Event();started=threading.Event();worker=InferenceWorkers(('fixed',))
        def infer():started.set();release.wait(3);return 42
        try:
            self.assertTrue(worker.submit('fixed',1.,infer));self.assertTrue(started.wait(1))
            self.assertFalse(worker.submit('fixed',2.,lambda:99))
            self.assertEqual(worker.poll(2.),[])
            release.set();worker.pending['fixed'][1].result(timeout=3)
            self.assertEqual(worker.poll(2.5),[('fixed',1.,42)])
            self.assertEqual(worker.records[0]['simulation_age_seconds'],1.5)
            self.assertTrue(worker.idle('fixed'))
        finally:release.set();worker.close()

    def test_worker_failure_propagates_instead_of_reusing_old_success(self):
        worker=InferenceWorkers(('fixed',))
        def bad():raise RuntimeError('inference failed')
        try:
            worker.submit('fixed',0.,bad)
            with self.assertRaises(RuntimeError):worker.pending['fixed'][1].result(timeout=3)
            with self.assertRaises(RuntimeError):worker.poll(.1)
        finally:worker.close()
