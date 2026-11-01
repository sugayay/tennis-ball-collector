"""Contracts for full-image coverage, duplicate detections and honest evaluation."""
import unittest

from fixed_camera_demo import evaluate
from fixed_detection import suppress_overlapping_boxes,tile_starts


class FixedDetectionTests(unittest.TestCase):
    def test_uniform_tiles_include_edges_without_gaps(self):
        for length in (3840,2160,400):
            starts = tile_starts(length,512,.2)
            self.assertEqual(starts[0],0)
            self.assertGreaterEqual(starts[-1]+512,length)
            self.assertTrue(all(b-a <= 512 for a,b in zip(starts,starts[1:])))

    def test_tile_duplicates_merge_without_removing_distinct_ball(self):
        items = [{'bbox_xyxy':[10,10,20,20],'confidence':.7},
                 {'bbox_xyxy':[11,10,21,20],'confidence':.9},
                 {'bbox_xyxy':[40,10,50,20],'confidence':.8}]
        result = suppress_overlapping_boxes(items,.3)
        self.assertEqual(len(result),2)
        self.assertEqual([item['confidence'] for item in result],[.9,.8])

    def test_one_detection_cannot_count_as_two_reference_balls(self):
        result = evaluate([{'position_m':[.01,0,0]}],[[0,0,0],[.02,0,0]],.3)
        self.assertEqual(result['matched_balls'],1)
        self.assertEqual(len(result['missed_ball_indices']),1)

    def test_missed_balls_and_false_positives_are_reported(self):
        result = evaluate([{'position_m':[3,3,0]}],[[0,0,0]],.3)
        self.assertEqual(result['matched_balls'],0)
        self.assertEqual(result['missed_ball_indices'],[0])
        self.assertEqual(result['false_positive_indices'],[0])


if __name__ == '__main__':
    unittest.main()
