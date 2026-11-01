import json,unittest
from pathlib import Path
import numpy as np
from mission_control import FleetCoordinator
ROOT=Path(__file__).resolve().parents[1]

class DeliveryTimestampTests(unittest.TestCase):
    def test_late_frame_keeps_capture_freshness_but_transition_occurs_at_delivery(self):
        cfg=json.loads((ROOT/'mission_straight_config.json').read_text())
        f=FleetCoordinator(cfg,[23.77,10.97]);pos={'A':np.array([-13.6,-6.5,.206]),'B':np.array([13.6,6.5,.206])}
        obs=[dict(position_m=[-12.5,-6.5,.0335],confidence=.9)]
        f.observe_fixed(obs,pos,1.,received_seconds=1.2,current_positions=pos)
        r=f.robots['A']
        self.assertEqual(r.stage_started,1.2);self.assertEqual(r.navigator.last_seen,1.)
        self.assertEqual(r.events[-1]['seconds'],1.2)
