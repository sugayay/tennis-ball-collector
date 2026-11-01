import json
from pathlib import Path
import unittest

import numpy as np

from mission_control import MissionRobot,PICKUP


class TransitShovelTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'mission_straight_config.json').read_text())
        self.position=np.array([-13.6,-6.5,.205])

    def robot(self,distance):
        robot=MissionRobot('A',self.config,[23.77,10.97])
        robot.assign({'position_m':(self.position+np.array([distance,0.,-.1715])).tolist()},self.position,0.)
        return robot

    def test_transit_raises_shovel_and_lowers_it_before_visual_handoff(self):
        robot=self.robot(3.5)
        for step in range(91):
            now=step/120;robot.navigator.last_seen=now
            _,joints=robot.command(self.position,np.eye(3),now)
        self.assertAlmostEqual(joints[0],-.30)
        self.assertAlmostEqual(float(joints.sum()),0.)
        robot.navigator.target=self.position+np.array([1.3,0.,-.1715])
        for step in range(91,260):
            now=step/120;robot.navigator.last_seen=now
            _,joints=robot.command(self.position,np.eye(3),now)
        self.assertAlmostEqual(joints[0],PICKUP)

    def test_short_approach_keeps_original_shovel_pose(self):
        robot=self.robot(1.3)
        for step in range(91):
            now=step/120;robot.navigator.last_seen=now
            _,joints=robot.command(self.position,np.eye(3),now)
            np.testing.assert_array_equal(joints,[PICKUP,-PICKUP])


if __name__=='__main__':unittest.main()
