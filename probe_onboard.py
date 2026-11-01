"""Inspect the real onboard view and pretrained YOLO on three initial poses."""
import json
from pathlib import Path

import genesis as gs
import numpy as np
from PIL import Image

from scene import ROOT, as_array, build_scene, initialize_shovel
from onboard_vision import OnboardCamera, BallDetector


def main():
    output = ROOT/'artifacts'/'vision_probe'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'scene_config.json').read_text())
    vision = json.loads((ROOT/'vision_config.json').read_text())
    detector = BallDetector(ROOT/vision['model'], vision)
    gs.init(backend=gs.amdgpu, seed=13, logging_level='warning')
    results = []
    try:
        scene, robot, balls, *_ = build_scene(config, (960, 720))
        onboard = OnboardCamera(scene, robot, vision['camera'], config['robot_position_m'])
        scene.build()
        initialize_shovel(robot)
        for _ in range(120):
            scene.step()
        start = as_array(robot.get_pos())
        for name, shift in [('start', 0.0), ('near', 0.18), ('contact', 0.235)]:
            # These are independent viewpoint inspections, not a driving test.
            robot.set_pos(start+np.array([shift, 0, 0]))
            rgb = onboard.capture()
            observations, annotated = detector.detect(rgb, onboard, config['ball_radius_m'])
            Image.fromarray(rgb).save(output/f'{name}_rgb.png')
            Image.fromarray(annotated).save(output/f'{name}_yolo.png')
            result = {'name':name, 'detections':[vars(o) for o in observations],
                      'reference_ball_position_m':as_array(balls[0].get_pos()).tolist(),
                      'robot_position_m':as_array(robot.get_pos()).tolist()}
            results.append(result)
            print(json.dumps(result), flush=True)
        (output/'probe.json').write_text(json.dumps(results, indent=2)+'\n')
    finally:
        gs.destroy()


if __name__ == '__main__':
    main()
