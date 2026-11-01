"""A chassis-mounted RGB camera and YOLO detections; no ball-state input."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image

from scene import as_array


def rotation_of(entity):
    w, x, y, z = as_array(entity.get_quat())
    scale = np.linalg.norm([w, x, y, z])
    w, x, y, z = np.array([w, x, y, z]) / scale
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


class OnboardCamera:
    """Pinhole RGB camera with a fixed transform relative to the chassis."""

    def __init__(self, scene, robot, config, initial_robot_position):
        self.robot = robot
        self.offset = np.array(config['position_in_robot_m'], dtype=float)
        pitch = np.deg2rad(config['downward_pitch_degrees'])
        self.forward = np.array([np.cos(pitch), 0, -np.sin(pitch)])
        p = np.array(initial_robot_position, dtype=float) + self.offset
        self.camera = scene.add_camera(
            res=tuple(config['resolution']), pos=p, lookat=p+self.forward,
            up=(0, 0, 1), fov=config['vertical_fov_degrees'],
            near=0.01, far=80, GUI=False,
        )

    def capture(self):
        rotation = rotation_of(self.robot)
        p = as_array(self.robot.get_pos()) + rotation @ self.offset
        self.camera.set_pose(pos=p, lookat=p+rotation@self.forward, up=rotation[:, 2])
        return np.asarray(self.camera.render(antialiasing=True)[0], dtype=np.uint8)

    def point_on_ball_center_plane(self, u, v, ball_radius):
        """Intersect an RGB detection ray with z=radius, for a ball on a flat court.

        The calibrated camera and robot pose are used; the target's simulator
        position, depth map, segmentation, and object IDs are never inputs.
        """
        intrinsic = self.camera.intrinsics
        ray_camera = np.array([
            (u-intrinsic[0, 2])/intrinsic[0, 0],
            -(v-intrinsic[1, 2])/intrinsic[1, 1], -1.0,
        ])
        transform = self.camera.transform
        origin = transform[:3, 3]
        direction = transform[:3, :3] @ ray_camera
        if direction[2] >= -1e-5:
            return None
        distance = (ball_radius-origin[2])/direction[2]
        if distance <= 0:
            return None
        world = origin + distance*direction
        local = rotation_of(self.robot).T @ (world-as_array(self.robot.get_pos()))
        if not (0.20 < local[0] < 8.0):
            return None
        return local


@dataclass
class BallObservation:
    bbox_xyxy: list[float]
    confidence: float
    position_in_robot_m: list[float]


class BallDetector:
    def __init__(self, weights: Path, config):
        from ultralytics import YOLO, settings
        settings.update({'sync': False})
        self.model = YOLO(str(weights))
        self.config = config
        self.classes = [int(k) for k, name in self.model.names.items()
                        if name.lower() in ('sports ball', 'tennis ball', 'ball')]
        if not self.classes:
            raise ValueError('The YOLO model has no ball class.')
        self.elapsed_seconds = []

    def detect(self, rgb, camera, ball_radius):
        start = time.perf_counter()
        result = self.model.predict(
            Image.fromarray(rgb), device=self.config['device'],
            imgsz=self.config['image_size'], conf=self.config['confidence'],
            classes=self.classes, verbose=False, save=False,
        )[0]
        self.elapsed_seconds.append(time.perf_counter()-start)
        observations = []
        for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            u, v = (box[:2]+box[2:])/2
            local = camera.point_on_ball_center_plane(float(u), float(v), ball_radius)
            if local is not None:
                observations.append(BallObservation(box.astype(float).tolist(), float(confidence), local.tolist()))
        observations.sort(key=lambda item: np.linalg.norm(item.position_in_robot_m[:2]))
        # Ultralytics returns a BGR annotated image.
        annotated = result.plot()[:, :, ::-1].copy()
        return observations, annotated
