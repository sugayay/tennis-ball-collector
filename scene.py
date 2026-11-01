"""Stage 1: build and inspect a tennis-ball collection scene in Genesis.

The robot is a free rigid body with two wheel joints. No driving, pickup
controller, or AI is implemented in this scene preview. The video moves only
the camera. shovel_demo.py separately tests the actuated shovel mechanism.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import av
import genesis as gs
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent


def as_array(value):
    return value.detach().cpu().numpy().astype(float)


def initialize_shovel(robot):
    """Set the starting pose once; all subsequent motion uses force control."""
    from support_suspension import configure_if_present
    configure_if_present(robot)
    joints = [robot.get_joint(name) for name in ("shovel_lift_joint", "shovel_tilt_joint")]
    dofs = [joint.dofs_idx_local[0] for joint in joints]
    pose = np.array([-0.448, 0.448])
    robot.set_dofs_position(pose, dofs_idx_local=dofs)
    robot.set_dofs_kp(np.array([120.0, 45.0]), dofs_idx_local=dofs)
    robot.set_dofs_kv(np.array([12.0, 4.5]), dofs_idx_local=dofs)
    robot.set_dofs_force_range(np.array([-14.0, -8.0]), np.array([14.0, 8.0]), dofs_idx_local=dofs)
    robot.control_dofs_position(pose, dofs_idx_local=dofs)
    return dofs


def add_box(scene, name, size, position, color, *, collision=False):
    return scene.add_entity(
        gs.morphs.Box(size=size, pos=position, fixed=True, collision=collision),
        surface=gs.surfaces.Rough(color=color),
        name=name,
    )


def build_scene(config, resolution):
    rolling_resistance = config.get("ball_rolling_resistance_m")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 120, substeps=config.get('physics_substeps', 2)),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            max_collision_pairs=100,
            enable_torsional_friction=rolling_resistance is not None,
            enable_rolling_friction=rolling_resistance is not None,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.045, 0.065, 0.09),
            shadow=False,
        ),
        viewer_options=gs.options.ViewerOptions(realtime_factor=None),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, -0.085)),
        surface=gs.surfaces.Rough(color=(0.12, 0.16, 0.19)),
        name="surrounding_floor",
    )
    court_x, court_y = config["court_size_m"]
    area_x, area_y = config["playing_area_m"]
    add_box(
        scene, "court_floor", (court_x, court_y, 0.08), (0, 0, -0.04),
        (0.07, 0.29, 0.23), collision=True,
    )
    add_box(
        scene, "playing_area", (area_x, area_y, 0.001), (0, 0, 0.0005),
        (0.075, 0.27, 0.47),
    )
    white = (0.88, 0.95, 0.93)
    line_width = 0.05
    singles_width = config["singles_width_m"]
    service_distance = config["service_line_distance_m"]
    for sign in (-1, 1):
        add_box(
            scene, f"doubles_sideline_{sign}", (area_x, line_width, 0.002),
            (0, sign * (area_y - line_width) / 2, 0.002), white,
        )
        add_box(
            scene, f"singles_sideline_{sign}", (area_x, line_width, 0.002),
            (0, sign * (singles_width - line_width) / 2, 0.002), white,
        )
        add_box(
            scene, f"baseline_{sign}", (line_width, area_y, 0.002),
            (sign * (area_x - line_width) / 2, 0, 0.002), white,
        )
        add_box(
            scene, f"service_line_{sign}", (line_width, singles_width, 0.002),
            (sign * (service_distance - line_width / 2), 0, 0.002), white,
        )
        add_box(
            scene, f"center_mark_{sign}", (0.10, line_width, 0.002),
            (sign * (area_x / 2 - line_width - 0.05), 0, 0.002), white,
        )
    add_box(
        scene, "center_service_line", (2 * service_distance, line_width, 0.002),
        (0, 0, 0.002), white,
    )

    # A rigid, eight-panel approximation: no deformable mesh or ball-through holes.
    net_half_span = area_y / 2 + config["net_post_offset_m"]
    for sign in (-1, 1):
        add_box(
            scene, f"net_post_{sign}", (0.08, 0.08, config["net_post_height_m"]),
            (0, sign * net_half_span, config["net_post_height_m"] / 2),
            (0.10, 0.15, 0.17), collision=True,
        )
    panel_count = 8
    panel_width = 2 * net_half_span / panel_count
    for index in range(panel_count):
        y = -net_half_span + (index + 0.5) * panel_width
        height = config["net_center_height_m"] + (
            config["net_post_height_m"] - config["net_center_height_m"]
        ) * (abs(y) / net_half_span) ** 2
        add_box(
            scene, f"net_panel_{index}", (0.025, panel_width, height),
            (0, y, height / 2), (0.11, 0.16, 0.17), collision=True,
        )
        add_box(
            scene, f"net_top_band_{index}", (0.035, panel_width, 0.05),
            (0, y, height - 0.025), white,
        )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(ROOT / "assets" / config.get("robot_urdf", "collector.urdf")),
            pos=tuple(config["robot_position_m"]),
            fixed=False,
            prioritize_urdf_material=True, decimate=False,
        ),
        material=gs.materials.Rigid(friction=0.8),
        name="collector_robot",
    )
    radius = config["ball_radius_m"]
    density = config["ball_mass_kg"] / (4 * math.pi * radius ** 3 / 3)
    balls = []
    for index, (x, y) in enumerate(config["ball_positions_m"], start=1):
        balls.append(scene.add_entity(
            gs.morphs.Sphere(radius=radius, pos=(x, y, radius + 0.008)),
            material=gs.materials.Rigid(rho=density, friction=0.65, friction_rolling=rolling_resistance),
            surface=gs.surfaces.Rough(color=(0.82, 0.96, 0.055)),
            name=f"tennis_ball_{index:02d}",
        ))
    camera = scene.add_camera(
        res=resolution, pos=(26, -31, 33), lookat=(0, 0, 0.04),
        fov=50, far=200, GUI=False,
    )
    top_camera = scene.add_camera(
        res=(1600, 1000), pos=(0, 0, 34), lookat=(0, 0, 0),
        up=(0, 1, 0), fov=48, far=200, GUI=False,
    )
    robot_x, robot_y, _ = config["robot_position_m"]
    detail_camera = scene.add_camera(
        res=(960, 720), pos=(robot_x + 1.35, robot_y - 1.4, 1.3),
        lookat=(robot_x + 0.25, robot_y, 0.16),
        fov=45, far=200, GUI=False,
    )
    return scene, robot, balls, camera, top_camera, detail_camera


def save_rgb(camera, path):
    rgb = np.asarray(camera.render(antialiasing=True)[0], dtype=np.uint8)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"Unexpected RGB shape: {rgb.shape}"
    assert rgb.std() > 10, "Rendered image has almost no visual content."
    Image.fromarray(rgb).save(path)
    return rgb


def record_camera_preview(camera, output_path, frame_count, fps):
    width, height = camera.res
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "19", "preset": "fast"}
    first = None
    last = None
    try:
        for index in range(frame_count):
            t = index / max(1, frame_count - 1)
            angle = math.radians(-60 + 26 * t)
            camera.set_pose(
                pos=(41 * math.cos(angle), 41 * math.sin(angle), 33),
                lookat=(0, 0, 0.04),
            )
            rgb = np.asarray(camera.render()[0], dtype=np.uint8)
            first = rgb.copy() if first is None else first
            last = rgb
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            if (index + 1) % 24 == 0:
                print(f"Preview rendered: {index + 1}/{frame_count}", flush=True)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    assert float(np.abs(first.astype(float) - last.astype(float)).mean()) > 1.0
    with av.open(str(output_path)) as check:
        decoded = list(check.decode(video=0))
        verified_fps = float(check.streams.video[0].average_rate)
    assert len(decoded) == frame_count, "Preview video frame count is incomplete."
    return {
        "frames": len(decoded), "fps": verified_fps,
        "seconds": len(decoded) / verified_fps, "camera_motion_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["amdgpu", "cpu"], default="amdgpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--preview-frames", type=int, default=72)
    args = parser.parse_args()
    if args.preview_frames < 2:
        parser.error("--preview-frames must be at least 2")
    config = json.loads((ROOT / "scene_config.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        config["court_size_m"],
        np.array(config["playing_area_m"]) + 2 * np.array(config["runoff_margin_m"]),
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gs.init(backend=getattr(gs, args.backend), seed=13, logging_level="warning")
    try:
        scene, robot, balls, camera, top_camera, detail_camera = build_scene(config, (1280, 720))
        scene.build()
        initialize_shovel(robot)
        wheel_joints = [robot.get_joint(name) for name in ("left_wheel_joint", "right_wheel_joint")]
        assert len(wheel_joints) == 2
        initial_robot = as_array(robot.get_pos())
        for _ in range(90):
            scene.step()
        settled_robot = as_array(robot.get_pos())
        robot_quaternion = as_array(robot.get_quat())
        ball_positions = np.stack([as_array(ball.get_pos()) for ball in balls])
        qpos = as_array(robot.get_qpos())
        assert np.isfinite(qpos).all() and np.isfinite(ball_positions).all()
        radius = config["ball_radius_m"]
        assert np.all(np.abs(ball_positions[:, 2] - radius) < 0.012), "A ball fell through or floats above the court."
        drift = float(np.linalg.norm(settled_robot[:2] - initial_robot[:2]))
        assert drift < 0.025, f"Robot moved unexpectedly at rest: {drift:.4f} m"
        assert 0.17 < settled_robot[2] < 0.24, "Robot is not standing on its wheels."
        up_z = 1 - 2 * (robot_quaternion[1] ** 2 + robot_quaternion[2] ** 2)
        tilt_degrees = math.degrees(math.acos(float(np.clip(up_z, -1, 1))))
        assert tilt_degrees < 3, f"Robot tilted unexpectedly: {tilt_degrees:.3f} degrees"
        print("Scene built and stationary physics checked.", flush=True)
        save_rgb(camera, output / "scene_overview.png")
        save_rgb(top_camera, output / "scene_top.png")
        save_rgb(detail_camera, output / "robot_detail.png")
        video_report = None
        if not args.image_only:
            video_report = record_camera_preview(camera, output / "scene_preview.mp4", args.preview_frames, 24)
        report = {
            "stage": 1,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "genesis_version": metadata.version("genesis-world"),
            "physics_backend": getattr(gs.backend, "name", str(gs.backend)),
            "physics_device": str(gs.device),
            "renderer": "Genesis rasterizer; existing software rendering configuration",
            "solver_dispatch": os.environ.get("QD_PERFDISPATCH_FORCE"),
            "court_size_m": config["court_size_m"],
            "playing_area_m": config["playing_area_m"],
            "singles_width_m": config["singles_width_m"],
            "runoff_margin_m": config["runoff_margin_m"],
            "court_dimension_source": "ITF Rules of Tennis 2026, Rule 1 and Appendix X",
            "net_model": "fixed rigid panels approximating the center sag; not a deformable mesh",
            "ball_count": len(balls),
            "balls_outside_playing_area": int(np.count_nonzero(
                (np.abs(ball_positions[:, 0]) > config["playing_area_m"][0] / 2)
                | (np.abs(ball_positions[:, 1]) > config["playing_area_m"][1] / 2)
            )),
            "ball_radius_m": radius,
            "ball_mass_kg": config["ball_mass_kg"],
            "settled_ball_positions_m": ball_positions.tolist(),
            "robot_free_base": True,
            "robot_wheel_joints": [joint.name for joint in wheel_joints],
            "settled_robot_position_m": settled_robot.tolist(),
            "robot_rest_drift_m": drift,
            "robot_tilt_degrees": tilt_degrees,
            "settle_steps": 90,
            "screenshots": ["scene_overview.png", "scene_top.png", "robot_detail.png"],
            "preview": video_report,
            "implemented": ["full_size_court", "runoff_area", "fixed_net", "dynamic_balls", "free_wheeled_robot_model", "actuated_shovel_model", "basket_collision_geometries", "passive_casters", "preview_cameras", "stationary_physics_check"],
            "not_implemented": ["autonomous_driving_controller", "ball_detection_ai", "autonomous_collection", "players", "moving_shots", "player_avoidance"],
            "design_pending_user_consultation": ["perception_camera", "ai_model", "autonomous_control_method", "court_entry_policy", "shovel_final_dimensions"],
            "verified": True,
        }
        (output / "scene_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
