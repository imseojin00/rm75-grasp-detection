import json
import numpy as np
import pyrealsense2 as rs
from datetime import datetime


def init_camera():
    """카메라 세팅 — 한 번만 실행"""
    WIDTH, HEIGHT, FPS = 640, 480, 30

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, align, depth_scale


def get_frame(pipeline, align):
    """사진 한 장 찍기"""
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    intrinsics_obj = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    intrinsics = {
        "fx": intrinsics_obj.fx,
        "fy": intrinsics_obj.fy,
        "cx": intrinsics_obj.ppx,
        "cy": intrinsics_obj.ppy,
        "width": intrinsics_obj.width,
        "height": intrinsics_obj.height,
    }

    return depth_image, color_image, intrinsics


def save_capture(depth_image, color_image, intrinsics, depth_scale,
                  ee_pose, object_name, angle_label, index):
    """
    depth_image: numpy array (uint16 raw)
    color_image: numpy array (BGR)
    intrinsics: dict {fx, fy, cx, cy}
    depth_scale: float (미터 변환 계수)
    ee_pose: list or dict
    """
    base_name = f"{object_name}_{index:02d}_{angle_label}"

    np.save(f"data/{base_name}.npy", depth_image)
    np.save(f"data/{base_name}_color.npy", color_image)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "object_name": object_name,
        "angle_label": angle_label,
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
        "ee_pose": ee_pose,
    }
    with open(f"data/{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"저장 완료: {base_name}")
