import os
import json
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


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


def get_median_frame(pipeline, align, num_frames=5):
    """
    여러 장의 depth 프레임을 픽셀별 median으로 합쳐서 노이즈를 줄임.
    회전되거나 모서리가 많은 물체에서 depth 노이즈가 심할 때 사용.
    color, intrinsics는 마지막 프레임 것을 사용.
    """
    depth_frames = []
    color_image = None
    intrinsics = None

    for i in range(num_frames):
        depth, color, intr = get_frame(pipeline, align)
        if depth is None:
            continue
        depth_frames.append(depth)
        color_image = color
        intrinsics = intr

    if len(depth_frames) == 0:
        return None, None, None

    depth_stack = np.stack(depth_frames, axis=0)
    median_depth = np.median(depth_stack, axis=0).astype(depth_frames[0].dtype)

    return median_depth, color_image, intrinsics


class PoseReader(Node):
    """joint_states 토픽에서 최신 관절값을 받아 저장해두는 노드"""
    def __init__(self):
        super().__init__('pose_reader')
        self.latest_joints = None
        self.create_subscription(JointState, '/joint_states', self._callback, 10)

    def _callback(self, msg):
        self.latest_joints = list(msg.position)


def init_robot():
    """ROS2 노드 초기화 — 한 번만 실행"""
    rclpy.init()
    node = PoseReader()
    return node


def get_ee_pose(node):
    """현재 joint 값 한 번 읽어오기"""
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_joints is not None:
            break
    return {"joints": node.latest_joints}


def save_capture(depth_image, color_image, intrinsics, depth_scale,
                  ee_pose, object_name, angle_label, index):
    """
    depth_image: numpy array (uint16 raw)
    color_image: numpy array (BGR)
    intrinsics: dict {fx, fy, cx, cy}
    depth_scale: float (미터 변환 계수)
    ee_pose: dict {joints} or None
    """
    base_name = f"{object_name}_{index:02d}_{angle_label}"

    np.save(os.path.join(DATA_DIR, f"{base_name}.npy"), depth_image)
    np.save(os.path.join(DATA_DIR, f"{base_name}_color.npy"), color_image)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "object_name": object_name,
        "angle_label": angle_label,
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
        "ee_pose": ee_pose,
    }
    with open(os.path.join(DATA_DIR, f"{base_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"저장 완료: {base_name}")
