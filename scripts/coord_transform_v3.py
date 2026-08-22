"""
find_grasp 결과(카메라 좌표계)를 base_link로 변환.
범열의 vision_to_base_test_v3.py의 to_base() 로직 재사용 (TF lookup 방식).
FK 직접계산(coord_transform.py, v1)은 버그가 있었으므로 폐기.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from tf2_ros import Buffer, TransformListener
import threading
import time

from safety_gate import check_safety

BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_color_optical_frame"
MAX_TF_AGE = 0.3


class GraspToBase(Node):
    def __init__(self):
        super().__init__("grasp_to_base")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def to_base_position(self, camera_point):
        transform = self.tf_buffer.lookup_transform(
            BASE_FRAME, CAMERA_FRAME, rclpy.time.Time()
        )
        t = transform.transform.translation
        q = transform.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        base_point = R @ camera_point + np.array([t.x, t.y, t.z])
        return base_point, R

    def to_base_full(self, camera_position, camera_rotation):
        base_position, R_base_cam = self.to_base_position(camera_position)
        R_base_grasp = R_base_cam @ camera_rotation
        return base_position, R_base_grasp


def transform_grasp(grasp_position_cam, grasp_rotation_cam, grasp_width,
                     offset_z=0.171, node=None):
    """
    find_grasp 결과를 base_link로 변환하고 안전 게이트까지 체크.
    node를 안 넘기면 내부에서 새로 만듦(단발성 사용).
    반환: dict {pos_base, R_base, link7_target, safety_ok, safety_reasons}
    """
    own_node = False
    if node is None:
        rclpy.init()
        node = GraspToBase()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        time.sleep(1.0)
        own_node = True

    pos_base, R_base = node.to_base_full(grasp_position_cam, grasp_rotation_cam)
    link7_target = pos_base + np.array([0, 0, offset_z])

    ok, reasons = check_safety(
        x=pos_base[0], y=pos_base[1], z_link7=link7_target[2], width=grasp_width
    )

    result = {
        "pos_base": pos_base,
        "R_base": R_base,
        "link7_target": link7_target,
        "safety_ok": ok,
        "safety_reasons": reasons,
    }

    if own_node:
        node.destroy_node()
        rclpy.shutdown()

    return result


if __name__ == "__main__":
    # block_1x1 (2026-08-21) find_grasp 결과로 테스트
    grasp_position_cam = np.array([-0.021909680333290552, 0.03511197929950179, 0.17250000685453415])
    grasp_rotation_cam = np.array([
        [-0.64634199, -0.17078528, 0.74368973],
        [0.76244821, -0.10591891, 0.63832117],
        [-0.03024506, 0.97959868, 0.19867477],
    ])
    grasp_width = 0.0331

    try:
        result = transform_grasp(grasp_position_cam, grasp_rotation_cam, grasp_width)

        print("=== base_link 기준 파지점 ===")
        print(f"x = {result['pos_base'][0]:.6f}")
        print(f"y = {result['pos_base'][1]:.6f}")
        print(f"z = {result['pos_base'][2]:.6f}")

        print("\n=== Link7 목표 (offset +0.171) ===")
        print(f"x = {result['link7_target'][0]:.6f}")
        print(f"y = {result['link7_target'][1]:.6f}")
        print(f"z = {result['link7_target'][2]:.6f}")

        print(f"\n=== 안전 게이트: {'PASS' if result['safety_ok'] else 'FAIL'} ===")
        for reason in result['safety_reasons']:
            print(f"  - {reason}")

    except Exception as e:
        print(f"TF lookup 실패 (로봇 미연결 상태로 예상됨, 22일은 정상): {e}")
