"""
TABLE_Z(base_link 기준 고정값)를 카메라 좌표계로 변환.
TF lookup 방향은 coord_transform_v3.py와 동일 (BASE_FRAME, CAMERA_FRAME).

수정: table_point_base의 x,y를 임의값(0.45, 0.0) 대신
카메라 위치의 x,y를 그대로 사용 - "카메라 바로 아래" 지점을 기준으로
계산해서, 임의 지점 대입으로 인한 오차를 없앰.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from tf2_ros import Buffer, TransformListener
import threading
import time

BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_color_optical_frame"
TABLE_Z_BASE = -0.0879  # [m] base_link 기준 테이블 면 (범열 실측)


class TableZFromTF(Node):
    def __init__(self):
        super().__init__('table_z_from_tf')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_table_z_in_camera(self, object_point_cam=None):
        """
        object_point_cam: 물체의 카메라 좌표(x,y,z), None이면 카메라 바로 아래 사용.
        카메라가 완벽히 수직이 아니면(실측상 약 5도 기울어짐), 물체 위치에 따라
        테이블까지의 거리가 달라지므로, 물체의 실제 x,y 위치에서 계산해야 정확함.
        """
        transform = self.tf_buffer.lookup_transform(
            BASE_FRAME, CAMERA_FRAME, rclpy.time.Time()
        )
        t = transform.transform.translation
        q = transform.transform.rotation

        x, y, z, w = q.x, q.y, q.z, q.w
        R_base_cam = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
        ])
        t_base_cam = np.array([t.x, t.y, t.z])

        R_cam_base = R_base_cam.T
        t_cam_base = -R_cam_base @ t_base_cam

        if object_point_cam is None:
            table_point_base = np.array([t.x, t.y, TABLE_Z_BASE])
        else:
            object_point_base = R_base_cam @ np.asarray(object_point_cam) + t_base_cam
            table_point_base = np.array([object_point_base[0], object_point_base[1], TABLE_Z_BASE])

        table_point_cam = R_cam_base @ table_point_base + t_cam_base

        return float(table_point_cam[2])


def main():
    rclpy.init()
    node = TableZFromTF()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        table_z_cam = node.get_table_z_in_camera()
        print(f"\n계산된 테이블 z (카메라 좌표계, TF 방식): {table_z_cam:.4f} m")
        print(f"실측 depth 값(테이블만 촬영): 0.2740 m")
        print(f"범열 실측값(plane_check.py): 0.2696 m")
        print(f"실측 대비 차이: {abs(table_z_cam - 0.2740)*1000:.1f} mm")
    except Exception as e:
        print(f"TF lookup 실패: {e}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
