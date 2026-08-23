"""
23일 실기체 1차용 통합 파이프라인.
실시간 촬영 -> find_grasp 계산 -> base_link 변환 -> 안전 게이트 -> 최종 출력

전제: 로봇(rm_bringup) + static TF(04) 켜져 있어야 함.
사용법: python3 run_grasp_pipeline.py block_1x1
"""
import sys
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
import threading
import time

from capture_utils import init_camera, get_frame, init_robot, get_ee_pose, save_capture
from find_grasp import find_grasp
from coord_transform_v3 import GraspToBase
from safety_gate import check_safety


def main():
    object_name = sys.argv[1] if len(sys.argv) > 1 else "block_1x1"

    print("=" * 50)
    print(f"실기체 파이프라인 시작: {object_name}")
    print("=" * 50)

    print("\n[1/5] 카메라 초기화 중...")
    pipeline, align, depth_scale = init_camera()

    print("[2/5] 로봇 연결 중 (pose 기록용)...")
    arm = init_robot()

    input("\n물체를 트레이 위에 놓고 Enter를 눌러 촬영하세요...")

    depth_image, color_image, intrinsics = get_frame(pipeline, align)
    ee_pose = get_ee_pose(arm)

    save_capture(depth_image, color_image, intrinsics, depth_scale,
                 ee_pose, object_name, "live", 0)

    pipeline.stop()
    base_name = f"{object_name}_00_live"
    print(f"촬영 완료: {base_name}")

    print("\n[3/5] find_grasp 계산 중...")
    result = find_grasp(base_name)

    if result is None or result.get("best_grasp") is None:
        print("!! grasp 계산 실패 (antipodal 조건 만족하는 후보 없음). 중단.")
        return

    grasp_pose = result["grasp_pose"]
    grasp_position_cam = np.array(grasp_pose["position"])
    grasp_rotation_cam = np.array(grasp_pose["rotation_matrix"])
    grasp_width = grasp_pose["width"]

    print(f"  카메라 좌표: {grasp_position_cam}")
    print(f"  width: {grasp_width*100:.2f} cm")

    print("\n[4/5] base_link 좌표 변환 중...")
    if not rclpy.ok():
        rclpy.init()
        rclpy.init()
    node = GraspToBase()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        pos_base, R_base = node.to_base_full(grasp_position_cam, grasp_rotation_cam)
        link7_target = pos_base + np.array([0, 0, 0.171])

        print(f"  base_link 파지점: {pos_base}")
        print(f"  Link7 목표: {link7_target}")

        print("\n[5/5] 안전 게이트 확인 중...")
        ok, reasons = check_safety(
            x=pos_base[0], y=pos_base[1], z_link7=link7_target[2],
            width=grasp_width
        )

        print("\n" + "=" * 50)
        if ok:
            print("안전 게이트: PASS")
            print("\n>>> topdown_test.py의 TEST_POSE에 아래 값을 넣으세요 <<<")
            print(f"TEST_POSE = ({link7_target[0]:.6f}, {link7_target[1]:.6f}, {link7_target[2]:.6f})")
        else:
            print("안전 게이트: FAIL — 절대 실행하지 마세요!")
            for r in reasons:
                print(f"  - {r}")
        print("=" * 50)

    except Exception as e:
        print(f"오류: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        arm.destroy_node()


if __name__ == "__main__":
    main()
