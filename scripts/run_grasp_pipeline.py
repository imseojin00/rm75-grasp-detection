"""
실기체 통합 파이프라인 (2026-08-23 최종, 완전 자동화 완료).
실시간 촬영 -> find_grasp 계산(위치/폭/방향) -> 테이블 높이 자동 계산(TF 기반)
-> base_link 변환 -> 안전 게이트

전제: 로봇이 go_topdown_scan.py의 SCAN_POSITION에 있어야 함
     rm_bringup + static TF(04) 켜져 있어야 함
사용법: python3 run_grasp_pipeline.py [물체이름(선택, 파일명 구분용일 뿐 계산과 무관)]

2026-08-23 밤 검증:
  GRASP_OFFSET_Z 오류(0.171 -> 0.150, 실측 오류) 발견 및 수정
  TABLE_Z를 depth 직접 실측으로 역산(-0.0879), 오차 3.7mm까지 개선
  물체 높이 자동계산 최종 검증: block_1x1 2.94cm (실제 2.5cm, 오차 0.44cm)
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
from table_z_from_tf import TableZFromTF
from safety_gate import check_safety

GRASP_OFFSET_Z = 0.150    # [m] Link7 -> 그리퍼 핀 손가락 끝 (2026-08-23 재실측)
HEIGHT_MIN, HEIGHT_MAX = 0.005, 0.15  # [m] 계산된 높이가 이 범위 밖이면 실패로 간주


def main():
    object_name = sys.argv[1] if len(sys.argv) > 1 else "object"

    print("=" * 50)
    print(f"실기체 파이프라인 시작: {object_name}")
    print("=" * 50)

    print("\n[1/6] 카메라 초기화 중...")
    pipeline, align, depth_scale = init_camera()

    print("[2/6] 로봇 연결 중 (pose 기록용)...")
    arm = init_robot()

    input("\n물체를 트레이 위에 놓고 Enter를 눌러 촬영하세요...")

    depth_image, color_image, intrinsics = get_frame(pipeline, align)
    ee_pose = get_ee_pose(arm)

    save_capture(depth_image, color_image, intrinsics, depth_scale,
                 ee_pose, object_name, "live", 0)

    pipeline.stop()
    base_name = f"{object_name}_00_live"
    print(f"촬영 완료: {base_name}")

    print("\n[3/6] find_grasp 계산 중 (위치/폭/방향)...")
    result = find_grasp(base_name)

    if result is None or result.get("best_grasp") is None:
        print("!! grasp 계산 실패 (antipodal 조건 만족하는 후보 없음). 중단.")
        return

    grasp_pose = result["grasp_pose"]
    grasp_position_cam = np.array(grasp_pose["position"])
    grasp_rotation_cam = np.array(grasp_pose["rotation_matrix"])
    grasp_width = grasp_pose["width"]

    print(f"  카메라 좌표(물체 윗면): {grasp_position_cam}")
    print(f"  width: {grasp_width*100:.2f} cm")

    print("\n[4/6] 좌표변환 + 테이블 높이 노드 준비...")
    if not rclpy.ok():
        rclpy.init()

    base_node = GraspToBase()
    table_node = TableZFromTF()
    executor = SingleThreadedExecutor()
    executor.add_node(base_node)
    executor.add_node(table_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        print("\n[5/6] 높이 자동 계산 (물체 이름 무관, TF 기반)...")
        table_z_cam = table_node.get_table_z_in_camera()
        top_z_cam = grasp_position_cam[2]
        object_height = table_z_cam - top_z_cam

        print(f"  테이블 z(카메라): {table_z_cam:.4f} m")
        print(f"  물체 윗면 z(카메라): {top_z_cam:.4f} m")
        print(f"  자동 계산된 물체 높이: {object_height*100:.2f} cm")

        if not (HEIGHT_MIN <= object_height <= HEIGHT_MAX):
            print(f"!! 높이 계산 이상함 ({object_height*100:.2f}cm, 정상범위 0.5~15cm) — 중단")
            return

        grasp_z_cam = (top_z_cam + table_z_cam) / 2.0
        grasp_position_cam_corrected = grasp_position_cam.copy()
        grasp_position_cam_corrected[2] = grasp_z_cam

        print("\n[6/6] base_link 변환 + 안전 게이트...")
        pos_base, R_base = base_node.to_base_full(
            grasp_position_cam_corrected, grasp_rotation_cam
        )
        link7_target = pos_base + np.array([0, 0, GRASP_OFFSET_Z])

        print(f"  base_link 파지점: {pos_base}")
        print(f"  Link7 목표: {link7_target}")
        print("  R_base_grasp:")
        print(R_base)

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
        base_node.destroy_node()
        table_node.destroy_node()
        rclpy.shutdown()
        arm.destroy_node()


if __name__ == "__main__":
    main()
