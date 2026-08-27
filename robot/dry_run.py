#!/usr/bin/env python3
"""
로봇/TF/ROS2 전혀 없이, 저장된 촬영 데이터로 계산 파이프라인만 끝까지 검증.
좌표값은 근사 TF를 쓰므로 부정확함 -- "코드가 안 죽고 끝까지 도는가",
"인자가 제대로 전달되는가"만 확인하는 용도. 실제 로봇 명령에는 쓰지 않음.
"""
import sys
sys.path.insert(0, '../scripts')
import numpy as np
from scipy.spatial.transform import Rotation
from find_grasp import find_grasp
from preprocess import cluster_geometry  # 존재 확인용, 안 쓰이면 무시됨

# ── 근사 TF (실측 아님, dry-run 전용) ─────────────────────
# 카메라가 테이블을 거의 수직으로 내려다본다고 가정.
# base_link 원점에서 카메라까지 대략적인 오프셋만 반영.
# 5도 기울기를 넣어야 인자 유무 차이가 실제로 드러남 (본론 문서에 기록된 실측 기울기)
_tilt = np.deg2rad(5.0)
_c, _s = np.cos(_tilt), np.sin(_tilt)
R_BASE_CAM_APPROX = np.array([
    [1,   0,    0],
    [0,  -_c,  _s],
    [0,  -_s, -_c],
])
T_BASE_CAM_APPROX = np.array([0.40, 0.0, 0.55])

TABLE_Z_BASE = -0.0879
HEIGHT_MIN, HEIGHT_MAX = 0.005, 0.15
GRASP_OFFSET_Z = 0.150

def fake_to_base_full(cam_pos, cam_rot):
    base_pos = R_BASE_CAM_APPROX @ cam_pos + T_BASE_CAM_APPROX
    base_rot = R_BASE_CAM_APPROX @ cam_rot
    return base_pos, base_rot

def fake_get_table_z_in_camera(object_point_cam=None):
    """TableZFromTF.get_table_z_in_camera()와 동일한 계산 로직,
    TF 대신 근사 R,t 사용."""
    R = R_BASE_CAM_APPROX
    t = T_BASE_CAM_APPROX
    R_cam_base = R.T
    t_cam_base = -R_cam_base @ t
    if object_point_cam is None:
        table_point_base = np.array([t[0], t[1], TABLE_Z_BASE])
    else:
        object_point_base = R @ np.asarray(object_point_cam) + t
        table_point_base = np.array([object_point_base[0], object_point_base[1], TABLE_Z_BASE])
    table_point_cam = R_cam_base @ table_point_base + t_cam_base
    return float(table_point_cam[2])

def fake_check_safety(x, y, z_link7, width):
    reasons = []
    if not (0.25 <= x <= 0.60): reasons.append(f"x={x:.4f} 범위 밖")
    if not (-0.25 <= y <= 0.25): reasons.append(f"y={y:.4f} 범위 밖")
    if not (z_link7 >= 0.058): reasons.append(f"z_link7={z_link7:.4f} 너무 낮음")
    if not (0.010 <= width <= 0.067): reasons.append(f"width={width:.4f} 범위 밖")
    return (len(reasons) == 0), reasons


def main(base_name):
    print(f"입력: {base_name}\n")

    # 1) find_grasp
    result = find_grasp(base_name)
    if result is None or result.get("best_grasp") is None:
        print("!! find_grasp 실패 -- 중단")
        return
    grasp_pose = result["grasp_pose"]
    grasp_position_cam = np.array(grasp_pose["position"])
    grasp_rotation_cam = np.array(grasp_pose["rotation_matrix"])
    grasp_width = grasp_pose["width"]
    print(f"[1] find_grasp 완료  width={grasp_width*100:.2f}cm")

    # 2) table_z 계산 (인자 있음 / 없음 둘 다 비교)
    z_with_arg = fake_get_table_z_in_camera(grasp_position_cam)
    z_no_arg = fake_get_table_z_in_camera(None)
    print(f"[2] table_z_cam  인자O={z_with_arg:.4f}  인자X={z_no_arg:.4f}  "
          f"차이={abs(z_with_arg - z_no_arg)*1000:.1f}mm")

    top_z_cam = grasp_position_cam[2]
    object_height = z_with_arg - top_z_cam
    print(f"    object_height = {object_height*100:.2f}cm")
    if not (HEIGHT_MIN <= object_height <= HEIGHT_MAX):
        print(f"!! 높이 이상함 ({object_height*100:.2f}cm) -- 근사 TF라 정상. "
              f"코드 흐름 검증을 위해 계속 진행")

    grasp_z_cam = (top_z_cam + z_with_arg) / 2.0
    grasp_position_cam_corrected = grasp_position_cam.copy()
    grasp_position_cam_corrected[2] = grasp_z_cam

    # 3) 좌표변환 (근사)
    pos_base, R_base = fake_to_base_full(grasp_position_cam_corrected, grasp_rotation_cam)
    link7_target = pos_base + np.array([0, 0, GRASP_OFFSET_Z])
    print(f"[3] pos_base(근사)= {pos_base}")
    print(f"    link7_target(근사) = {link7_target}")

    # 4) 안전 게이트
    ok, reasons = fake_check_safety(pos_base[0], pos_base[1], link7_target[2], grasp_width)
    print(f"[4] safety_gate = {'PASS' if ok else 'FAIL'}")
    for r in reasons:
        print(f"    - {r}")

    # 5) 회전 (grasp_quat 생성까지)
    points_cam = np.array(result["points"])
    try:
        from robot.pick_and_lift import compute_rotation_angle_minarea
    except Exception:
        compute_rotation_angle_minarea = None
    angle = 0.0
    if compute_rotation_angle_minarea:
        a = compute_rotation_angle_minarea(points_cam)
        angle = a if a is not None else 0.0
    z_axis = np.array([0, 0, -1])
    x_axis = np.array([np.cos(angle), np.sin(angle), 0])
    y_axis = np.cross(z_axis, x_axis)
    R_target = np.column_stack([x_axis, y_axis, z_axis])
    quat = Rotation.from_matrix(R_target).as_quat()
    print(f"[5] grasp_quat = {quat}  (angle={np.degrees(angle):.1f}도)")

    print("\n=== 여기까지 정상 (근사 TF, 좌표값 자체는 신뢰 불가) ===")
    print("=== 로봇 이동 없음 / 그리퍼 명령 없음 ===")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "block_1x2_00_live1"
    main(name)
