#!/usr/bin/env python3
"""
완전 자동 파지 파이프라인 (2026-08-24, 실기체 검증 완료 절차 기반).
촬영 -> 계산(위치/폭/높이) -> 좌표변환 -> 이동(접근+하강) -> 그리퍼 파지 -> 상승

전제:
  - 로봇이 go_topdown_scan.py로 스캔 자세에 있어야 함 (이 스크립트 실행 전에 미리)
  - rm_bringup + static TF(04) 켜져 있어야 함

기존 검증된 로직 재사용:
  - 계산부: run_grasp_pipeline.py
  - 이동부: topdown_test.py의 move_pose/move_linear 로직
"""
import sys
import time
import copy

import numpy as np
import cv2
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject, CollisionObject, Constraints,
    OrientationConstraint, PlanningScene, PositionConstraint, RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from shape_msgs.msg import SolidPrimitive

sys.path.insert(0, '../scripts')
from capture_utils import init_camera, get_frame, get_median_frame, init_robot, get_ee_pose, save_capture
from find_grasp import find_grasp
from coord_transform_v3 import GraspToBase
from table_z_from_tf import TableZFromTF
from safety_gate import check_safety

# ── 실측값 (오늘 검증됨) ──────────────────────────────────
TABLE_Z = -0.0879
GRASP_OFFSET_Z = 0.150
TOPDOWN_Q = (0.0, 1.0, 0.0, 0.0)
LIFT = 0.10  # [m] 들어올리는 높이

BASE_FRAME = "base_link"
ARM_GROUP = "rm_group"
EE_LINK = "Link7"
VEL_SCALE = 0.1
ACC_SCALE = 0.1

GRIPPER_BOX_SIZE = (0.11, 0.13, 0.18)
GRIPPER_BOX_CENTER = (0.0, 0.0, 0.09)
TOUCH_LINKS = ["Link7", "Link6", "Link5", "Link4", "Link3", "Link2", "Link1", "base_link"]

HEIGHT_MIN, HEIGHT_MAX = 0.005, 0.15


def compute_rotation_angle_minarea(points):
    """
    물체 point cloud의 x-y를 이미지로 투영해서,
    minAreaRect로 실제 사각형 회전각을 계산. PCA보다 45도 근처에서 안정적.
    반환: 라디안 각도, 실패시 None
    """
    xy = points[:, :2] * 1000  # m -> mm
    xy_min = xy.min(axis=0)
    xy_shifted = xy - xy_min
    img_size = int(xy_shifted.max()) + 20
    if img_size <= 0 or img_size > 2000:
        return None
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for x, y in xy_shifted:
        cv2.circle(mask, (int(x) + 10, int(y) + 10), 2, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest = max(contours, key=cv2.contourArea)

    # 반지름 변동성으로 원형/사각형 판별 (fill_ratio보다 명확히 구분됨, 검증됨)
    # 원형: 변동성 0.2 이상, 사각형: 0.1 이하
    M = cv2.moments(largest)
    if M['m00'] > 0:
        mcx = M['m10'] / M['m00']
        mcy = M['m01'] / M['m00']
        contour_points = largest.reshape(-1, 2)
        distances = np.sqrt((contour_points[:, 0] - mcx) ** 2 + (contour_points[:, 1] - mcy) ** 2)
        variability = distances.std() / distances.mean() if distances.mean() > 0 else 1.0
        if variability > 0.15:
            print(f"    [회전계산] 반지름 변동성={variability:.3f} -> 원형으로 판단, 회전 보정 안 함")
            return 0.0

    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle_deg = rect
    angle_rad = np.radians(angle_deg)

    long_side, short_side = max(w, h), min(w, h)
    is_square = (short_side / long_side) > 0.85 if long_side > 0 else True

    if is_square:
        period = np.pi / 2
    else:
        period = np.pi

    while angle_rad > period / 2:
        angle_rad -= period
    while angle_rad < -period / 2:
        angle_rad += period

    return angle_rad


def make_pose(x, y, z, quat=(0.0, 0.0, 0.0, 1.0)):
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) = (
        float(v) for v in quat
    )
    return p


class PickAndLift(Node):
    def __init__(self):
        super().__init__("pick_and_lift")

        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )

        self.move_cli = ActionClient(self, MoveGroup, "/move_action")
        self.exec_cli = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.scene_cli = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.cart_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path")

        self.get_logger().info("서버 대기 중...")
        ok = (
            self.move_cli.wait_for_server(timeout_sec=10.0)
            and self.exec_cli.wait_for_server(timeout_sec=10.0)
            and self.scene_cli.wait_for_service(timeout_sec=10.0)
            and self.cart_cli.wait_for_service(timeout_sec=10.0)
        )
        if not ok:
            self.get_logger().error("MoveIt 인터페이스를 찾지 못했습니다. bringup이 떠 있나요?")
            sys.exit(1)

        for _ in range(50):
            if self.current_joint_state is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_joint_state is None:
            self.get_logger().error("/joint_states를 받지 못했습니다.")
            sys.exit(1)

        self.get_logger().info("서버 연결 + joint_states 수신 완료")

    def _joint_state_callback(self, msg):
        self.current_joint_state = copy.deepcopy(msg)

    def _get_current_robot_state(self):
        if self.current_joint_state is None:
            return None
        state = RobotState()
        state.joint_state = copy.deepcopy(self.current_joint_state)
        return state

    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _send_goal(self, client, goal):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            return None
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result().result

    def confirm(self, label):
        answer = input(f"  ▶ [{label}] 확인했으면 Enter, 중단은 q+Enter: ").strip()
        if answer.lower() == "q":
            self.get_logger().warn("사용자 중단")
            return False
        return True

    def _execute(self, trajectory, label):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        result = self._send_goal(self.exec_cli, goal)
        ok = result is not None and result.error_code.val == 1
        self.get_logger().info(f"[실행] {label} -> {'성공' if ok else '실패'}")
        return ok

    def setup_scene(self):
        scene = PlanningScene(is_diff=True)

        table = CollisionObject()
        table.header.frame_id = BASE_FRAME
        table.id = "table"
        table.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.2, 1.2, 0.02])]
        table.primitive_poses = [make_pose(0.45, 0.0, TABLE_Z - 0.011)]
        table.operation = CollisionObject.ADD
        scene.world.collision_objects.append(table)

        envelope = CollisionObject()
        envelope.header.frame_id = EE_LINK
        envelope.id = "gripper_envelope"
        envelope.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(GRIPPER_BOX_SIZE))]
        envelope.primitive_poses = [make_pose(*GRIPPER_BOX_CENTER)]
        envelope.operation = CollisionObject.ADD

        attached = AttachedCollisionObject()
        attached.link_name = EE_LINK
        attached.object = envelope
        attached.touch_links = TOUCH_LINKS
        scene.robot_state.attached_collision_objects.append(attached)
        scene.robot_state.is_diff = True

        result = self._call(self.scene_cli, ApplyPlanningScene.Request(scene=scene))
        ok = result is not None and result.success
        self.get_logger().info(f"[scene] 등록 {'성공' if ok else '실패'}")
        return ok

    def move_pose(self, x, y, z, quat=TOPDOWN_Q, label=None):
        label = label or f"pose ({x:.3f}, {y:.3f}, {z:.3f})"

        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = EE_LINK
        pc.target_point_offset = Vector3()
        pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])]
        pc.constraint_region.primitive_poses = [make_pose(x, y, z)]
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK
        oc.orientation = make_pose(0, 0, 0, quat).orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0

        goal = MoveGroup.Goal()
        goal.request.group_name = ARM_GROUP

        current_state = self._get_current_robot_state()
        if current_state is None:
            self.get_logger().error("현재 RobotState를 만들 수 없습니다.")
            return False
        goal.request.start_state = current_state

        goal.request.goal_constraints = [Constraints(position_constraints=[pc], orientation_constraints=[oc])]
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = VEL_SCALE
        goal.request.max_acceleration_scaling_factor = ACC_SCALE
        goal.planning_options.plan_only = True

        self.get_logger().info(f"[계획 요청] {label}")
        result = self._send_goal(self.move_cli, goal)

        if result is None or result.error_code.val != 1:
            code = "None" if result is None else result.error_code.val
            self.get_logger().error(f"[계획] {label} 실패 (error_code={code})")
            return False

        self.get_logger().info(f"[계획] {label} 성공")
        if not self.confirm(label):
            return False
        return self._execute(result.planned_trajectory, label)

    def move_linear(self, x, y, z, quat=TOPDOWN_Q, label=None):
        label = label or f"직선 ({x:.3f}, {y:.3f}, {z:.3f})"

        request = GetCartesianPath.Request()
        request.header.frame_id = BASE_FRAME
        request.group_name = ARM_GROUP
        request.link_name = EE_LINK

        current_state = self._get_current_robot_state()
        if current_state is None:
            self.get_logger().error("현재 RobotState를 만들 수 없습니다.")
            return False
        request.start_state = current_state

        request.waypoints = [make_pose(x, y, z, quat)]
        request.max_step = 0.005
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = VEL_SCALE
        request.max_acceleration_scaling_factor = ACC_SCALE

        result = self._call(self.cart_cli, request)
        if result is None:
            self.get_logger().error(f"[계획] {label}: Cartesian 결과 없음")
            return False

        fraction = result.fraction
        print(f"    직선 생성 비율: {fraction*100:.1f}%")

        if fraction < 0.9:
            self.get_logger().warn(f"[계획] {label}: 직선이 {fraction*100:.1f}%만 생성됨 -> 중단")
            return False

        self.get_logger().info(f"[계획] {label} Cartesian 성공")
        if not self.confirm(label):
            return False
        return self._execute(result.solution, label)
    def gripper_open(self):
        """파지 전에 그리퍼를 365 위치로 연다."""
        from rm_ros_interfaces.msg import Gripperset

        pub = self.create_publisher(
            Gripperset,
            '/rm_driver/set_gripper_position_cmd',
            10
        )

        time.sleep(0.5)

        msg = Gripperset()
        msg.position = 365
        msg.block = False
        msg.timeout = 0

        print("  -> 그리퍼 열기: position=365")

        for _ in range(30):
            pub.publish(msg)
            time.sleep(0.1)

        print("  -> 그리퍼 열기 완료")

    def gripper_pick(self, grasp_width):
        """하강 후 힘 제어로 물체를 잡는다."""
        from rm_ros_interfaces.msg import Gripperpick

        width_mm = grasp_width * 1000.0

        pub = self.create_publisher(
            Gripperpick,
            '/rm_driver/set_gripper_pick_cmd',
            10
        )

        time.sleep(0.5)

        msg = Gripperpick()
        msg.speed = 200
        msg.force = 300
        msg.block = False
        msg.timeout = 0

        print(f"  -> 그리퍼 파지 시작: width={width_mm:.1f}mm, force=700")

        for _ in range(30):
            pub.publish(msg)
            time.sleep(0.1)

        print("  -> 그리퍼 파지 완료")


        print("  -> 그리퍼 파지 완료")


def main():
    object_name = sys.argv[1] if len(sys.argv) > 1 else "object"

    print("=" * 60)
    print(f"완전 자동 파지 파이프라인: {object_name}")
    print("(전제: 로봇이 스캔 자세에 있어야 함 - go_topdown_scan.py 먼저 실행)")
    print("=" * 60)

    # ── 1. 촬영 ──
    print("\n[1/7] 카메라 초기화...")
    pipeline, align, depth_scale = init_camera()

    print("[2/7] 로봇 연결 (pose 기록용)...")
    arm = init_robot()

    input("\n물체를 트레이 위에 놓고 Enter를 눌러 촬영을 시작하세요...")

    MAX_ATTEMPTS = 15
    MAX_TIME_SEC = 60
    MIN_GOOD_POINTS = 150  # find_grasp 내부 기준과 동일

    start_time = time.time()
    attempt = 0
    result = None
    base_name = None

    while attempt < MAX_ATTEMPTS and (time.time() - start_time) < MAX_TIME_SEC:
        attempt += 1
        elapsed = time.time() - start_time
        print(f"  [{attempt}번째 촬영 시도, 경과 {elapsed:.0f}초]")

        depth_image, color_image, intrinsics = get_frame(pipeline, align)
        ee_pose = get_ee_pose(arm)
        save_capture(depth_image, color_image, intrinsics, depth_scale, ee_pose, object_name, f"live{attempt}", 0)
        trial_name = f"{object_name}_00_live{attempt}"
        try:
            trial_result = find_grasp(trial_name)
        except Exception as e:
            print(f"    -> preprocess/계산 중 오류 ({e}), 재시도...")
            trial_result = None

        if trial_result is not None and trial_result.get("best_grasp") is not None:
            num_points = trial_result["num_points"]
            if num_points >= MIN_GOOD_POINTS:
                print(f"    -> 좋은 촬영 발견! (점 {num_points}개) 진행합니다.")
                result = trial_result
                base_name = trial_name
                break
            else:
                print(f"    -> 점 개수 부족 ({num_points}개), 재시도...")
        else:
            print(f"    -> grasp 계산 실패, 재시도...")

    pipeline.stop()

    if result is None:
        elapsed = time.time() - start_time
        print(f"\n!! {attempt}번 시도({elapsed:.0f}초) 후에도 좋은 촬영을 못 찾았습니다. 중단.")
        print("물체 위치/각도를 바꿔서 다시 시도해보세요.")
        return

    print(f"\n선택된 촬영: {base_name} (총 {attempt}번 시도)")
    grasp_pose = result["grasp_pose"]
    pca_axis_0 = np.array(result["axis_0"])  # PCA가 계산한 짧은 축 (closing_axis보다 안정적)
    grasp_position_cam = np.array(grasp_pose["position"])
    grasp_rotation_cam = np.array(grasp_pose["rotation_matrix"])
    grasp_width = grasp_pose["width"]
    print(f"  width: {grasp_width*100:.2f} cm")

    print("\n[4/7] 좌표변환 + 높이 계산 노드 준비...")
    if not rclpy.ok():
        rclpy.init()

    base_node = GraspToBase()
    table_node = TableZFromTF()
    move_node = PickAndLift()

    executor = SingleThreadedExecutor()
    executor.add_node(base_node)
    executor.add_node(table_node)
    spin_thread = __import__('threading').Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        table_z_cam = table_node.get_table_z_in_camera(grasp_position_cam)
        top_z_cam = grasp_position_cam[2]
        object_height = table_z_cam - top_z_cam
        print(f"  자동 계산된 물체 높이: {object_height*100:.2f} cm")

        if not (HEIGHT_MIN <= object_height <= HEIGHT_MAX):
            print(f"!! 높이 계산 이상함 ({object_height*100:.2f}cm) — 중단")
            return

        grasp_z_cam = (top_z_cam + table_z_cam) / 2.0
        grasp_position_cam_corrected = grasp_position_cam.copy()
        grasp_position_cam_corrected[2] = grasp_z_cam

        pos_base, R_base = base_node.to_base_full(grasp_position_cam_corrected, grasp_rotation_cam)
        link7_target = pos_base + np.array([0, 0, GRASP_OFFSET_Z])

        ok, reasons = check_safety(x=pos_base[0], y=pos_base[1], z_link7=link7_target[2], width=grasp_width)

        print(f"\n계산된 목표: x={link7_target[0]:.4f}, y={link7_target[1]:.4f}, z={link7_target[2]:.4f}")

        if not ok:
            print("안전 게이트: FAIL — 중단")
            for r in reasons:
                print(f"  - {r}")
            return

        print("안전 게이트: PASS")

        # closing_axis(base_link 기준)의 방향으로 z축 회전 각도 계산
        # (수직 접근은 유지하되, 그리퍼가 벌어지는 방향만 물체에 맞춤)
        # PCA 대신 minAreaRect(사각형 전용, 45도 근처에서 더 안정적)로 각도 계산
        points_cam = np.array(result["points"])
        angle_cam = compute_rotation_angle_minarea(points_cam)
        if angle_cam is None:
            print("  minAreaRect 계산 실패, 회전 보정 없이 진행 (angle=0)")
            angle_cam = 0.0
        # 카메라 좌표계 각도를, x축 방향 벡터로 만들어서 base_link로 회전변환
        axis_dir_cam = np.array([np.cos(angle_cam), np.sin(angle_cam), 0.0])
        _, R_cam_to_base = base_node.to_base_position(axis_dir_cam)
        axis_dir_base = R_cam_to_base @ axis_dir_cam
        angle = np.arctan2(axis_dir_base[1], axis_dir_base[0])
        print(f"  그리퍼 회전 각도: {np.degrees(angle):.1f}도")

        # 목표 회전행렬을 직접 구성 (검증된 방식)
        # z축은 아래(-1)를 보고, x축(그리퍼 벌어지는 방향)은 angle 방향
        z_axis = np.array([0, 0, -1])
        x_axis = np.array([np.cos(angle), np.sin(angle), 0])
        y_axis = np.cross(z_axis, x_axis)
        R_target = np.column_stack([x_axis, y_axis, z_axis])

        from scipy.spatial.transform import Rotation
        grasp_quat = tuple(Rotation.from_matrix(R_target).as_quat())

    except Exception as e:
        print(f"오류: {e}")
        return
    finally:
        base_node.destroy_node()
        table_node.destroy_node()

    x, y, z = link7_target
    APPROACH_HEIGHT = 0.10  # [m] 물체 위 10cm 대기점
    z_approach = z + APPROACH_HEIGHT

    # ── 3. 이동 (접근(위쪽 대기점) -> 하강(실제 목표)) ──
    print("\n[5/7] Scene 등록 + 이동...")
    steps = [
        ("Scene 등록", move_node.setup_scene),
        ("접근 자세(위쪽 대기점)", lambda: move_node.move_pose(x, y, z_approach, quat=grasp_quat, label="접근 자세")),
        ("하강", lambda: move_node.move_linear(x, y, z, quat=grasp_quat, label="하강")),
    ]
    for name, action in steps:
        print(f"\n-- {name} --")
        if not action():
            print("중단됨.")
            move_node.destroy_node()
            rclpy.shutdown()
            arm.destroy_node()
            return

    # ── 4. 그리퍼 파지 ──
    print("\n[6/7] 그리퍼 파지...")
    if not move_node.confirm("그리퍼 파지"):
        print("중단됨.")
        move_node.destroy_node()
        rclpy.shutdown()
        arm.destroy_node()
        return
    move_node.gripper_pick(grasp_width)
    time.sleep(1.0)

    # ── 5. 상승 (들어올리기) ──
    print("\n[7/7] 들어올리기...")
    if not move_node.move_linear(x, y, z + LIFT, quat=grasp_quat, label="상승(들어올리기)"):
        print("들어올리기 실패.")
    else:
        print("\n" + "=" * 60)
        print("완전 자동 파지 성공!")
        print("=" * 60)

    move_node.destroy_node()
    rclpy.shutdown()
    arm.destroy_node()


if __name__ == "__main__":
    main()
