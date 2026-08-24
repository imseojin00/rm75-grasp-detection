"""
[06 테스트 D] MoveIt 프리미티브 단독 검증 — 비전 없음

목적:
  - 실제 /joint_states를 받아 MoveIt 시작 상태(start_state)에 명시적으로 넣는다.
  - TEST_POSE + TOPDOWN_Q가 MoveIt에서 계획 가능한지 확인한다.
  - Scene에 테이블 + 그리퍼/카메라 envelope을 등록한다.
  - 기본값 EXECUTE=False로 실제 로봇은 움직이지 않는다.

실행 순서:
  [0] Scene 등록
  [1] 접근 자세 계획
  [2] +5 cm Cartesian 계획
  [3] -5 cm Cartesian 계획

주의:
  EXECUTE = False  -> 계획만 수행, 실제 로봇 이동 없음
  EXECUTE = True   -> Enter 이후 실제 trajectory 실행
"""

import sys
import copy

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Vector3

from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
    RobotState,
)

from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from shape_msgs.msg import SolidPrimitive


# ============================================================
# 안전 설정
# ============================================================

# False = 계획만 확인하고 실제 로봇은 움직이지 않음
# True  = Enter 후 실제 trajectory 실행
EXECUTE = False


# ============================================================
# 실측값
# ============================================================

TABLE_Z = -0.0879

# 05에서 검증된 파지 기하
SIDE_Q = (0.0, 0.707, 0.0, 0.707)

# 2026-08-21 실측한 수직 파지 자세
TOPDOWN_Q = (0.0, 1.0, 0.0, 0.0)

# 테스트 위치
TEST_POSE = (0.371142, 0.042051, 0.077117)

FORWARD = 0.08
LIFT = 0.05


# ============================================================
# MoveIt 설정
# ============================================================

BASE_FRAME = "base_link"
ARM_GROUP = "rm_group"
EE_LINK = "Link7"

VEL_SCALE = 0.1
ACC_SCALE = 0.1


# ============================================================
# 그리퍼 + 카메라 충돌 envelope
# ============================================================

GRIPPER_BOX_SIZE = (0.11, 0.13, 0.18)

GRIPPER_BOX_CENTER = (0.0, 0.0, 0.09)

TOUCH_LINKS = [
    "Link7",
    "Link6",
    "Link5",
    "Link4",
    "Link3",
    "Link2",
    "Link1",
    "base_link",
]


# ============================================================
# Pose 생성
# ============================================================

def make_pose(x, y, z, quat=(0.0, 0.0, 0.0, 1.0)):
    p = Pose()

    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)

    (
        p.orientation.x,
        p.orientation.y,
        p.orientation.z,
        p.orientation.w,
    ) = (float(v) for v in quat)

    return p


# ============================================================
# MoveIt 테스트 Node
# ============================================================

class MoveItSideTest(Node):

    def __init__(self):
        super().__init__("moveit_side_test")

        # ----------------------------------------------------
        # 현재 관절 상태 저장
        # ----------------------------------------------------

        self.current_joint_state = None

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )

        # ----------------------------------------------------
        # MoveIt 액션 / 서비스
        # ----------------------------------------------------

        self.move_cli = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

        self.exec_cli = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

        self.scene_cli = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.cart_cli = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )

        # ----------------------------------------------------
        # 서버 대기
        # ----------------------------------------------------

        self.get_logger().info("서버 대기 중...")

        ok = (
            self.move_cli.wait_for_server(timeout_sec=10.0)
            and self.exec_cli.wait_for_server(timeout_sec=10.0)
            and self.scene_cli.wait_for_service(timeout_sec=10.0)
            and self.cart_cli.wait_for_service(timeout_sec=10.0)
        )

        if not ok:
            self.get_logger().error(
                "MoveIt 인터페이스를 찾지 못했습니다. "
                "bringup이 떠 있나요?"
            )
            sys.exit(1)

        self.get_logger().info("서버 연결 완료")

        # ----------------------------------------------------
        # 현재 JointState가 들어올 때까지 대기
        # ----------------------------------------------------

        self.get_logger().info(
            "/joint_states 수신 대기 중..."
        )

        for _ in range(50):
            if self.current_joint_state is not None:
                break

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        if self.current_joint_state is None:
            self.get_logger().error(
                "/joint_states를 받지 못했습니다."
            )
            sys.exit(1)

        self.get_logger().info(
            "현재 /joint_states 수신 완료"
        )

        self._print_current_joint_state()

    # ========================================================
    # JointState callback
    # ========================================================

    def _joint_state_callback(self, msg):
        self.current_joint_state = copy.deepcopy(msg)

    # ========================================================
    # 현재 관절값 출력
    # ========================================================

    def _print_current_joint_state(self):

        msg = self.current_joint_state

        self.get_logger().info(
            "현재 관절 상태:"
        )

        for name, pos in zip(msg.name, msg.position):
            self.get_logger().info(
                f"  {name}: {pos:.6f} rad"
            )

    # ========================================================
    # 현재 RobotState 생성
    # ========================================================

    def _get_current_robot_state(self):

        if self.current_joint_state is None:
            return None

        state = RobotState()

        state.joint_state = copy.deepcopy(
            self.current_joint_state
        )

        return state

    # ========================================================
    # Service 호출
    # ========================================================

    def _call(self, client, request):

        future = client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        return future.result()

    # ========================================================
    # Action 호출
    # ========================================================

    def _send_goal(self, client, goal):

        future = client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        handle = future.result()

        if handle is None:
            return None

        if not handle.accepted:
            return None

        result_future = handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        return result_future.result().result

    # ========================================================
    # 실행 전 확인
    # ========================================================

    def confirm(self, label):

        if not EXECUTE:
            print(
                f"  ▶ [{label}] 계획 확인 완료 "
                f"(EXECUTE=False → 실제 로봇 이동 안 함)"
            )
            return True

        answer = input(
            f"  ▶ [{label}] RViz 잔상을 확인했으면 "
            f"Enter, 중단은 q+Enter: "
        ).strip()

        if answer.lower() == "q":

            self.get_logger().warn(
                "사용자 중단"
            )

            return False

        return True

    # ========================================================
    # trajectory 실행
    # ========================================================

    def _execute(self, trajectory, label):

        # ----------------------------------------------------
        # 안전 모드
        # ----------------------------------------------------

        if not EXECUTE:

            self.get_logger().info(
                f"[실행 생략] {label} "
                f"→ EXECUTE=False"
            )

            return True

        # ----------------------------------------------------
        # 실제 실행
        # ----------------------------------------------------

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = trajectory

        result = self._send_goal(
            self.exec_cli,
            goal,
        )

        ok = (
            result is not None
            and result.error_code.val == 1
        )

        self.get_logger().info(
            f"[실행] {label} → "
            f"{'성공' if ok else '실패'}"
        )

        return ok

    # ========================================================
    # Planning Scene 등록
    # ========================================================

    def setup_scene(self):

        scene = PlanningScene(
            is_diff=True
        )

        # ----------------------------------------------------
        # 테이블
        # ----------------------------------------------------

        table = CollisionObject()

        table.header.frame_id = BASE_FRAME

        table.id = "table"

        table.primitives = [
            SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=[
                    1.2,
                    1.2,
                    0.02,
                ],
            )
        ]

        table.primitive_poses = [
            make_pose(
                0.45,
                0.0,
                TABLE_Z - 0.011,
            )
        ]

        table.operation = CollisionObject.ADD

        scene.world.collision_objects.append(
            table
        )

        # ----------------------------------------------------
        # 그리퍼 + 카메라 envelope
        # ----------------------------------------------------

        envelope = CollisionObject()

        envelope.header.frame_id = EE_LINK

        envelope.id = "gripper_envelope"

        envelope.primitives = [
            SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=list(
                    GRIPPER_BOX_SIZE
                ),
            )
        ]

        envelope.primitive_poses = [
            make_pose(
                *GRIPPER_BOX_CENTER
            )
        ]

        envelope.operation = CollisionObject.ADD

        attached = AttachedCollisionObject()

        attached.link_name = EE_LINK

        attached.object = envelope

        attached.touch_links = TOUCH_LINKS

        scene.robot_state.attached_collision_objects.append(
            attached
        )

        scene.robot_state.is_diff = True

        # ----------------------------------------------------
        # Scene 적용
        # ----------------------------------------------------

        result = self._call(
            self.scene_cli,
            ApplyPlanningScene.Request(
                scene=scene
            ),
        )

        ok = (
            result is not None
            and result.success
        )

        self.get_logger().info(
            "[scene] 테이블 + 그리퍼 봉투 등록 "
            f"{'성공' if ok else '실패'}"
        )

        return ok

    # ========================================================
    # MoveGroup pose 계획
    # ========================================================

    def move_pose(
        self,
        x,
        y,
        z,
        quat=TOPDOWN_Q,
        label=None,
    ):

        label = (
            label
            or f"pose ({x:.3f}, {y:.3f}, {z:.3f})"
        )

        # ----------------------------------------------------
        # Position Constraint
        # ----------------------------------------------------

        pc = PositionConstraint()

        pc.header.frame_id = BASE_FRAME

        pc.link_name = EE_LINK

        pc.target_point_offset = Vector3()

        pc.constraint_region.primitives = [
            SolidPrimitive(
                type=SolidPrimitive.SPHERE,
                dimensions=[0.01],
            )
        ]

        pc.constraint_region.primitive_poses = [
            make_pose(
                x,
                y,
                z,
            )
        ]

        pc.weight = 1.0

        # ----------------------------------------------------
        # Orientation Constraint
        # ----------------------------------------------------

        oc = OrientationConstraint()

        oc.header.frame_id = BASE_FRAME

        oc.link_name = EE_LINK

        oc.orientation = make_pose(
            0,
            0,
            0,
            quat,
        ).orientation

        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1

        oc.weight = 1.0

        # ----------------------------------------------------
        # MoveGroup Goal
        # ----------------------------------------------------

        goal = MoveGroup.Goal()

        goal.request.group_name = ARM_GROUP

        # ⭐ 핵심 수정
        # 현재 실제 관절 상태를 MoveIt 시작 상태로 전달
        current_state = self._get_current_robot_state()

        if current_state is None:
            self.get_logger().error(
                "현재 RobotState를 만들 수 없습니다."
            )
            return False

        goal.request.start_state = current_state

        # ----------------------------------------------------
        # 목표 constraint
        # ----------------------------------------------------

        goal.request.goal_constraints = [
            Constraints(
                position_constraints=[pc],
                orientation_constraints=[oc],
            )
        ]

        goal.request.num_planning_attempts = 10

        goal.request.allowed_planning_time = 5.0

        goal.request.max_velocity_scaling_factor = (
            VEL_SCALE
        )

        goal.request.max_acceleration_scaling_factor = (
            ACC_SCALE
        )

        # 계획만 먼저 수행
        goal.planning_options.plan_only = True

        # ----------------------------------------------------
        # 계획 요청
        # ----------------------------------------------------

        self.get_logger().info(
            f"[계획 요청] {label}"
        )

        result = self._send_goal(
            self.move_cli,
            goal,
        )

        if (
            result is None
            or result.error_code.val != 1
        ):

            code = (
                "None"
                if result is None
                else result.error_code.val
            )

            self.get_logger().error(
                f"[계획] {label} 실패 "
                f"(error_code={code})"
            )

            return False

        # ----------------------------------------------------
        # 성공
        # ----------------------------------------------------

        self.get_logger().info(
            f"[계획] {label} 성공"
        )

        self.get_logger().info(
            "MoveIt이 유효한 trajectory를 생성했습니다."
        )

        # 실제 실행 여부 확인
        if not self.confirm(label):
            return False

        return self._execute(
            result.planned_trajectory,
            label,
        )

    # ========================================================
    # Cartesian Path
    # ========================================================

    def move_linear(
        self,
        x,
        y,
        z,
        quat=TOPDOWN_Q,
        label=None,
    ):

        label = (
            label
            or f"직선 ({x:.3f}, {y:.3f}, {z:.3f})"
        )

        # ----------------------------------------------------
        # Cartesian request
        # ----------------------------------------------------

        request = GetCartesianPath.Request()

        request.header.frame_id = BASE_FRAME

        request.group_name = ARM_GROUP

        request.link_name = EE_LINK

        # ⭐ 현재 관절 상태를 시작 상태로 명시
        current_state = self._get_current_robot_state()

        if current_state is None:
            self.get_logger().error(
                "현재 RobotState를 만들 수 없습니다."
            )
            return False

        request.start_state = current_state

        request.waypoints = [
            make_pose(
                x,
                y,
                z,
                quat,
            )
        ]

        request.max_step = 0.005

        request.avoid_collisions = True

        request.max_velocity_scaling_factor = (
            VEL_SCALE
        )

        request.max_acceleration_scaling_factor = (
            ACC_SCALE
        )

        # ----------------------------------------------------
        # Cartesian 계산
        # ----------------------------------------------------

        result = self._call(
            self.cart_cli,
            request,
        )

        if result is None:
            self.get_logger().error(
                f"[계획] {label}: "
                "Cartesian 결과 없음"
            )
            return False

        fraction = result.fraction

        print(
            f"    직선 생성 비율: "
            f"{fraction * 100:.1f}%"
        )

        # ----------------------------------------------------
        # fraction 검사
        # ----------------------------------------------------

        if fraction < 0.9:

            self.get_logger().warn(
                f"[계획] {label}: "
                f"직선이 {fraction * 100:.1f}%만 생성됨 "
                "→ 중단"
            )

            return False

        self.get_logger().info(
            f"[계획] {label} Cartesian 성공"
        )

        # ----------------------------------------------------
        # 실행
        # ----------------------------------------------------

        if not self.confirm(label):
            return False

        return self._execute(
            result.solution,
            label,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = MoveItSideTest()

    print(
        "\n"
        + "=" * 60
    )

    print(
        "  [06 테스트 D] "
        "MoveIt 프리미티브 검증"
    )

    print(
        f"  EXECUTE = {EXECUTE}"
    )

    if EXECUTE:
        print(
            "  ⚠️ 실제 trajectory를 실행합니다."
        )
    else:
        print(
            "  ✅ 계획만 수행합니다. "
            "실제 로봇은 움직이지 않습니다."
        )

    print(
        "  RViz Scene 박스는 눈으로만 확인"
    )

    print(
        "=" * 60
        + "\n"
    )

    x, y, z = TEST_POSE

    steps = [
        (
            "Scene 등록",
            node.setup_scene,
        ),

        (
            "1) 접근 자세 계획",
            lambda: node.move_pose(
                x,
                y,
                z,
                label="접근 자세",
            ),
        ),

        (
            "2) +5cm 상승 계획",
            lambda: node.move_linear(
                x,
                y,
                z + LIFT,
                label="상승",
            ),
        ),

        (
            "3) -5cm 하강 계획",
            lambda: node.move_linear(
                x,
                y,
                z,
                label="하강",
            ),
        ),
    ]

    for name, action in steps:

        print(
            f"\n── {name} ──"
        )

        if not action():

            print(
                "\n중단됨. "
                "RViz와 MoveIt 로그를 확인하세요."
            )

            break

    else:

        print(
            "\n✓ D 테스트 완료"
        )

        print(
            "  MoveIt 계획 / Cartesian 검증 완료"
        )

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
