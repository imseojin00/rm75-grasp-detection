"""
[06 테스트 D] MoveIt 프리미티브 단독 검증 — 비전 없음, 알려진 좌표만

real_pick_place.py의 MoveIt 조작(계획→잔상 확인→실행, Cartesian 직선,
Scene 등록)이 실기체에서 실제로 동작하는지를, 05에서 검증된 측면 파지
자세와 좌표로 확인한다. 비전과 그리퍼는 이 단계에서 쓰지 않는다.

이번에 처음 확인하는 것들 (전부 미검증):
  1. bringup의 /move_action 이 실물을 움직이는가 (real_pick_place.py는 미검증 코드였다)
  2. 측면 파지 자세 (0, 0.707, 0, 0.707) 로 MoveGroup 계획이 풀리는가
  3. Scene에 박스(테이블 + 그리퍼 부착물)가 있는 상태에서 200 Hz RViz가 버티는가
     ⚠️ RViz의 Scene 표시는 눈으로만 본다 — 절대 클릭·드래그하지 않는다 (크래시)
  4. GetCartesianPath 직선이 어느 비율(fraction)로 생성되는가

시퀀스:
  [0] 테이블 + 그리퍼·카메라 박스(Link7 부착) Scene 등록 → RViz에서 눈으로 확인
  [1] MoveGroup: 접근 자세로 계획 → RViz 잔상 확인 → Enter → 실행
  [2] Cartesian: +8 cm 전진 (직선) → 잔상 확인 → 실행
  [3] Cartesian: −8 cm 후퇴 → 잔상 확인 → 실행

전제:
  터미널1: ros2 launch rm_bringup rm_75_bringup.launch.py
  터미널2: 이 스크립트 (venv 불필요 — 표준 ROS 패키지만 사용)
           source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
"""

import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from shape_msgs.msg import SolidPrimitive

# ── 실측값 (이 환경 기준 — 다른 환경이면 재실측) ───────────
TABLE_Z = -0.120         # 테이블(받침대) 면의 base_link z — 원점에서 자로 직접 실측.
                          # 접촉 역산(0.171 - 손끝길이)은 손끝 길이가 개폐 상태에 따라
                          # 변해서(열림 0.15 / 뻗음 0.18) 오차가 났다. 직접 측정이 이긴다.

# ── 05에서 검증된 파지 기하 ───────────────────────────────
SIDE_Q = (0.0, 0.707, 0.0, 0.707)   # 측면 파지 (그리퍼가 +x)
TOPDOWN_Q = (0.0, 1.0, 0.0, 0.0)   # 수직 파지 (2026-08-21 실측)
TEST_POSE = (0.351, 0.004, 0.063)    # [m] 2026-08-21 수직 파지 실측
FORWARD = 0.08                       # Cartesian 전진량
LIFT = 0.05                          # [m] 수직 상승량

# ── MoveIt 설정 (real_pick_place.py와 동일) ───────────────
BASE_FRAME = "base_link"
ARM_GROUP = "rm_group"
EE_LINK = "Link7"
VEL_SCALE = 0.1
ACC_SCALE = 0.1

# ── 그리퍼+카메라 부착 박스 (Link7 프레임 기준) ────────────
# bringup 모델에는 그리퍼·카메라가 없다 → MoveIt이 그 부피의 충돌을 모른다.
# Link7에 여유 있는 박스를 '부착'해 계획이 이 부피까지 피하게 만든다.
#   - 그리퍼: 툴 z 방향으로 약 0.15 m 뻗음
#   - 카메라+브래킷: 툴 -x 쪽 상부에 돌출
# 정밀 모델이 아니라 '안전 봉투(envelope)'다. 크기는 실물보다 조금 크게.
GRIPPER_BOX_SIZE = (0.11, 0.13, 0.18)    # x, y, z (Link7 프레임)
GRIPPER_BOX_CENTER = (0.0, 0.0, 0.09)    # Link7 원점에서 툴 z로 9 cm 지점이 중심
TOUCH_LINKS = ["Link7", "Link6", "Link5", "Link4", "Link3", "Link2", "Link1", "base_link"]  # 봉투 확대(0.18)로 확장 — 2026-08-21


def make_pose(x, y, z, quat=(0.0, 0.0, 0.0, 1.0)):
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
        float(v) for v in quat
    )
    return p


class MoveItSideTest(Node):
    def __init__(self):
        super().__init__("moveit_side_test")

        self.move_cli = ActionClient(self, MoveGroup, "/move_action")
        self.exec_cli = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory"
        )
        self.scene_cli = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.cart_cli = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )

        self.get_logger().info("서버 대기 중...")
        ok = (
            self.move_cli.wait_for_server(timeout_sec=10.0)
            and self.exec_cli.wait_for_server(timeout_sec=10.0)
            and self.scene_cli.wait_for_service(timeout_sec=10.0)
            and self.cart_cli.wait_for_service(timeout_sec=10.0)
        )
        if not ok:
            self.get_logger().error(
                "MoveIt 인터페이스를 찾지 못했습니다. bringup이 떠 있나요?"
            )
            sys.exit(1)
        self.get_logger().info("서버 연결 완료")

    # ── 공용 유틸 (real_pick_place.py 패턴) ──
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
        answer = input(
            f"  ▶ [{label}] RViz 잔상을 확인했으면 Enter, 중단은 q+Enter: "
        ).strip()
        if answer.lower() == "q":
            self.get_logger().warn("사용자 중단")
            return False
        return True

    def _execute(self, trajectory, label):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        result = self._send_goal(self.exec_cli, goal)
        ok = result is not None and result.error_code.val == 1
        self.get_logger().info(f"[실행] {label} → {'성공' if ok else '실패'}")
        return ok

    # ── Scene ──
    def setup_scene(self):
        scene = PlanningScene(is_diff=True)

        # 테이블 — 면이 TABLE_Z에 오도록 두께 2 cm 박스를 그 아래에 배치
        table = CollisionObject()
        table.header.frame_id = BASE_FRAME
        table.id = "table"
        table.primitives = [
            SolidPrimitive(
                type=SolidPrimitive.BOX, dimensions=[1.2, 1.2, 0.02]
            )
        ]
        table.primitive_poses = [make_pose(0.45, 0.0, TABLE_Z - 0.011)]
        table.operation = CollisionObject.ADD
        scene.world.collision_objects.append(table)

        # 그리퍼+카메라 봉투 — Link7에 부착
        envelope = CollisionObject()
        envelope.header.frame_id = EE_LINK
        envelope.id = "gripper_envelope"
        envelope.primitives = [
            SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=list(GRIPPER_BOX_SIZE),
            )
        ]
        envelope.primitive_poses = [make_pose(*GRIPPER_BOX_CENTER)]
        envelope.operation = CollisionObject.ADD

        attached = AttachedCollisionObject()
        attached.link_name = EE_LINK
        attached.object = envelope
        attached.touch_links = TOUCH_LINKS
        scene.robot_state.attached_collision_objects.append(attached)
        scene.robot_state.is_diff = True

        result = self._call(
            self.scene_cli, ApplyPlanningScene.Request(scene=scene)
        )
        ok = result is not None and result.success
        self.get_logger().info(
            f"[scene] 테이블 + 그리퍼 봉투 등록 {'성공' if ok else '실패'}"
        )
        return ok

    # ── 이동 ──
    def move_pose(self, x, y, z, quat=TOPDOWN_Q, label=None):
        label = label or f"pose ({x:.3f}, {y:.3f}, {z:.3f})"

        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = EE_LINK
        pc.target_point_offset = Vector3()
        pc.constraint_region.primitives = [
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])
        ]
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
        goal.request.goal_constraints = [
            Constraints(
                position_constraints=[pc], orientation_constraints=[oc]
            )
        ]
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = VEL_SCALE
        goal.request.max_acceleration_scaling_factor = ACC_SCALE
        goal.planning_options.plan_only = True

        result = self._send_goal(self.move_cli, goal)
        if result is None or result.error_code.val != 1:
            code = "None" if result is None else result.error_code.val
        else:
            code = None

        if code is not None:
            self.get_logger().error(
                f"[계획] {label} 실패 (error_code={code}) — "
                "도달 불가·자세·충돌(Scene) 세 갈래를 점검"
            )
            return False

        self.get_logger().info(f"[계획] {label} 성공 — RViz 잔상 확인")
        if not self.confirm(label):
            return False
        return self._execute(result.planned_trajectory, label)

    def move_linear(self, x, y, z, quat=TOPDOWN_Q, label=None):
        label = label or f"직선 ({x:.3f}, {y:.3f}, {z:.3f})"

        request = GetCartesianPath.Request()
        request.header.frame_id = BASE_FRAME
        request.group_name = ARM_GROUP
        request.link_name = EE_LINK
        request.waypoints = [make_pose(x, y, z, quat)]
        request.max_step = 0.005
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = VEL_SCALE
        request.max_acceleration_scaling_factor = ACC_SCALE

        result = self._call(self.cart_cli, request)
        fraction = 0.0 if result is None else result.fraction

        print(f"    직선 생성 비율: {fraction * 100:.0f}%")
        if fraction < 0.9:
            self.get_logger().warn(
                f"[계획] {label}: 직선이 {fraction * 100:.0f}%만 생성됨 — 중단"
            )
            return False

        if not self.confirm(label):
            return False
        return self._execute(result.solution, label)


def main():
    rclpy.init()
    node = MoveItSideTest()

    print("\n" + "=" * 60)
    print("  [06 테스트 D] MoveIt 프리미티브 검증 (비전 없음)")
    print("  ⚠️ RViz의 Scene 박스는 눈으로만 — 클릭·드래그 금지")
    print("=" * 60 + "\n")

    x, y, z = TEST_POSE

    steps = [
        ("Scene 등록", node.setup_scene),
        ("1) 접근 자세로 계획·실행",
         lambda: node.move_pose(x, y, z, label="접근 자세")),
        ("2) +5cm 상승",
         lambda: node.move_linear(x, y, z + LIFT, label="상승")),
        ("3) -5cm 하강",
         lambda: node.move_linear(x, y, z, label="하강")),
    ]

    for name, action in steps:
        print(f"\n── {name} ──")
        if not action():
            print("\n중단됨. RViz와 터미널1 로그를 확인하세요.")
            break
    else:
        print("\n✓ D 테스트 완료 — MoveIt 프리미티브 전부 동작 확인")
        print("  관찰 기록: RViz 생존 여부 / 직선 fraction / 계획 소요 시간")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
