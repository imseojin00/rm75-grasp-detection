"""
[06 테스트 E] 비전 + MoveIt 파지 — 충돌 인지 계획으로 물병 집기

05의 비전 파이프라인(latch, 반지름 보정)과 D에서 검증한 MoveIt 층을 결합한다.
05와의 차이는 셋:

  1. 물병을 Scene에 등록한다 → 접근 경로가 물병을 '알고' 피한다
  2. 접근이 MoveJ_P가 아니라 MoveGroup 계획→RViz 잔상 확인→실행
  3. 진입·상승이 rm_driver MoveL이 아니라 GetCartesianPath (테이블 충돌 검사 유지)

파지 순간의 모순 처리(방법 ①):
  물병이 Scene에 있으면 그리퍼가 감싸는 것 자체가 '충돌'로 거부된다.
  → 접근까지는 물병을 Scene에 두고 회피 계획,
    진입 직전에 물병만 Scene에서 제거 (테이블·봉투는 유지).

시퀀스:
  [0] latch → 테이블+봉투+물병 Scene 등록 (RViz 확인)
  [1] MoveGroup  접근 자세 계획 → 잔상 확인 → 실행
  [2] 그리퍼 개방 (rm_driver)
  [3] 물병을 Scene에서 제거
  [4] Cartesian  파지 위치로 직선 진입 (테이블 검사는 유지)
  [5] pick_on    힘 제어 파지 (rm_driver)
  [6] Cartesian  5cm 상승
  --- 확인 후 ---
  [7] 하강 → 개방 → 직선 후퇴

전제:
  터미널1: ros2 launch rm_bringup rm_75_bringup.launch.py
  터미널2: 이 스크립트 (ROS source 후 venv activate — 비전 사용)

⚠️ RViz의 Scene 물체(테이블·봉투·물병)는 눈으로만 본다. 클릭·드래그 금지.
"""

import statistics
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO

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
from std_msgs.msg import Bool

from rclpy.qos import QoSProfile, ReliabilityPolicy

from rm_ros_interfaces.msg import Gripperpick, Gripperset

# ── 비전 (05 확정값) ─────────────────────────────────
TARGET_CLASS = "bottle"
CONF_THRESHOLD = 0.5
DEPTH_MIN, DEPTH_MAX = 0.15, 2.0
SMOOTH_WINDOW = 7
ROI_RATIO = 0.3
MAX_RADIUS = 0.06
CAMERA_FRAME = "camera_color_optical_frame"
BASE_FRAME = "base_link"
MAX_TF_AGE = 0.3
STILL_THRESHOLD = 0.005   # 검출 노이즈(±2~3mm)가 2mm 문턱을 넘나들며 MOVING/READY가 널뛰던 것 완화

# ── 파지 기하 (05 확정값) ────────────────────────────
GRASP_OFFSET = 0.098
APPROACH_BACK = 0.10
FIXED_Z = 0.142
LIFT_HEIGHT = 0.05
SIDE_Q = (0.0, 0.707, 0.0, 0.707)

# ── 그리퍼 (05 확정값) ───────────────────────────────
GRIP_SPEED = 200
GRIP_FORCE = 600
OPEN_POSITION = 1000

# ── Scene (D 확정값) ─────────────────────────────────
TABLE_Z = -0.090                          # base_link 원점에서 자로 직접 실측
GRIPPER_BOX_SIZE = (0.11, 0.13, 0.16)    # 실측 손끝 0.15 + 여유. 0.19는 물병과
GRIPPER_BOX_CENTER = (0.0, 0.0, 0.075)   # 간격 8mm뿐이라 접근 계획이 확률적으로 실패했음
TOUCH_LINKS = ["Link7", "Link6"]
BOTTLE_HEIGHT = 0.20                      # Scene용 실린더 높이 (여유 포함)
BOTTLE_RADIUS_MARGIN = 0.005              # 검출 반지름에 더할 여유

# ── MoveIt ───────────────────────────────────────────
ARM_GROUP = "rm_group"
EE_LINK = "Link7"
VEL_SCALE = 0.1
ACC_SCALE = 0.1

# ── 안전 한계 (05와 동일) ────────────────────────────
X_RANGE = (0.15, 0.55)
Y_RANGE = (-0.35, 0.35)
Z_RANGE = (-0.15, 0.45)


def make_pose(x, y, z, quat=(0.0, 0.0, 0.0, 1.0)):
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
        float(v) for v in quat
    )
    return p


class VisionMoveItGrasp(Node):
    def __init__(self):
        super().__init__("vision_moveit_grasp_test")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # MoveIt
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

        # 그리퍼 (rm_driver)
        self.pub_grip_set = self.create_publisher(
            Gripperset, "/rm_driver/set_gripper_position_cmd", 10
        )
        self.pub_grip_pick = self.create_publisher(
            Gripperpick, "/rm_driver/set_gripper_pick_on_cmd", 10
        )
        self.grip_results = {}
        for key, topic in (
            ("set", "/rm_driver/set_gripper_position_result"),
            ("pick", "/rm_driver/set_gripper_pick_on_result"),
        ):
            self.create_subscription(
                Bool, topic,
                lambda message, k=key: self.grip_results.__setitem__(
                    k, message.data
                ),
                QoSProfile(
                    depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

        # 카메라
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self.model = YOLO("yolov8n.pt")
        self.history = {"x": [], "y": [], "z": []}
        self.previous_camera_position = None
        self.still_since = None
        self.latest = None
        self.lock = threading.Lock()

    # ══ MoveIt 통신 — 폴링 대기 ══════════════════════
    # 별도 스레드의 executor가 콜백을 처리하므로,
    # spin_until_future_complete 대신 future.done()을 폴링한다.
    # (같은 노드에 executor를 두 개 붙이면 충돌한다)

    def _wait_future(self, future, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.05)
        return None

    def _call(self, client, request, timeout=10.0):
        return self._wait_future(client.call_async(request), timeout)

    def _send_goal(self, client, goal, timeout=60.0):
        handle = self._wait_future(client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            return None
        # get_result_async()의 결과는 래퍼(GetResult_Response)다.
        # 실제 액션 Result는 .result 안에 있다 — 이걸 빼먹으면
        # error_code 접근에서 AttributeError가 난다 (실기 검증 중 발견).
        result_response = self._wait_future(handle.get_result_async(), timeout)
        return None if result_response is None else result_response.result

    def confirm(self, label):
        answer = input(
            f"  ▶ [{label}] RViz 잔상을 확인했으면 Enter, 중단은 q+Enter: "
        ).strip()
        return answer.lower() != "q"

    def _execute(self, trajectory, label):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        result = self._send_goal(self.exec_cli, goal)
        ok = result is not None and result.error_code.val == 1
        self.get_logger().info(f"[실행] {label} → {'성공' if ok else '실패'}")
        return ok

    # ══ Scene ════════════════════════════════════════
    def apply_scene(self, scene):
        result = self._call(
            self.scene_cli, ApplyPlanningScene.Request(scene=scene)
        )
        return result is not None and result.success

    def setup_static_scene(self):
        """테이블 + 그리퍼 봉투 (D와 동일)."""
        scene = PlanningScene(is_diff=True)

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

        envelope = CollisionObject()
        envelope.header.frame_id = EE_LINK
        envelope.id = "gripper_envelope"
        envelope.primitives = [
            SolidPrimitive(
                type=SolidPrimitive.BOX, dimensions=list(GRIPPER_BOX_SIZE)
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

        ok = self.apply_scene(scene)
        self.get_logger().info(
            f"[scene] 테이블+봉투 등록 {'성공' if ok else '실패'}"
        )
        return ok

    def add_bottle(self, x, y, radius):
        """비전이 준 좌표로 물병 실린더 등록."""
        bottle = CollisionObject()
        bottle.header.frame_id = BASE_FRAME
        bottle.id = "bottle"
        bottle.primitives = [
            SolidPrimitive(
                type=SolidPrimitive.CYLINDER,
                dimensions=[BOTTLE_HEIGHT, radius + BOTTLE_RADIUS_MARGIN],
            )
        ]
        bottle.primitive_poses = [
            make_pose(x, y, TABLE_Z + BOTTLE_HEIGHT / 2.0)
        ]
        bottle.operation = CollisionObject.ADD

        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(bottle)
        ok = self.apply_scene(scene)
        self.get_logger().info(
            f"[scene] 물병 등록 {'성공' if ok else '실패'} "
            f"(r={radius * 1000:.0f}+{BOTTLE_RADIUS_MARGIN * 1000:.0f}mm)"
        )
        return ok

    def remove_bottle(self):
        bottle = CollisionObject()
        bottle.id = "bottle"
        bottle.header.frame_id = BASE_FRAME
        bottle.operation = CollisionObject.REMOVE

        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(bottle)
        ok = self.apply_scene(scene)
        self.get_logger().info(
            f"[scene] 물병 제거 {'성공' if ok else '실패'} — 진입 준비"
        )
        return ok

    # ══ 이동 (D와 동일) ══════════════════════════════
    def check_target(self, target):
        problems = []
        for name, value, (low, high) in (
            ("x", target[0], X_RANGE),
            ("y", target[1], Y_RANGE),
            ("z", target[2], Z_RANGE),
        ):
            if not (low <= value <= high):
                problems.append(f"{name}={value:+.4f} 범위 밖 ({low}~{high})")
        return problems

    def move_pose(self, target, label):
        problems = self.check_target(target)
        if problems:
            for problem in problems:
                print(f"    [거부] {problem}")
            return False

        x, y, z = target
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
        oc.orientation = make_pose(0, 0, 0, SIDE_Q).orientation
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
        goal.request.num_planning_attempts = 20
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = VEL_SCALE
        goal.request.max_acceleration_scaling_factor = ACC_SCALE
        goal.planning_options.plan_only = True

        # 확률적 플래너라 실패가 확률적으로 섞인다 → 최대 3회 자동 재시도.
        # 3회 모두 실패하면 운이 아니라 구조적 문제(도달 불가·충돌)로 본다.
        result = None
        for attempt in range(1, 4):
            result = self._send_goal(self.move_cli, goal)
            if result is not None and result.error_code.val == 1:
                break
            code = "None" if result is None else result.error_code.val
            self.get_logger().warn(
                f"[계획] {label} {attempt}차 실패 (error_code={code})"
                + (" — 자동 재시도" if attempt < 3 else "")
            )
        else:
            self.get_logger().error(
                f"[계획] {label} 3회 모두 실패 — 목표·Scene을 점검하세요"
            )
            return False

        self.get_logger().info(f"[계획] {label} 성공 — RViz 잔상 확인")
        if not self.confirm(label):
            return False
        return self._execute(result.planned_trajectory, label)

    def move_linear(self, target, label):
        problems = self.check_target(target)
        if problems:
            for problem in problems:
                print(f"    [거부] {problem}")
            return False

        request = GetCartesianPath.Request()
        request.header.frame_id = BASE_FRAME
        request.group_name = ARM_GROUP
        request.link_name = EE_LINK
        request.waypoints = [make_pose(*target, quat=SIDE_Q)]
        request.max_step = 0.005
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = VEL_SCALE
        request.max_acceleration_scaling_factor = ACC_SCALE

        result = self._call(self.cart_cli, request)
        fraction = 0.0 if result is None else result.fraction
        print(f"    직선 생성 비율: {fraction * 100:.0f}%")

        if fraction < 0.9:
            self.get_logger().warn(
                f"[계획] {label}: 직선 {fraction * 100:.0f}%만 생성 — 중단. "
                "(물병 제거를 건너뛰었거나 테이블과 간섭인지 확인)"
            )
            return False

        if not self.confirm(label):
            return False
        return self._execute(result.solution, label)

    # ══ 그리퍼 (05와 동일) ═══════════════════════════
    def _grip_wait(self, key, timeout=15.0):
        self.grip_results.pop(key, None)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if key in self.grip_results:
                return self.grip_results[key]
            time.sleep(0.1)
        return None

    def gripper_open(self):
        message = Gripperset()
        message.position = OPEN_POSITION
        message.block = True
        message.timeout = 10
        self.pub_grip_set.publish(message)
        result = self._grip_wait("set")
        print(f"    결과: {result}  ← 실제로 열렸는지 눈으로 확인")
        return result is True

    def gripper_pick(self):
        message = Gripperpick()
        message.speed = GRIP_SPEED
        message.force = GRIP_FORCE
        message.block = True
        message.timeout = 10
        self.pub_grip_pick.publish(message)
        result = self._grip_wait("pick")
        print(f"    결과: {result}  ← 물병이 물렸는지 눈으로 확인")
        return result is True

    # ══ 비전 (05와 동일) ═════════════════════════════
    def camera_xyz(self, color_image, depth_frame):
        results = self.model(color_image, verbose=False)[0]

        best = None
        for box in results.boxes:
            name = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            if name == TARGET_CLASS and confidence >= CONF_THRESHOLD:
                if best is None or confidence > best[1]:
                    best = (box, confidence)

        if best is None:
            self.history = {"x": [], "y": [], "z": []}
            self.previous_camera_position = None
            self.still_since = None
            return None, 0.0, color_image

        box, confidence = best
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        half_w = max(int((x2 - x1) * ROI_RATIO / 2), 2)
        half_h = max(int((y2 - y1) * ROI_RATIO / 2), 2)

        depths = []
        for v in range(cy - half_h, cy + half_h + 1, 2):
            for u in range(cx - half_w, cx + half_w + 1, 2):
                if 0 <= u < 640 and 0 <= v < 480:
                    d = depth_frame.get_distance(u, v)
                    if DEPTH_MIN < d < DEPTH_MAX:
                        depths.append(d)

        if len(depths) < 5:
            return None, 0.0, color_image

        depth = statistics.median(depths)
        point = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [cx, cy], depth
        )

        width_m = (x2 - x1) * depth / self.intrinsics.fx
        radius = min(width_m / 2.0, MAX_RADIUS)

        direction = np.array(point, dtype=float)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            point = (direction + radius * direction / norm).tolist()

        for key, value in zip(("x", "y", "z"), point):
            self.history[key].append(value)
            if len(self.history[key]) > SMOOTH_WINDOW:
                self.history[key].pop(0)

        smoothed = np.array([
            statistics.median(self.history[key]) for key in ("x", "y", "z")
        ])

        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            color_image,
            f"{TARGET_CLASS} {confidence:.2f} r={radius*1000:.0f}mm",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )
        return smoothed, radius, color_image

    def to_base(self, camera_point):
        transform = self.tf_buffer.lookup_transform(
            BASE_FRAME, CAMERA_FRAME, rclpy.time.Time()
        )
        stamp = transform.header.stamp
        age = (
            self.get_clock().now().nanoseconds * 1e-9
            - (stamp.sec + stamp.nanosec * 1e-9)
        )
        t = transform.transform.translation
        q = transform.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return rotation @ camera_point + np.array([t.x, t.y, t.z]), age

    def run(self, stop_event):
        while not stop_event.is_set():
            frames = self.align.process(self.pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            camera_point, radius, color_image = self.camera_xyz(
                color_image, depth_frame
            )

            status, color = "NO DETECTION", (0, 0, 255)

            if camera_point is not None:
                if self.previous_camera_position is not None:
                    delta = np.linalg.norm(
                        camera_point - self.previous_camera_position
                    )
                    if delta < STILL_THRESHOLD:
                        if self.still_since is None:
                            self.still_since = time.time()
                    else:
                        self.still_since = None
                self.previous_camera_position = camera_point.copy()

                try:
                    base_point, age = self.to_base(camera_point)
                except Exception:
                    base_point, age = None, None

                if base_point is None:
                    status, color = "TF WAIT", (0, 165, 255)
                elif age > MAX_TF_AGE:
                    status, color = "TF STALE", (0, 0, 255)
                elif (
                    self.still_since is None
                    or time.time() - self.still_since < 0.7
                ):
                    status, color = "MOVING", (0, 165, 255)
                else:
                    status, color = "READY", (0, 255, 0)
                    with self.lock:
                        self.latest = (base_point, radius)

            if status != "READY":
                with self.lock:
                    self.latest = None

            cv2.putText(
                color_image, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
            cv2.imshow("vision_moveit_grasp_test (E)", color_image)
            cv2.waitKey(1)

        self.pipeline.stop()
        cv2.destroyAllWindows()

    # ══ 시퀀스 ═══════════════════════════════════════
    def sequence(self):
        with self.lock:
            latest = self.latest
        if latest is None:
            print("  READY 상태가 아닙니다.")
            return

        bottle, radius = latest              # ★ latch
        front_x = bottle[0] - radius
        grasp = np.array([front_x - GRASP_OFFSET, bottle[1], FIXED_Z])
        approach = grasp - np.array([APPROACH_BACK, 0.0, 0.0])
        lift = grasp + np.array([0.0, 0.0, LIFT_HEIGHT])

        print(f"\n  물병 중심 X={bottle[0]:+.4f} Y={bottle[1]:+.4f} "
              f"(반지름 {radius * 1000:.0f}mm)")
        print(f"  접근 {approach.round(4)} / 파지 {grasp.round(4)} "
              f"/ 상승 {lift.round(4)}")
        print("  ※ 좌표 고정. 이후 물병을 움직이지 마세요.\n")

        if not self.add_bottle(bottle[0], bottle[1], radius):
            return
        print("    → RViz에서 물병 실린더 확인 (클릭 금지)")

        steps = [
            ("1) MoveGroup 접근 (물병 회피 계획)",
             lambda: self.move_pose(approach, "접근")),
            ("2) 그리퍼 개방", self.gripper_open),
            ("3) 물병 Scene 제거", self.remove_bottle),
            ("4) Cartesian 직선 진입",
             lambda: self.move_linear(grasp, "진입")),
            ("5) pick_on 파지", self.gripper_pick),
            ("6) Cartesian 5cm 상승",
             lambda: self.move_linear(lift, "상승")),
        ]

        for name, action in steps:
            answer = input(f"  {name} — Enter=실행, q=중단: ")
            if answer.strip().lower() == "q":
                print("  중단했습니다.")
                self.remove_bottle()
                return
            if not action():
                print("  단계 실패 — 시퀀스를 멈춥니다.")
                self.remove_bottle()
                return

        print("\n  ✓ 파지 완료. 물병이 들려 있는지 확인하세요.")
        answer = input("  7) 내려놓기(하강→개방→후퇴) — Enter=실행, q=유지: ")
        if answer.strip().lower() == "q":
            return

        if not self.move_linear(grasp, "하강"):
            return
        if not self.gripper_open():
            return
        input("    물병이 안정된 것을 확인 후 Enter → 후퇴: ")
        self.move_linear(approach, "후퇴")
        print("  ✓ 시퀀스 종료.")


def main():
    rclpy.init()
    node = VisionMoveItGrasp()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # MoveIt 서버 확인
    node.get_logger().info("MoveIt 서버 대기 중...")
    ok = (
        node.move_cli.wait_for_server(timeout_sec=10.0)
        and node.exec_cli.wait_for_server(timeout_sec=10.0)
        and node.scene_cli.wait_for_service(timeout_sec=10.0)
        and node.cart_cli.wait_for_service(timeout_sec=10.0)
    )
    if not ok:
        node.get_logger().error("MoveIt 인터페이스 없음 — bringup 확인")
        rclpy.shutdown()
        return

    if not node.setup_static_scene():
        node.get_logger().error("Scene 등록 실패")
        rclpy.shutdown()
        return

    stop_event = threading.Event()
    vision_thread = threading.Thread(
        target=node.run, args=(stop_event,), daemon=True
    )
    vision_thread.start()
    time.sleep(2.0)

    print("\n" + "=" * 60)
    print("  [06 테스트 E] 비전 + MoveIt 파지")
    print("  ⚠️ RViz Scene 물체는 눈으로만 — 클릭·드래그 금지")
    print("=" * 60)
    print("\n영상 창이 READY(초록)일 때 Enter. 종료: q + Enter\n")

    try:
        while True:
            key = input("Enter=시퀀스 시작, q=종료: ")
            if key.strip().lower() == "q":
                break
            node.sequence()
    except (KeyboardInterrupt, EOFError):
        pass

    stop_event.set()
    vision_thread.join(timeout=2.0)
    executor.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
