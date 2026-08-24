#!/usr/bin/env python3

import time
import copy

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    RobotState,
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
)

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Vector3
from shape_msgs.msg import SolidPrimitive


# ============================================================
# 기본 설정
# ============================================================

ARM_GROUP = "rm_group"
EE_LINK = "Link7"
BASE_FRAME = "base_link"

VEL_SCALE = 0.1
ACC_SCALE = 0.1


# ============================================================
# 우리가 이전에 검증한 TEST POSE
# ============================================================

TEST_POSE = (
    0.371142,
    0.042051,
    0.077117,
)

TOPDOWN_Q = (
    0.0,
    1.0,
    0.0,
    0.0,
)


# ============================================================
# Grasp Planner
# ============================================================

class GraspPlanner(Node):

    def __init__(self):

        super().__init__("grasp_planner")

        # ====================================================
        # 현재 joint state
        # ====================================================

        self.current_joint_state = None

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )

        # ====================================================
        # IK service
        # ====================================================

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
        )

        self.get_logger().info(
            "Waiting for MoveIt IK service..."
        )

        while not self.ik_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                "Waiting for /compute_ik..."
            )

        self.get_logger().info(
            "MoveIt IK service connected."
        )

        # ====================================================
        # MoveGroup
        # ====================================================

        self.move_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

        self.get_logger().info(
            "Waiting for MoveIt /move_action..."
        )

        while not self.move_client.wait_for_server(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                "Waiting for /move_action..."
            )

        self.get_logger().info(
            "MoveIt /move_action connected."
        )

        # ====================================================
        # ExecuteTrajectory
        # ====================================================

        self.exec_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

        self.get_logger().info(
            "Waiting for MoveIt /execute_trajectory..."
        )

        while not self.exec_client.wait_for_server(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                "Waiting for /execute_trajectory..."
            )

        self.get_logger().info(
            "MoveIt /execute_trajectory connected."
        )


    # ========================================================
    # /joint_states callback
    # ========================================================

    def _joint_state_callback(self, msg):

        self.current_joint_state = copy.deepcopy(msg)


    # ========================================================
    # 현재 joint state
    # ========================================================

    def get_current_joint_state(self):

        start = time.time()

        while (
            self.current_joint_state is None
            and time.time() - start < 3.0
        ):

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        if self.current_joint_state is None:
            return None

        return copy.deepcopy(
            self.current_joint_state
        )


    # ========================================================
    # Pose 생성
    # ========================================================

    def make_pose(
        self,
        x,
        y,
        z,
        quat,
    ):

        pose = Pose()

        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        return pose


    # ========================================================
    # TEST_POSE 계획
    # ========================================================

    def plan_test_pose(self):

        x, y, z = TEST_POSE

        target_pose = self.make_pose(
            x,
            y,
            z,
            TOPDOWN_Q,
        )

        # ----------------------------------------------------
        # 현재 상태
        # ----------------------------------------------------

        current = self.get_current_joint_state()

        if current is None:

            self.get_logger().error(
                "Could not get current joint state."
            )

            return None

        start_state = RobotState()

        start_state.joint_state = (
            copy.deepcopy(current)
        )

        # ----------------------------------------------------
        # Position constraint
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
            target_pose
        ]

        pc.weight = 1.0

        # ----------------------------------------------------
        # Orientation constraint
        # ----------------------------------------------------

        oc = OrientationConstraint()

        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK

        oc.orientation = (
            target_pose.orientation
        )

        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1

        oc.weight = 1.0

        # ----------------------------------------------------
        # MoveGroup
        # ----------------------------------------------------

        goal = MoveGroup.Goal()

        goal.request.group_name = ARM_GROUP

        goal.request.start_state = start_state

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

        # ----------------------------------------------------
        # TEST_POSE는 계획만 먼저
        # ----------------------------------------------------

        goal.planning_options.plan_only = True

        self.get_logger().info("")
        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "TEST_POSE PLANNING"
        )
        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            f"Position: "
            f"[{x:.6f}, {y:.6f}, {z:.6f}]"
        )

        self.get_logger().info(
            f"Quaternion: "
            f"[{TOPDOWN_Q[0]:.6f}, "
            f"{TOPDOWN_Q[1]:.6f}, "
            f"{TOPDOWN_Q[2]:.6f}, "
            f"{TOPDOWN_Q[3]:.6f}]"
        )

        # ----------------------------------------------------
        # Goal 전송
        # ----------------------------------------------------

        future = self.move_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        handle = future.result()

        if handle is None:

            self.get_logger().error(
                "Failed to send TEST_POSE goal."
            )

            return None

        if not handle.accepted:

            self.get_logger().error(
                "TEST_POSE goal rejected."
            )

            return None

        result_future = (
            handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        result = (
            result_future.result().result
        )

        error_code = result.error_code.val

        self.get_logger().info(
            f"TEST_POSE MoveIt error code: "
            f"{error_code}"
        )

        if error_code != 1:

            self.get_logger().error(
                "TEST_POSE planning FAILED."
            )

            return None

        trajectory = (
            result.planned_trajectory
        )

        joint_trajectory = (
            trajectory.joint_trajectory
        )

        self.get_logger().info(
            "TEST_POSE planning SUCCESS!"
        )

        self.get_logger().info(
            f"Trajectory points: "
            f"{len(joint_trajectory.points)}"
        )

        # ====================================================
        # ★ TEST_POSE 최종 joint 출력
        # ====================================================

        if len(joint_trajectory.points) > 0:

            final = joint_trajectory.points[-1]

            self.get_logger().info(
                "TEST_POSE final planned joint position:"
            )

            for name, value in zip(
                joint_trajectory.joint_names,
                final.positions,
            ):

                self.get_logger().info(
                    f"  {name}: {value:.6f} rad"
                )

        return trajectory


    # ========================================================
    # TEST_POSE 실제 실행
    # ========================================================

    def execute_test_pose(
        self,
        trajectory,
    ):

        x, y, z = TEST_POSE

        print("")
        print("========================================")
        print("TEST_POSE 실제 이동 확인")
        print("========================================")

        print(
            f"x = {x:.6f}"
        )

        print(
            f"y = {y:.6f}"
        )

        print(
            f"z = {z:.6f}"
        )

        print(
            "q = "
            f"[{TOPDOWN_Q[0]:.6f}, "
            f"{TOPDOWN_Q[1]:.6f}, "
            f"{TOPDOWN_Q[2]:.6f}, "
            f"{TOPDOWN_Q[3]:.6f}]"
        )

        print("")
        print(
            "실제 로봇이 TEST_POSE로 이동합니다."
        )
        print(
            "주변에 장애물이 없는지 확인하세요."
        )
        print("")
        print(
            "계속하려면 Enter"
        )
        print(
            "중단하려면 q + Enter"
        )

        answer = input(
            ">>> "
        ).strip()

        if answer.lower() == "q":

            self.get_logger().warn(
                "TEST_POSE execution cancelled."
            )

            return False

        # ----------------------------------------------------
        # ExecuteTrajectory
        # ----------------------------------------------------

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = trajectory

        self.get_logger().info(
            "Executing TEST_POSE..."
        )

        future = (
            self.exec_client.send_goal_async(
                goal
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        handle = future.result()

        if handle is None:

            self.get_logger().error(
                "Failed to send execution goal."
            )

            return False

        if not handle.accepted:

            self.get_logger().error(
                "TEST_POSE execution rejected."
            )

            return False

        result_future = (
            handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        result = (
            result_future.result().result
        )

        error_code = result.error_code.val

        self.get_logger().info(
            f"TEST_POSE execution error code: "
            f"{error_code}"
        )

        if error_code != 1:

            self.get_logger().error(
                "TEST_POSE execution FAILED."
            )

            return False

        self.get_logger().info(
            "TEST_POSE execution SUCCESS!"
        )

        # ----------------------------------------------------
        # 실제 joint state 확인
        # ----------------------------------------------------

        time.sleep(0.5)

        actual = self.get_current_joint_state()

        if actual is not None:

            self.get_logger().info(
                "Actual /joint_states after TEST_POSE:"
            )

            for name, value in zip(
                actual.name,
                actual.position,
            ):

                if name.startswith("joint"):

                    self.get_logger().info(
                        f"  {name}: "
                        f"{value:.6f} rad"
                    )

        self.get_logger().info(
            "Robot reached TEST_POSE."
        )

        return True


    # ========================================================
    # Grasp Pose
    # ========================================================

    def get_grasp_pose(self):

        # ----------------------------------------------------
        # 현재는 임시 hard-coded grasp pose
        #
        # TODO:
        # Camera -> base_link 좌표변환 결과로 교체
        # ----------------------------------------------------

        position = [
            0.42415417,
            0.10695407,
            -0.06000401       
        ]

        quaternion = [
            0.30867291,
            0.78475241,
            0.40639081,
            0.35175445,
        ]

        return self.make_pose(
            position[0],
            position[1],
            position[2],
            quaternion,
        )


    # ========================================================
    # IK
    # ========================================================

    def solve_ik(
        self,
        pose,
    ):

        request = GetPositionIK.Request()

        request.ik_request.group_name = (
            ARM_GROUP
        )

        request.ik_request.ik_link_name = (
            EE_LINK
        )

        request.ik_request.pose_stamped.header.frame_id = (
            BASE_FRAME
        )

        request.ik_request.pose_stamped.pose = (
            pose
        )

        # ----------------------------------------------------
        # 현재 joint state를 IK seed로 사용
        # ----------------------------------------------------

        current = self.get_current_joint_state()

        if current is not None:

            state = RobotState()

            state.joint_state = (
                copy.deepcopy(current)
            )

            request.ik_request.robot_state = (
                state
            )

        self.get_logger().info("")
        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "GRASP IK"
        )
        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            f"Position: "
            f"[{pose.position.x:.6f}, "
            f"{pose.position.y:.6f}, "
            f"{pose.position.z:.6f}]"
        )

        self.get_logger().info(
            f"Quaternion: "
            f"[{pose.orientation.x:.6f}, "
            f"{pose.orientation.y:.6f}, "
            f"{pose.orientation.z:.6f}, "
            f"{pose.orientation.w:.6f}]"
        )

        future = self.ik_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:

            self.get_logger().error(
                "IK service failed."
            )

            return None

        self.get_logger().info(
            f"IK error code: "
            f"{response.error_code.val}"
        )

        if response.error_code.val != 1:

            self.get_logger().error(
                "IK solution NOT found."
            )

            return None

        self.get_logger().info(
            "IK solution found!"
        )

        joints = (
            response.solution.joint_state
        )

        self.get_logger().info(
            "Joint solution:"
        )

        for name, value in zip(
            joints.name,
            joints.position,
        ):

            if name.startswith("joint"):

                self.get_logger().info(
                    f"  {name}: "
                    f"{value:.6f} rad"
                )

        return joints


    # ========================================================
    # Grasp Pose Motion Planning
    #
    # ★ 실제 실행하지 않음
    # ========================================================

    def plan_motion(
        self,
        target_joints,
    ):

        self.get_logger().info("")
        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "GRASP MOTION PLANNING"
        )
        self.get_logger().info(
            "========================================"
        )

        # ----------------------------------------------------
        # 현재 실제 joint state
        # ----------------------------------------------------

        current = self.get_current_joint_state()

        if current is None:

            self.get_logger().error(
                "Could not get current joint state."
            )

            return False

        start_state = RobotState()

        start_state.joint_state = (
            copy.deepcopy(current)
        )

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        goal = MoveGroup.Goal()

        goal.request.group_name = ARM_GROUP

        goal.request.start_state = (
            start_state
        )

        # ----------------------------------------------------
        # Joint constraints
        # ----------------------------------------------------

        constraints = Constraints()

        for name, position in zip(
            target_joints.name,
            target_joints.position,
        ):

            if not name.startswith("joint"):
                continue

            jc = JointConstraint()

            jc.joint_name = name
            jc.position = position

            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01

            jc.weight = 1.0

            constraints.joint_constraints.append(
                jc
            )

        goal.request.goal_constraints.append(
            constraints
        )

        # ----------------------------------------------------
        # Planning
        # ----------------------------------------------------

        goal.request.num_planning_attempts = 10

        goal.request.allowed_planning_time = 10.0

        goal.request.max_velocity_scaling_factor = 0.1

        goal.request.max_acceleration_scaling_factor = 0.1

        goal.request.pipeline_id = "ompl"

        goal.request.planner_id = ""

        # ★ 계획만
        goal.planning_options.plan_only = True

        goal.planning_options.look_around = False

        goal.planning_options.replan = True

        goal.planning_options.replan_attempts = 2

        # ----------------------------------------------------
        # 전송
        # ----------------------------------------------------

        self.get_logger().info(
            "Sending grasp motion planning request..."
        )

        future = (
            self.move_client.send_goal_async(
                goal
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        handle = future.result()

        if handle is None:

            self.get_logger().error(
                "Failed to send MoveIt goal."
            )

            return False

        if not handle.accepted:

            self.get_logger().error(
                "MoveIt rejected the planning request."
            )

            return False

        self.get_logger().info(
            "Planning request accepted."
        )

        result_future = (
            handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        result = (
            result_future.result().result
        )

        error_code = result.error_code.val

        self.get_logger().info("")
        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "GRASP PLANNING RESULT"
        )
        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            f"MoveIt error code: {error_code}"
        )

        if error_code == 1:

            self.get_logger().info(
                "PLANNING SUCCESS!"
            )

            trajectory = (
                result.planned_trajectory
                .joint_trajectory
            )

            self.get_logger().info(
                f"Trajectory points: "
                f"{len(trajectory.points)}"
            )

            if len(trajectory.points) > 0:

                final = trajectory.points[-1]

                self.get_logger().info(
                    "Final planned joint position:"
                )

                for name, value in zip(
                    trajectory.joint_names,
                    final.positions,
                ):

                    self.get_logger().info(
                        f"  {name}: "
                        f"{value:.6f} rad"
                    )

            self.get_logger().info("")
            self.get_logger().info(
                "현재 단계에서는 실제 로봇을 "
                "grasp pose로 이동시키지 않습니다."
            )

            return True

        else:

            self.get_logger().error(
                "PLANNING FAILED!"
            )

            self.get_logger().error(
                f"MoveIt error code: {error_code}"
            )

            return False


# ============================================================
# main
# ============================================================

def main():

    rclpy.init()

    node = GraspPlanner()

    try:

        # ====================================================
        # STEP 1
        # TEST_POSE 계획
        # ====================================================

        trajectory = (
            node.plan_test_pose()
        )

        if trajectory is None:

            node.get_logger().error(
                "TEST_POSE planning failed."
            )

            return

        # ====================================================
        # STEP 2
        # TEST_POSE 실제 이동
        # ====================================================

        success = (
            node.execute_test_pose(
                trajectory
            )
        )

        if not success:

            node.get_logger().error(
                "TEST_POSE execution failed."
            )

            return

        # ====================================================
        # STEP 3
        # TEST_POSE 도착
        # ====================================================

        node.get_logger().info("")
        node.get_logger().info(
            "========================================"
        )
        node.get_logger().info(
            "TEST_POSE ARRIVED"
        )
        node.get_logger().info(
            "========================================"
        )

        # ====================================================
        # STEP 4
        # Grasp Pose
        # ====================================================

        pose = node.get_grasp_pose()

        # ====================================================
        # STEP 5
        # IK
        # ====================================================

        target_joints = (
            node.solve_ik(
                pose
            )
        )

        if target_joints is None:

            node.get_logger().error(
                "Stopping because IK failed."
            )

            return

        # ====================================================
        # STEP 6
        # Grasp Motion Planning
        #
        # 실제 이동 X
        # ====================================================

        success = (
            node.plan_motion(
                target_joints
            )
        )

        if success:

            node.get_logger().info("")
            node.get_logger().info(
                "========================================"
            )
            node.get_logger().info(
                "FINAL RESULT"
            )
            node.get_logger().info(
                "========================================"
            )
            node.get_logger().info(
                "TEST_POSE 이동 성공"
            )
            node.get_logger().info(
                "IK 성공"
            )
            node.get_logger().info(
                "Grasp Pose Planning 성공"
            )
            node.get_logger().info(
                "Grasp Pose 실제 실행은 하지 않음"
            )

        else:

            node.get_logger().error(
                "FINAL RESULT:"
            )

            node.get_logger().error(
                "TEST_POSE 성공"
            )

            node.get_logger().error(
                "IK 성공"
            )

            node.get_logger().error(
                "Grasp Pose Planning 실패"
            )

    except KeyboardInterrupt:

        node.get_logger().warn(
            "Interrupted by user."
        )

    finally:

        node.destroy_node()

        rclpy.shutdown()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
