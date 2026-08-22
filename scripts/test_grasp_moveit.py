import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose

from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
)

from rclpy.action import ActionClient


class GraspMoveItTest(Node):

    def __init__(self):
        super().__init__("grasp_moveit_test")

        self._action_client = ActionClient(
            self,
            MoveGroup,
            "/move_action"
        )

    def run(self):

        print("================================")
        print("RM75 GRASP MOVEIT TEST")
        print("================================")

        print("MoveIt action 서버 대기 중...")

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            print("MoveIt action 서버 연결 실패")
            return

        print("MoveIt action 서버 연결 성공")
        print()

        # ============================================================
        # GPD grasp position
        # ============================================================

        position = [
            -0.02653188,
             0.03867231,
             0.22600001
        ]

        # ============================================================
        # GPD grasp quaternion
        # 현재 단계에서는 orientation constraint에 사용하지 않음
        # ============================================================

        quaternion = [
             0.41326808,
            -0.04902774,
             0.87791039,
             0.23681029
        ]

        print("grasp position:")
        print(position)
        print()

        print("grasp quaternion [x,y,z,w]:")
        print(quaternion)
        print()

        print("현재 테스트:")
        print("grasp 위치만 MoveIt에 전달")
        print("orientation constraint는 제외")
        print()

        # ============================================================
        # MoveGroup Goal
        # ============================================================

        goal_msg = MoveGroup.Goal()

        goal_msg.request.group_name = "rm_group"

        goal_msg.request.allowed_planning_time = 5.0

        goal_msg.request.num_planning_attempts = 10

        # ============================================================
        # Position Constraint
        # ============================================================

        position_constraint = PositionConstraint()

        position_constraint.header.frame_id = "base_link"

        position_constraint.link_name = "Link7"

        # ============================================================
        # BOX constraint
        # ============================================================

        primitive = SolidPrimitive()

        primitive.type = SolidPrimitive.BOX

        primitive.dimensions = [
            0.01,
            0.01,
            0.01
        ]

        position_constraint.constraint_region.primitives.append(
            primitive
        )

        # ============================================================
        # grasp position
        # ============================================================

        pose = Pose()

        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]

        # orientation은 현재 사용하지 않지만
        # Pose 메시지이므로 유효한 기본 quaternion을 넣음
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        position_constraint.constraint_region.primitive_poses.append(
            pose
        )

        position_constraint.weight = 1.0

        # ============================================================
        # Constraints
        # ============================================================

        constraints = Constraints()

        constraints.position_constraints.append(
            position_constraint
        )

        goal_msg.request.goal_constraints.append(
            constraints
        )

        # ============================================================
        # Planning
        # ============================================================

        print("MoveIt planning 시작...")
        print()

        future = self._action_client.send_goal_async(
            goal_msg
        )

        rclpy.spin_until_future_complete(
            self,
            future
        )

        goal_handle = future.result()

        if goal_handle is None:
            print("Goal 전송 실패")
            return

        if not goal_handle.accepted:
            print("MoveIt goal rejected")
            return

        print("MoveIt goal accepted")
        print()

        # ============================================================
        # 결과 대기
        # ============================================================

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        result = result_future.result().result

        # ============================================================
        # 결과 출력
        # ============================================================

        print("================================")
        print("MOVEIT RESULT")
        print("================================")

        print("error code:")
        print(result.error_code.val)

        print()

        if result.error_code.val == 1:

            print("SUCCESS")
            print("Grasp position까지 planning 성공")

        else:

            print("Planning 실패")

        print("================================")


def main(args=None):

    rclpy.init(args=args)

    node = GraspMoveItTest()

    try:
        node.run()

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
