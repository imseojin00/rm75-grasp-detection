import rclpy
from rclpy.node import Node

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient


class SimpleMoveItTest(Node):

    def __init__(self):
        super().__init__("simple_moveit_test")

        self.client = ActionClient(
            self,
            MoveGroup,
            "/move_action"
        )

    def run(self):

        print("=" * 40)
        print("RM75 SIMPLE MOVEIT TEST")
        print("=" * 40)

        print("MoveIt action 서버 대기 중...")

        if not self.client.wait_for_server(timeout_sec=10.0):
            print("MoveIt action 서버 연결 실패")
            return

        print("MoveIt action 서버 연결 성공")

        goal = MoveGroup.Goal()

        goal.request.group_name = "rm_group"
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.2
        goal.request.max_acceleration_scaling_factor = 0.2

        # grasp position
        x = -0.02653188
        y = 0.03867231
        z = 0.22600001

        print()
        print("테스트 목표 위치:")
        print([x, y, z])

        print()
        print("grasp quaternion [x,y,z,w]:")
        print([
            0.41326808,
            -0.04902774,
            0.87791039,
            0.23681029
        ])

        constraint = Constraints()

        position_constraint = PositionConstraint()

        position_constraint.header.frame_id = "base_link"
        position_constraint.link_name = "Link7"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.03]

        position_constraint.constraint_region.primitives.append(
            primitive
        )

        pose = Pose()

        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        pose.orientation.x = 0.41326808
        pose.orientation.y = -0.04902774
        pose.orientation.z = 0.87791039
        pose.orientation.w = 0.23681029

        position_constraint.constraint_region.primitive_poses.append(
            pose
        )

        position_constraint.weight = 1.0

        constraint.position_constraints.append(
            position_constraint
        )

        goal.request.goal_constraints.append(
            constraint
        )

        print()
        print("현재 테스트:")
        print("grasp 위치 + quaternion")

        print()
        print("MoveIt planning 시작...")

        future = self.client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        goal_handle = future.result()

        if not goal_handle.accepted:
            print("MoveIt goal 거절됨")
            return

        print("MoveIt goal accepted")

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        result = result_future.result().result

        print()
        print("=" * 40)
        print("MOVEIT RESULT")
        print("=" * 40)

        print("error code:")
        print(result.error_code.val)

        if result.error_code.val == 1:

            print()
            print("Planning 성공!")

            points = result.planned_trajectory.joint_trajectory.points

            print("trajectory points:", len(points))

        else:

            print()
            print("Planning 실패")


def main(args=None):

    rclpy.init(args=args)

    node = SimpleMoveItTest()

    try:
        node.run()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
