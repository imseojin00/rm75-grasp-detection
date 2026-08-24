import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Vector3
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

BASE_FRAME = "base_link"
EE_LINK = "Link7"
ARM_GROUP = "rm_group"


def make_pose(x, y, z):
    p = Pose()
    p.position.x = x
    p.position.y = y
    p.position.z = z
    p.orientation.w = 1.0
    return p


class PositionOnlyTest(Node):
    def __init__(self):
        super().__init__("position_only_test")
        self._client = ActionClient(self, MoveGroup, "/move_action")

    def test(self, x, y, z):
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = EE_LINK
        pc.target_point_offset = Vector3()
        pc.constraint_region.primitives = [
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.02])
        ]
        pc.constraint_region.primitive_poses = [make_pose(x, y, z)]
        pc.weight = 1.0

        goal = MoveGroup.Goal()
        goal.request.group_name = ARM_GROUP
        goal.request.goal_constraints = [Constraints(position_constraints=[pc])]
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0

        print(f"목표: x={x}, y={y}, z={z} (orientation 제약 없음)")
        self._client.wait_for_server()
        send_goal_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            print("목표 거부됨")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        print(f"error_code: {result.error_code.val}")


def main():
    rclpy.init()
    node = PositionOnlyTest()
    x, y, z = 0.311557, 0.083962, 0.178498  # approach 위치로 변경
    node.test(x, y, z)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
