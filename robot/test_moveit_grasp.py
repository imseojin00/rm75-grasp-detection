#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

from sensor_msgs.msg import JointState


class GraspIKTest(Node):

    def __init__(self):
        super().__init__("grasp_ik_test")

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik"
        )

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Waiting for /compute_ik ..."
            )

        self.get_logger().info(
            "MoveIt IK service connected."
        )

    def test(self):

        # ====================================================
        # Vision → base_link에서 계산된 grasp pose
        # ====================================================

        position = [
            0.42415417,
            0.10695407,
            -0.11000401
        ]

        quaternion = [
            0.30867291,
            0.78475241,
            0.40639081,
            0.35175445
        ]

        # ====================================================
        # Pose
        # ====================================================

        pose = Pose()

        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]

        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        # ====================================================
        # IK request
        # ====================================================

        request = GetPositionIK.Request()

        request.ik_request.group_name = "rm_group"

        request.ik_request.pose_stamped.header.frame_id = "base_link"

        request.ik_request.pose_stamped.pose = pose

        request.ik_request.ik_link_name = "Link7"

        request.ik_request.timeout.sec = 2

        # ====================================================
        # 현재 joint state를 seed로 사용
        # ====================================================

        joint_state = JointState()

        joint_state.name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7"
        ]

        request.ik_request.robot_state = RobotState()

        request.ik_request.robot_state.joint_state = joint_state

        # ====================================================
        # 요청
        # ====================================================

        self.get_logger().info("")
        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "Grasp IK TEST"
        )
        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            f"Position: {position}"
        )

        self.get_logger().info(
            f"Quaternion: {quaternion}"
        )

        self.get_logger().info(
            "group: rm_group"
        )

        self.get_logger().info(
            "ik_link: Link7"
        )

        future = self.ik_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        if future.result() is None:

            self.get_logger().error(
                "IK service call failed."
            )

            return

        response = future.result()

        # ====================================================
        # 결과
        # ====================================================

        self.get_logger().info("")

        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            f"IK error code: "
            f"{response.error_code.val}"
        )

        self.get_logger().info(
            "========================================"
        )

        if response.error_code.val == 1:

            self.get_logger().info(
                "SUCCESS: IK solution found!"
            )

            self.get_logger().info(
                "Joint solution:"
            )

            names = (
                response.solution.joint_state.name
            )

            positions = (
                response.solution.joint_state.position
            )

            for name, position in zip(
                names,
                positions
            ):

                if name.startswith("joint"):

                    self.get_logger().info(
                        f"  {name}: "
                        f"{position:.6f} rad"
                    )

        else:

            self.get_logger().error(
                "FAILED: IK solution not found."
            )

            self.get_logger().error(
                "The grasp pose may be unreachable "
                "or the orientation may be incorrect."
            )


def main(args=None):

    rclpy.init(args=args)

    node = GraspIKTest()

    node.test()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
