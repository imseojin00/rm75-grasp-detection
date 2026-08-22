import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState


class FKClient(Node):
    def __init__(self):
        super().__init__('fk_client')
        self.client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('compute_fk service waiting...')

    def compute_fk(self, joint_names, joint_positions, link_name='Link7'):
        request = GetPositionFK.Request()
        request.header.frame_id = 'base_link'
        request.fk_link_names = [link_name]

        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = joint_names
        joint_state.position = joint_positions
        robot_state.joint_state = joint_state
        request.robot_state = robot_state

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    rclpy.init()
    node = FKClient()

    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    joint_positions = [
        -0.23946634254455568,
        -1.3387117000579833,
        -2.957932096862793,
        2.1372236042022705,
        -0.1434390046596527,
        -1.9061682010650636,
        -0.22130090279579162,
    ]

    result = node.compute_fk(joint_names, joint_positions)

    print("error_code:", result.error_code.val)
    for i, pose_stamped in enumerate(result.pose_stamped):
        pose = pose_stamped.pose
        print("link:", result.fk_link_names[i])
        print("position: x=%.6f y=%.6f z=%.6f" % (pose.position.x, pose.position.y, pose.position.z))
        print("orientation: x=%.6f y=%.6f z=%.6f w=%.6f" % (
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
