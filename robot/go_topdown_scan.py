#!/usr/bin/env python3
"""수직 파지용 스캔 자세로 복귀 (2026-08-24, 그리퍼 열기 추가)"""
import time
import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Movej, Gripperset

SCAN_POSITION = [
    0.0021464,
    0.4226041,
    -0.0314972,
    2.1178541,
    -0.0028967,
    0.5066782,
    -0.0052350,
]
SPEED = 10


def main():
    rclpy.init()
    node = Node('go_topdown_scan')
    joint_pub = node.create_publisher(Movej, '/rm_driver/movej_cmd', 10)
    gripper_pub = node.create_publisher(Gripperset, '/rm_driver/set_gripper_position_cmd', 10)

    print('스캔 자세로 이동합니다. 작업 공간 확인하세요.')
    input('Enter=실행, Ctrl+C=취소: ')

    # 연결 대기 (최대 5초)
    print('구독자 연결 대기 중...')
    for i in range(50):
        if joint_pub.get_subscription_count() > 0 and gripper_pub.get_subscription_count() > 0:
            print(f'연결 확인됨 ({i * 0.1:.1f}초)')
            break
        time.sleep(0.1)
    else:
        print('경고: 구독자 연결을 확인 못했습니다. 그래도 발행 시도합니다.')

    # ── 1. 그리퍼 먼저 열기 ──
    print('그리퍼 여는 중...')
    gripper_msg = Gripperset()
    gripper_msg.position = 1000
    for i in range(5):
        gripper_pub.publish(gripper_msg)
        time.sleep(0.2)
    print('그리퍼 열기 명령 전송 완료')
    time.sleep(1.5)  # 그리퍼가 실제로 열릴 시간 확보

    # ── 2. 스캔 자세로 이동 ──
    msg = Movej()
    msg.joint = SCAN_POSITION
    msg.speed = SPEED
    msg.block = True
    msg.trajectory_connect = 0
    msg.dof = 7

    for i in range(3):
        joint_pub.publish(msg)
        print(f'스캔 자세 발행 {i+1}/3')
        time.sleep(0.5)

    print('발행 완료. 이동 중...')
    time.sleep(8)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
