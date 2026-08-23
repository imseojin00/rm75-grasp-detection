#!/usr/bin/env python3
"""수직 파지용 스캔 자세로 복귀 (2026-08-23 실측)"""
import time
import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Movej

SCAN_POSITION = [
    0.0001396,
    0.0843009,
    -0.0781586,
    1.7060341,
    -0.0032283,
    1.2843200,
    -0.0078700,
]
SPEED = 10


def main():
    rclpy.init()
    node = Node('go_topdown_scan')
    pub = node.create_publisher(Movej, '/rm_driver/movej_cmd', 10)

    print('스캔 자세로 이동합니다. 작업 공간 확인하세요.')
    input('Enter=실행, Ctrl+C=취소: ')

    msg = Movej()
    msg.joint = SCAN_POSITION
    msg.speed = SPEED
    msg.block = True
    msg.trajectory_connect = 0
    msg.dof = 7

    time.sleep(1)                 # 퍼블리셔 연결 대기
    pub.publish(msg)
    print('발행 완료. 이동 중...')
    time.sleep(6)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
