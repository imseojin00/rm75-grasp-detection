#!/usr/bin/env python3
"""수직 파지용 스캔 자세로 복귀 (2026-08-23 실측, 연결 대기 개선)"""
import time
import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Movej

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
    pub = node.create_publisher(Movej, '/rm_driver/movej_cmd', 10)

    print('스캔 자세로 이동합니다. 작업 공간 확인하세요.')
    input('Enter=실행, Ctrl+C=취소: ')

    # 연결될 때까지 대기 (최대 5초)
    print('구독자 연결 대기 중...')
    for i in range(50):
        if pub.get_subscription_count() > 0:
            print(f'연결 확인됨 ({i * 0.1:.1f}초)')
            break
        time.sleep(0.1)
    else:
        print('경고: 구독자 연결을 확인 못했습니다. 그래도 발행 시도합니다.')

    msg = Movej()
    msg.joint = SCAN_POSITION
    msg.speed = SPEED
    msg.block = True
    msg.trajectory_connect = 0
    msg.dof = 7

    # 안전하게 3번 발행 (중복 발행은 로봇에 무해, 연결 안정성 확보용)
    for i in range(3):
        pub.publish(msg)
        print(f'발행 {i+1}/3')
        time.sleep(0.5)

    print('발행 완료. 이동 중...')
    time.sleep(8)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
