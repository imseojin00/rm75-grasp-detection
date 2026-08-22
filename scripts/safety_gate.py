"""
범열 안전 게이트 초안 반영.
실행 직전 이 체크를 반드시 통과해야 함.
"""

def check_safety(x, y, z_link7, width):
    """
    x, y: base_link 기준 파지점 좌표 (z는 파지점, z_link7은 Link7 목표)
    width: 그리퍼 폭 (m)
    반환: (통과여부: bool, 실패사유 리스트)
    """
    failures = []

    if not (0.25 <= x <= 0.60):
        failures.append(f"x={x:.4f} out of range [0.25, 0.60]")
    if not (-0.25 <= y <= 0.25):
        failures.append(f"y={y:.4f} out of range [-0.25, 0.25]")
    if not (z_link7 >= 0.058):
        failures.append(f"z_link7={z_link7:.4f} below minimum 0.058")
    if not (0.010 <= width <= 0.062):
        failures.append(f"width={width:.4f} out of gripper range [0.010, 0.062]")

    passed = len(failures) == 0
    return passed, failures


if __name__ == "__main__":
    # 어제 잘못됐던 값으로 테스트 (반드시 FAIL 나와야 정상)
    print("=== 테스트 1: 어제 버그 값 (FAIL 예상) ===")
    ok, reasons = check_safety(x=-0.479954, y=-0.039350, z_link7=0.341779, width=0.0331)
    print(f"통과: {ok}")
    for r in reasons:
        print(f"  - {r}")

    print("\n=== 테스트 2: 범열 실측값 근처 (PASS 예상) ===")
    ok, reasons = check_safety(x=0.351, y=0.004, z_link7=0.063, width=0.0331)
    print(f"통과: {ok}")
    for r in reasons:
        print(f"  - {r}")
