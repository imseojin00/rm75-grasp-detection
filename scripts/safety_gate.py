"""
범열 안전 게이트 초안 반영.
실행 직전 이 체크를 반드시 통과해야 함.

22일 수정: W_MAX를 0.062 -> 0.067로 정정.
(0.07m가 그리퍼 실측 원본, 0.067m가 Day1에 이미 마진 적용된 값이었는데,
 범열이 21일 문서화 때 0.067m를 원본으로 착각해 마진을 중복 적용해서
 0.062m가 나왔던 것. 범열 확인 후 정정.)
"""

W_MAX = 0.067
W_MIN = 0.010


def check_safety(x, y, z_link7, width):
    """
    x, y: base_link 기준 파지점 좌표
    z_link7: Link7 목표 z값
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
    if not (W_MIN <= width <= W_MAX):
        failures.append(f"width={width:.4f} out of gripper range [{W_MIN}, {W_MAX}]")

    passed = len(failures) == 0
    return passed, failures


if __name__ == "__main__":
    print("=== 테스트 1: 어제 버그 값 (FAIL 예상 - x범위) ===")
    ok, reasons = check_safety(x=-0.479954, y=-0.039350, z_link7=0.341779, width=0.0331)
    print(f"통과: {ok}")
    for r in reasons:
        print(f"  - {r}")

    print("\n=== 테스트 2: 범열 실측값 근처 (PASS 예상) ===")
    ok, reasons = check_safety(x=0.351, y=0.004, z_link7=0.063, width=0.0331)
    print(f"통과: {ok}")

    print("\n=== 테스트 3: can (width 6.33cm, 0.067 기준 PASS 예상) ===")
    ok, reasons = check_safety(x=0.351, y=0.004, z_link7=0.063, width=0.0633)
    print(f"통과: {ok}")
    for r in reasons:
        print(f"  - {r}")

    print("\n=== 테스트 4: block_2x2 (width 6.46cm, 0.067 기준 PASS 예상) ===")
    ok, reasons = check_safety(x=0.351, y=0.004, z_link7=0.063, width=0.0646)
    print(f"통과: {ok}")
    for r in reasons:
        print(f"  - {r}")
