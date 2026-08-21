# 2026-08-21 수직 파지 실기체 검증

**작업자:** 범열
**환경:** RM75-BI + D435 + EG2-4C2, Docker(jazzy_dev) 내 ROS2 Jazzy
**배경:** 블록(2.5cm)은 측면 파지가 어려워 수직 파지로 전환.
테이블 위치 변경으로 관련 상수 전면 재실측.

---

## 확정 상수

| 파라미터 | 값 | 근거 |
|---|---|---|
| TABLE_Z | -0.120 [m] | base_link 원점에서 자로 직접 실측 |
| **GRASP_OFFSET_Z** | **0.171 [m]** | 파지 자세에서 tf2_echo Link7 z=+0.063, 파지점 -0.108 |
| 파지 자세 | (0, 1, 0, 0) | 실측 (-0.017, 0.999, 0.022, 0.032) 근사 |
| GRIPPER_BOX_SIZE | (0.11, 0.13, 0.18) [m] | 실측 손끝 0.175 + 5mm |
| GRIPPER_BOX_CENTER | (0, 0, 0.09) [m] | z의 절반 |
| TOUCH_LINKS | Link1~7 + base_link | 봉투 확대로 확장 필요 |
| LIFT | 0.05 [m] | 수직 상승량 |
| 손끝 길이 (열림) | 0.175 [m] | Link7 z + 테이블 간격 0.8cm 역산 |

---

## ⚠️ GRASP_OFFSET 불일치 (중요)
측면 파지 (05): 0.098 [m]
수직 파지 (오늘): 0.171 [m]

같은 그리퍼인데 4.2cm 차이. **원인 미확인.**
물체 크기 차이로는 설명 안 됨 (패드 위치는 물체와 무관).

→ **각 자세에서 실측한 값을 그대로 쓸 것.**
→ 임의 방향 파지로 확장 시 반드시 재실측.
→ "방향만 바꾸고 같은 거리 쓰면 된다"는 가정은 틀릴 수 있음.

---

## 검증 결과 (테스트 D — 비전 없음)

Link7 목표 `(0.351, 0.004, 0.063) [m]`

| 단계 | 결과 |
|---|---|
| 1) MoveGroup 접근 | 계획 성공 → 실행 성공 |
| 2) +5cm 상승 (Cartesian) | fraction **100%** |
| 3) -5cm 하강 (Cartesian) | fraction **100%** |

fraction 100%는 Scene이 실물과 일치한다는 뜻.
어제 TABLE_Z 오류 시 5%·0% 거부였던 것과 대조됨.
→ **ACM 예외 불필요.**

---

## 05(측면) 대비 변경점

- GRASP_OFFSET: 0.098 → 0.171 [m] (위 경고 참조)
- 봉투 z: 0.16 → 0.18. 기존 값은 실물보다 1.5cm 짧았음
- TOUCH_LINKS 확장 — 봉투 확대로 시작상태에서 `Link1 - gripper_envelope` 충돌 발생
- 직선 방향: x 전진 → z 상하

---

## 카메라

- 뎁스 정상. 뷰어에서 2.5cm 블록이 뚜렷하게 구분됨
- 스크립트 창에서 안 보였던 건 컬러맵 스케일 문제
- 시리얼 `335522070758` (04 기록 `212223021503`과 다름 — 기록 오류로 추정)
- 펌웨어 업데이트 권장 알림 뜨지만 **하지 말 것**

---

## 검출 알고리즘 이슈 (서진 전달)

`find_grasp.py`에서 block_1x1(25mm) 폭이 **5.2mm**로 계산됨.

**원인:** 단일 시점 점군은 물체 표면 한 면만 담김.
3D PCA 제3축이 물체 두께가 아니라 표면 평탄도를 재게 됨.
→ 짧은 축이 **항상 시선 방향으로 퇴화**. 물체 모양과 무관.

**제안:** 테이블 평면 투영 후 2D PCA
- 수직 파지라 접근축은 z 고정
- 파지 폭은 xy 평면에서만 결정
- 3축 퇴화 문제 원천 소멸

**검증 기준:** block_1x1 → 25mm, block_2x2 → 50mm 근처

---

## 인터페이스 (서진 A ↔ 범열 B)
좌표계: base_link
단위: 전부 [m]

파지점 정의: 조 안쪽 패드 중심 (그리퍼 열림 상태)
Link7 목표 = 파지점 + (0, 0, 0.171)

방향: 파지 축 단위벡터 (x, y, z) — 쿼터니언 변환은 B가 처리
폭: [m], 유효 범위 0.010 ~ 0.062

**미정 (합의 필요)**
1. 검출 실패 시 반환값?
2. 후보 여러 개면?
3. 반환 타입?

---

## 재현 방법

```bash
# 터미널1
ros2 launch rm_bringup rm_75_bringup.launch.py

# 터미널2 — 04 static TF
ros2 run tf2_ros static_transform_publisher \
  --x -0.092640 --y 0.048367 --z -0.020571 \
  --qx 0.001056 --qy 0.003856 --qz -0.699441 --qw 0.714679 \
  --frame-id Link7 --child-frame-id camera_color_optical_frame

# 터미널3 — venv 밖에서
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
python3 topdown_test.py
```

**그리퍼 (토픽 제어)**
```bash
# 열기
timeout 3 ros2 topic pub -r 10 /rm_driver/set_gripper_position_cmd \
  rm_ros_interfaces/msg/Gripperset "{position: 1000}"

# 파지
timeout 3 ros2 topic pub -r 10 /rm_driver/set_gripper_pick_cmd \
  rm_ros_interfaces/msg/Gripperpick "{speed: 200, force: 300}"
```
| GRIP_FORCE | 300 | 블록(딱딱·가벼움). 05의 600은 물 찬 병 기준이라 과함 |
| GRIP_SPEED | 200 | 05 기본값 유지 |
⚠️ `--once`는 구독자가 없으면 그냥 빠져나감. `-r 10` + `timeout` 방식 사용.
⚠️ `timeout` 없이 `-r 10`만 쓰면 명령이 계속 반복 발행됨.

---

## 남은 것 (24일)

- [ ] 좌표변환 검증 — `vision_to_base_test_v3.py` 활용 예정
  - `RADIUS_COMPENSATION=False` 필요 (원통 가정, 블록엔 부적합)
  - 검출 z는 블록 윗면 → 파지점은 -0.0125 [m] 보정 필요
- [ ] 안전 게이트 구현 (코드 초안 있음)
- [ ] 비전 통합

---

## 백업

`docker commit jazzy_dev jazzy_dev:v5_topdown` 완료