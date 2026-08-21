# 개발 환경 및 워크스페이스

**작성:** 범열 (2026-08-21)
**대상:** 로봇 실행층을 다루는 사람 — 현재 범열, 25일 이후 서진

실기체를 만지기 전에 이 문서를 먼저 읽을 것. 환경이 어긋난 상태로
MoveIt을 띄우면 `CONTROL_FAILED`가 나고 원인 추적에 하루가 간다.

---

## 왜 Docker인가

선배 가이드(`ros2_jazzy_guide/`)가 **Ubuntu 24.04 + ROS2 Jazzy** 기준인데,
작업 PC 호스트는 **22.04 + Humble**이다.

호스트를 재설치하지 않고 유저스페이스만 24.04로 바꿔 끼우려고 컨테이너를 쓴다.

**컨테이너는 시뮬레이션이 아니다.** 커널·드라이버·하드웨어는 호스트 것을
그대로 쓰므로 USB 카메라와 이더넷 로봇에 직접 접근된다.

### 세 가지 "가상"의 구분

| 이름 | 정체 | 실물과 통신? |
|---|---|---|
| venv | 파이썬 패키지 격리 폴더 | ✅ 통신함 |
| Docker 컨테이너 | 유저스페이스 교체 | ✅ 통신함 |
| Gazebo | 물리 엔진 시뮬레이터 | ❌ 가짜 팔 |

venv 안에서 돌아간 `watch_rm_pose.py`가 실제 팔의 Z=0.8505m를 읽어온 것,
`test_realsense.py`가 실제 시리얼 번호를 뱉은 것이 그 증거다.

---

## 세 겹 구조

호스트 (Ubuntu 22.04)
└─ Docker 컨테이너 jazzy_dev (Ubuntu 24.04 + ROS2 Jazzy)
├─ /root/ros2_ws colcon 워크스페이스 (01~03)
└─ /root/robot_vision 비전 스크립트 (04~06)
└─ vision_env Python venv


**격리된 건 소프트웨어 환경뿐. 하드웨어는 전부 진짜다.**

---

## 워크스페이스 구분

| 경로 | 용도 | venv |
|---|---|---|
| `/root/ros2_ws` | RealMan 공식 `ros2_rm_robot` 빌드. rm_driver, rm_bringup 등 20개 패키지 | ✗ |
| `/root/robot_vision` | 04~06 비전·파지 스크립트 | ✓ |

### ⚠️ 저장소가 두 개다

- **`rm75-edu`** — 가이드 **문서** 저장소. 빌드 대상 아님
- **`ros2_rm_robot`** — RealMan 공식. 실제 빌드 대상. **`-b jazzy` 필수**
  (배포판별로 브랜치가 나뉘어 있어 빠뜨리면 기본 브랜치가 받아진다)

그리고 `rm75-edu` 안에서도 두 갈래다.

- 최상위 `README` — 후배 교육용 (Humble, **시뮬레이션 전용**)
- `ros2_jazzy_guide/` — **원본. 우리가 쓰는 것** (커밋 로그에서 확인됨)

`~/robot_ws`, `rm75_edu_moveit_config` 같은 이름이 보이면 후배 교육용
쪽을 보고 있는 것이다. 실기체에는 쓰지 않는다.

---

## venv 철칙

colcon build, ros2 launch, ros2 run → venv 밖에서
04~06 파이썬 스크립트 → venv 안에서


프롬프트의 `(vision_env)`가 구분 기준.

**`colcon build`를 venv 안에서 하면 빌드가 깨진다** — colcon이 venv의
파이썬을 잡기 때문.

venv를 쓰는 이유는 `pyrealsense2`·`opencv`·`ultralytics`·`torch`를
시스템 파이썬에 설치하면 ROS 패키지와 의존성이 충돌하기 때문이다.
ROS 패키지들이 numpy·PyYAML을 자기 버전으로 요구하는데 비전
라이브러리는 다른 버전을 요구한다.

### venv 구성 (재현용)

```bash
docker exec -it jazzy_dev bash
mkdir -p /root/robot_vision && cd /root/robot_vision
python3 -m venv vision_env
source vision_env/bin/activate
pip install opencv-python==4.10.0.84 pyrealsense2==2.58.1.10581 \
            numpy==2.0.2 scipy==1.13.1 pyyaml==6.0.2
pip install Robotic-Arm
```

⚠️ **venv 디렉터리는 복사해서 옮길 수 없다** (절대 경로가 하드코딩됨).
다른 PC로 옮길 땐 `pip freeze`로 목록을 뽑아 재생성할 것.

---

## 네트워크

로봇 192.168.1.18
PC 192.168.1.100/24 (게이트웨이 비움 — WiFi 인터넷 유지)


`rm_75_config.yaml`의 `arm_ip`, `udp_ip`가 여기 맞춰져 있다.

GUI에서 `Activation failed`가 나면 `nmcli`로:

```bash
sudo nmcli con add type ethernet ifname enp4s0 con-name rm75 \
  ipv4.method manual ipv4.addresses 192.168.1.100/24
sudo nmcli con up rm75
```

---

## 실기체 실행 절차

### 터미널 1 — bringup

```bash
docker exec -it jazzy_dev bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch rm_bringup rm_75_bringup.launch.py
```

**이거 하나로 rm_driver + move_group + RViz + TF가 전부 뜬다.**

### ⚠️ moveit_config를 따로 켜지 마라

bringup 안에 move_group이 이미 들어 있다. 별도로 `*_moveit_config`를
실행하면 두 개가 겹쳐 실행이 `CONTROL_FAILED(-4)`로 실패한다.
계획(planning)은 성공하는데 실행만 안 되는 증상이 이것이다.

`use_sim_time:=True`도 넣지 않는다. 실기체는 실시간이다.

### 터미널 2 — 04 static TF (T_ee_cam)

```bash
source /opt/ros/jazzy/setup.bash
ros2 run tf2_ros static_transform_publisher \
  --x -0.092640 --y 0.048367 --z -0.020571 \
  --qx 0.001056 --qy 0.003856 --qz -0.699441 --qw 0.714679 \
  --frame-id Link7 --child-frame-id camera_color_optical_frame
```

이걸 빼면 비전 스크립트가 TF WAIT에 머문다.
04 검증 결과 평균 오차 **8.1 mm** (선배 값 6.9mm와 동급 — 브래킷 안 움직임).

### 터미널 3 — 스크립트

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/robot_vision/vision_env/bin/activate   # 비전 스크립트만
cd ~/robot_vision && python3 <스크립트>
```

---

## 그리퍼 (EG2-4C2)

**MoveIt으로 제어하지 않는다.** 두 가지 이유:

1. 파지의 핵심은 힘 제어인데 MoveIt 그리퍼 그룹은 관절 각도 목표
   (위치 제어)다. 위치 제어로 닫으면 과부하 보호로 힘이 풀려
   이송 중 미끄러진다.
2. 실물에는 ros2_control 하드웨어 인터페이스가 없고, 개방도를
   `/joint_states`로 보고하지도 않는다.

`gripper_finger1_joint`를 못 찾는다는 경고가 뜬다면, 그리퍼 포함
moveit_config(`rm_75_jaw_config` 계열)를 쓰고 있는 것이다.
그건 **시뮬레이션 전용**이다. bringup 모델에는 그리퍼가 없다.

### 토픽으로 제어

```bash
# 열기
timeout 3 ros2 topic pub -r 10 /rm_driver/set_gripper_position_cmd \
  rm_ros_interfaces/msg/Gripperset "{position: 1000}"

# 파지 (블록 300 / 물 찬 병 600)
timeout 3 ros2 topic pub -r 10 /rm_driver/set_gripper_pick_cmd \
  rm_ros_interfaces/msg/Gripperpick "{speed: 200, force: 300}"

# 결과 확인 (별도 터미널)
ros2 topic echo /rm_driver/set_gripper_pick_result
```

⚠️ `--once`는 구독자가 없으면 그냥 빠져나간다. `-r 10` + `timeout` 방식.
⚠️ `timeout` 없이 `-r 10`만 쓰면 명령이 무한 반복 발행된다.

### 실측 스펙

스트로크 0.067 m (계약서 "0.07"은 반올림)
W_MAX (마진 포함) 0.062 m


---

## 그리퍼 충돌 봉투 (envelope)

bringup 모델에 그리퍼·카메라가 없어 MoveIt이 그 부피를 모른다.
Link7에 박스를 `AttachedCollisionObject`로 부착해 충돌 검사에 반영한다.

정밀 모델이 아니라 **실물보다 조금 큰 안전 봉투**다.

GRIPPER_BOX_SIZE = (0.11, 0.13, 0.18) [m] 실측 손끝 0.175 + 5mm
GRIPPER_BOX_CENTER = (0, 0, 0.09) [m] z의 절반
TOUCH_LINKS = Link1~7 + base_link


`TOUCH_LINKS`가 좁으면 시작 상태에서
`CheckStartStateCollision failed: LinkN - gripper_envelope`가 뜬다.

**02의 `rm_75_jaw_config`를 실기체에 안 쓰는 이유:** 손가락 관절이
가동 관절인데 실물이 상태를 보고하지 않아 robot_state_publisher가
그리퍼 TF를 못 만들고 MoveIt 상태 모니터가 막힌다.

---

## 자주 겪는 문제

| 증상 | 원인 · 대응 |
|---|---|
| `CONTROL_FAILED(-4)` | moveit_config를 bringup과 중복 실행. bringup만 쓸 것 |
| `use_sim_time:=True`가 켜져 있음 | Gazebo 프로세스 잔존. `pkill -9 -f "gz sim"` |
| `/joint_states`가 300Hz | 발행자가 둘 (Gazebo 97Hz + 실기체 200Hz). 위와 동일 |
| `gripper_finger*_joint` 없음 경고 | 그리퍼 포함 moveit_config 사용 중. 실기체엔 안 씀 |
| 영상 창 TF WAIT 지속 | 터미널2(static TF) 미실행 |
| 카메라 Device busy | 카메라는 한 번에 한 프로그램만. realsense-viewer 종료 |
| USB·이더넷 동시 두절 | PC 절전(s2idle) 복귀 실패. 케이블 재연결. **절전 꺼둘 것** |
| 계획 실패 `error_code=99999` | 확률적 플래너의 정상 실패. 3회 모두 실패할 때만 Scene·목표 점검 |
| `LinkN - gripper_envelope` 충돌 | `TOUCH_LINKS` 확장 |
| 직선 fraction < 90% | Scene이 실물과 어긋남. TABLE_Z 재실측 |
| 그리퍼 결과 None (15초 침묵) | 툴 전원 이상. CLI로 position 500 쏴서 실물 반응 확인 |
| RViz 종료 | Scene 물체를 **클릭**했는지. 표시만으로는 안 죽음 |
| `move_group` exit code -11 | 종료 시 segfault. 실행 중엔 정상. Ctrl+C로 한 번에 종료 |
| 종료 시 Qt 타이머 경고 | 무시 (OpenCV 창을 스레드에서 닫는 부수 현상) |

### 아침 점검 루틴

```bash
lsusb | grep -i intel        # D435
ping -c 3 192.168.1.18       # 로봇
```

절전 복귀 시 USB와 이더넷이 **동시에** 죽는다. 두 번 겪었다.

---

## 안전 수칙

- `VEL_SCALE` / `ACC_SCALE` = **0.1**. 올리지 않는다
- Cartesian `fraction` 하한 **0.9**. 낮추지 않는다
  (낮춘다는 건 "충돌 직전까지는 가겠다"는 뜻)
- 계획 → **RViz 잔상 확인** → Enter → 실행. 이 순서를 건너뛰지 않는다
- RViz의 Scene 물체는 **눈으로만**. 클릭·드래그하면 RViz가 죽는다
- 로봇 명령 발행부(`movej_p`·`pick_on`·execute)는 Claude Code에 맡기지 않고
  직접 작성·검토한다
- 계산 결과 → 로봇 명령 사이에 **범위 검사 + 사람 승인** 게이트를 둔다

---

## 실측 원칙 — 연쇄 측정보다 직접 측정

TABLE_Z를 정할 때 두 방법이 3 cm 어긋난 사례가 있다.

| 방법 | 문제 |
|---|---|
| 접촉 역산 (Link7 z − 손끝 길이) | 손끝 길이가 개폐 상태에 따라 변한다 |
| base_link 원점에서 자로 직접 | **채택** |

**기준점은 불변량으로.** Link7(플랜지 면)이나 base_link 원점.
"손끝"처럼 상태에 따라 변하는 것은 기준으로 쓰지 않는다.

기준점 함정에 이틀 연속 걸렸다 — 파지점 정의 차이(6.5cm), TABLE_Z(3cm).
**같은 수치를 두 사람이 다르게 재고 있지 않은지 명시적으로 합의할 것.**

---

## 백업

`ros2_ws`와 `robot_vision`이 컨테이너 안에만 있어 **`docker commit`이
유일한 보존 수단**이다.

```bash
docker commit jazzy_dev jazzy_dev:v5_topdown
docker images | grep jazzy_dev
```

현재 스냅샷: `v1`~`v4` (02 단계), `v5_topdown` (2026-08-21 수직 파지 검증)

### 다른 PC로 옮기려면

```bash
docker save jazzy_dev:v5_topdown | gzip > jazzy_dev_v5.tar.gz
# 옮긴 뒤
gunzip -c jazzy_dev_v5.tar.gz | docker load
```

수 GB이므로 시간이 걸린다.

### 컨테이너 ↔ 호스트 파일 이동

```bash
docker cp jazzy_dev:/root/robot_vision/topdown_test.py ~/dest/
docker cp ~/src/find_grasp.py jazzy_dev:/root/robot_vision/
```

---

## 25일 인수인계 체크리스트

- [ ] 컨테이너 이미지 전달 또는 대상 PC에 환경 구축
- [ ] `docker commit`으로 최신 상태 스냅샷
- [ ] 네트워크 설정 (`nmcli` 명령 포함)
- [ ] 04 static TF 값 (브래킷 안 움직였는지 확인)
- [ ] 확정 상수 전체 → `docs/2026-08-21_topdown_verified.md`
- [ ] 실행 절차 실연 (인수자가 직접 한 번 돌려볼 것)

---

## ⚠️ 미결 — 환경 공유 방법

현재 실기체 환경은 **범열 PC의 컨테이너 하나뿐**이다.
다른 팀원이 자기 PC에서 세우려 하면 워크스페이스 혼동으로 막힌다
(2026-08-21 실제 발생).

정해야 할 것:
- `docker save`로 이미지 전달할 것인가
- 한 대에서 같이 작업할 것인가
- 각자 PC에 구축할 것인가 (하루 소요 예상)

**25일 이후 로봇 파트를 서진이 인수하므로 그 전에 결정 필요.**
