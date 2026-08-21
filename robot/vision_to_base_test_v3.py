"""
[05 테스트 A-3] 비전 좌표 → base_link 좌표 변환 검증 (반지름 보정 추가)

v3 추가:
  - 원통 반지름 보정. 깊이 센서가 주는 점은 물체의 '카메라를 향한 표면'이므로,
    보는 방향이 바뀌면 표면점도 옮겨간다(원통이면 최대 지름만큼). 파지 목표인
    중심축을 얻으려면 시선 방향으로 반지름만큼 밀어야 한다.
      물체 폭 = bbox 폭(px) x 깊이 / fx,  반지름 = 폭 / 2
    RADIUS_COMPENSATION = False 로 두면 v2와 동일하게 동작한다(비교용).

v1 대비 수정:
  1. TF를 별도 spin 스레드로 처리 — v1은 루프당 spin_once 1회라
     200 Hz TF 스트림을 못 따라가 버퍼가 밀렸고, 그 결과 팔이 멈춘 뒤에도
     과거 자세의 TF로 변환해 base 좌표가 크게 흔들렸다.
     (카메라 좌표는 고정인데 base 좌표만 변하면 이 증상이다.)
  2. TF 나이 검사 — 오래된 변환이면 측정을 거부한다.
  3. 깊이를 중심 픽셀 1개가 아니라 bbox 중앙 ROI의 median으로 계산.
  4. 팔 정지 감지 — TF가 충분히 안정된 뒤에만 '측정 확정'으로 표시.
  5. Enter를 눌러 한 점씩 기록하고, 종료 시 자세 간 산포를 계산.

전제 (터미널 구성):
  터미널1: ros2 launch rm_bringup rm_75_bringup.launch.py
  터미널2: 04의 static TF (Link7 → camera_color_optical_frame)
  터미널3: 이 스크립트 (ROS source 후 venv activate)

사용법:
  영상 창에서 물병이 검출된 상태로 팔을 원하는 자세에 두고,
  터미널에서 Enter → 한 점 기록. q + Enter → 종료 및 산포 출력.
"""

import statistics
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────
TARGET_CLASS = "bottle"
CONF_THRESHOLD = 0.5
DEPTH_MIN, DEPTH_MAX = 0.15, 2.0
SMOOTH_WINDOW = 7
ROI_RATIO = 0.3           # bbox 중앙 이 비율만큼을 깊이 ROI로 사용
CAMERA_FRAME = "camera_color_optical_frame"
BASE_FRAME = "base_link"
MAX_TF_AGE = 0.3          # 이보다 오래된 TF는 거부 (초)
STILL_THRESHOLD = 0.002   # 이 이하로 움직이면 정지로 간주 (m)
RADIUS_COMPENSATION = True   # 표면점 → 중심축 보정 사용 여부
MAX_RADIUS = 0.06            # 반지름 상한 (검출 오류로 과보정되는 것 방지)


class VisionToBase(Node):
    def __init__(self):
        super().__init__("vision_to_base_test")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = (
            color_stream.as_video_stream_profile().get_intrinsics()
        )

        self.model = YOLO("yolov8n.pt")
        self.history = {"x": [], "y": [], "z": []}

        # 최신 상태 (메인 루프가 쓰고, 입력 스레드가 읽는다)
        self.latest = None
        self.lock = threading.Lock()

        self.previous_camera_position = None
        self.still_since = None
        self.last_radius = 0.0

    # ── 카메라 좌표 ──
    def camera_xyz(self, color_image, depth_frame):
        results = self.model(color_image, verbose=False)[0]

        best = None
        for box in results.boxes:
            name = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            if name == TARGET_CLASS and confidence >= CONF_THRESHOLD:
                if best is None or confidence > best[1]:
                    best = (box, confidence)

        if best is None:
            self.history = {"x": [], "y": [], "z": []}
            self.previous_camera_position = None
            self.still_since = None
            return None, color_image

        box, confidence = best
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # bbox 중앙 ROI의 median 깊이 — 중심 픽셀 하나는 가장자리·배경에 잘 걸린다
        half_w = max(int((x2 - x1) * ROI_RATIO / 2), 2)
        half_h = max(int((y2 - y1) * ROI_RATIO / 2), 2)

        depths = []
        for v in range(cy - half_h, cy + half_h + 1, 2):
            for u in range(cx - half_w, cx + half_w + 1, 2):
                if 0 <= u < 640 and 0 <= v < 480:
                    d = depth_frame.get_distance(u, v)
                    if DEPTH_MIN < d < DEPTH_MAX:
                        depths.append(d)

        if len(depths) < 5:
            return None, color_image

        depth = statistics.median(depths)

        point = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [cx, cy], depth
        )

        # 표면점 → 중심축 보정
        # bbox 폭이 물체의 지름(실루엣)에 해당한다고 보고 반지름을 추정한 뒤,
        # 카메라에서 물체로 향하는 시선 방향으로 그만큼 밀어 넣는다.
        radius = 0.0
        if RADIUS_COMPENSATION:
            width_m = (x2 - x1) * depth / self.intrinsics.fx
            radius = min(width_m / 2.0, MAX_RADIUS)

            direction = np.array(point, dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                point = (direction + radius * direction / norm).tolist()

        self.last_radius = radius

        for key, value in zip(("x", "y", "z"), point):
            self.history[key].append(value)
            if len(self.history[key]) > SMOOTH_WINDOW:
                self.history[key].pop(0)

        smoothed = np.array([
            statistics.median(self.history[key]) for key in ("x", "y", "z")
        ])

        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(
            color_image,
            (cx - half_w, cy - half_h), (cx + half_w, cy + half_h),
            (0, 255, 255), 1,
        )
        cv2.circle(color_image, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            color_image,
            f"{TARGET_CLASS} {confidence:.2f}  d={depth:.3f}m  "
            f"r={radius * 1000:.0f}mm  n={len(depths)}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )

        return smoothed, color_image

    # ── base_link 변환 ──
    def to_base(self, camera_point):
        transform = self.tf_buffer.lookup_transform(
            BASE_FRAME, CAMERA_FRAME, rclpy.time.Time()
        )

        stamp = transform.header.stamp
        age = (
            self.get_clock().now().nanoseconds * 1e-9
            - (stamp.sec + stamp.nanosec * 1e-9)
        )

        t = transform.transform.translation
        q = transform.transform.rotation

        x, y, z, w = q.x, q.y, q.z, q.w
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

        base_point = rotation @ camera_point + np.array([t.x, t.y, t.z])
        return base_point, age

    # ── 메인 영상 루프 ──
    def run(self, stop_event):
        while not stop_event.is_set():
            frames = self.align.process(self.pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            camera_point, color_image = self.camera_xyz(
                color_image, depth_frame
            )

            status, color = "NO DETECTION", (0, 0, 255)

            if camera_point is not None:
                # 팔 정지 판정 (카메라 기준 좌표가 안 변하면 정지)
                if self.previous_camera_position is not None:
                    delta = np.linalg.norm(
                        camera_point - self.previous_camera_position
                    )
                    if delta < STILL_THRESHOLD:
                        if self.still_since is None:
                            self.still_since = time.time()
                    else:
                        self.still_since = None
                self.previous_camera_position = camera_point.copy()

                try:
                    base_point, age = self.to_base(camera_point)
                except Exception:
                    base_point, age = None, None

                if base_point is None:
                    status, color = "TF WAIT", (0, 165, 255)
                elif age > MAX_TF_AGE:
                    status = f"TF STALE {age * 1000:.0f}ms"
                    color = (0, 0, 255)
                elif self.still_since is None or time.time() - self.still_since < 0.7:
                    status, color = "MOVING", (0, 165, 255)
                else:
                    status, color = "READY", (0, 255, 0)
                    with self.lock:
                        self.latest = (camera_point.copy(), base_point, age)

                if base_point is not None:
                    cv2.putText(
                        color_image,
                        f"base X={base_point[0]:+.4f} "
                        f"Y={base_point[1]:+.4f} Z={base_point[2]:+.4f}",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 0), 2,
                    )

            if status != "READY":
                with self.lock:
                    self.latest = None

            cv2.putText(
                color_image, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
            cv2.imshow("vision_to_base_test v2", color_image)
            cv2.waitKey(1)

        self.pipeline.stop()
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = VisionToBase()

    # TF는 반드시 별도 스레드에서 계속 처리해야 버퍼가 밀리지 않는다.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    stop_event = threading.Event()
    vision_thread = threading.Thread(
        target=node.run, args=(stop_event,), daemon=True
    )
    vision_thread.start()

    time.sleep(2.0)

    samples = []
    print("\n영상 창이 READY(초록)일 때 Enter를 누르세요. 종료: q + Enter\n")

    try:
        while True:
            key = input(f"[{len(samples) + 1}번째] Enter=기록, q=종료: ")

            if key.strip().lower() == "q":
                break

            with node.lock:
                latest = node.latest

            if latest is None:
                print("  READY 상태가 아닙니다. 검출·정지 상태를 확인하세요.")
                continue

            camera_point, base_point, age = latest
            samples.append(base_point)

            print(
                f"  카메라 [{camera_point[0]:+.3f} {camera_point[1]:+.3f} "
                f"{camera_point[2]:+.3f}]  →  base "
                f"X={base_point[0]:+.4f} Y={base_point[1]:+.4f} "
                f"Z={base_point[2]:+.4f}  (TF {age * 1000:.0f}ms)"
            )

    except (KeyboardInterrupt, EOFError):
        pass

    stop_event.set()
    vision_thread.join(timeout=2.0)

    if len(samples) >= 2:
        data = np.array(samples)
        mean = data.mean(axis=0)
        spread = data.max(axis=0) - data.min(axis=0)
        errors = np.linalg.norm(data - mean, axis=1)

        print("\n" + "=" * 56)
        print(f"측정 개수: {len(samples)}")
        print(
            f"평균 위치: {mean[0]:+.4f}, {mean[1]:+.4f}, {mean[2]:+.4f} m"
        )
        print(
            f"축별 산포: X={spread[0] * 1000:.1f} "
            f"Y={spread[1] * 1000:.1f} Z={spread[2] * 1000:.1f} mm"
        )
        print(f"평균 오차: {errors.mean() * 1000:.1f} mm")
        print(f"최대 오차: {errors.max() * 1000:.1f} mm")
        print(
            f"반지름 보정: {'사용' if RADIUS_COMPENSATION else '미사용'}"
        )
        print("=" * 56)

    executor.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
