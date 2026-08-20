import cv2
import numpy as np
import rclpy
from capture_utils import init_camera, get_frame, save_capture, init_robot, get_ee_pose

pipeline, align, depth_scale = init_camera()

print("로봇 연결 시도 중...")
arm = init_robot()
print("로봇 연결 완료")

# ===== 촬영 목록 =====
capture_list = []

for angle in ["front", "side", "top", "diagonal"]:
    for i in [1, 2]:
        capture_list.append(("block_1x1", angle, i))

for angle in ["long_axis_front", "long_axis_diagonal", "short_axis_front", "short_axis_diagonal", "top"]:
    for i in [1, 2]:
        capture_list.append(("block_1x2", angle, i))

for angle in ["front", "back", "top", "rot90_1", "rot90_2", "diagonal"]:
    for i in [1, 2]:
        capture_list.append(("block_L3", angle, i))

for angle in ["front", "top", "side", "diagonal"]:
    for i in [1, 2]:
        capture_list.append(("block_2x2", angle, i))

for angle in ["standing_top", "standing_side", "lying_front", "lying_end", "lying_diagonal"]:
    for i in [1, 2]:
        capture_list.append(("can", angle, i))

for angle in ["wide_face_front", "wide_face_back", "short_axis", "diagonal"]:
    for i in [1, 2]:
        capture_list.append(("sponge", angle, i))

# ===== 실시간 화면 + 촬영 =====
idx = 0
total = len(capture_list)

print(f"총 {total}장 예정")
print("스페이스바: 촬영 | q: 종료")

try:
    while idx < total:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        object_name, angle_label, index = capture_list[idx]
        display = color_image.copy()
        cv2.putText(display, f"[{idx+1}/{total}] {object_name}_{angle_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE: capture | q: quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Live", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            intrinsics_obj = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            intrinsics = {
                "fx": intrinsics_obj.fx,
                "fy": intrinsics_obj.fy,
                "cx": intrinsics_obj.ppx,
                "cy": intrinsics_obj.ppy,
                "width": intrinsics_obj.width,
                "height": intrinsics_obj.height,
            }
            ee_pose = get_ee_pose(arm)
            save_capture(depth_image, color_image, intrinsics, depth_scale,
                         ee_pose, object_name, angle_label, index)
            idx += 1

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    arm.destroy_node()
    rclpy.shutdown()
    print(f"종료. {idx}/{total}장 촬영됨")
