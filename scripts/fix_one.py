import cv2
import numpy as np
from capture_utils import init_camera, get_frame, save_capture, init_robot, get_ee_pose

pipeline, align, depth_scale = init_camera()
arm = init_robot()

object_name = "block_1x2"
angle_label = "short_axis_front"
index = 2   # 원래 몇 번째였는지 확인 후 수정

print(f"{object_name}_{angle_label} 다시 찍기. 준비되면 스페이스바, q로 종료")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        display = color_image.copy()
        cv2.putText(display, f"{object_name}_{angle_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Fix", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            intrinsics_obj = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            intrinsics = {
                "fx": intrinsics_obj.fx, "fy": intrinsics_obj.fy,
                "cx": intrinsics_obj.ppx, "cy": intrinsics_obj.ppy,
                "width": intrinsics_obj.width, "height": intrinsics_obj.height,
            }
            ee_pose = get_ee_pose(arm)
            save_capture(depth_image, color_image, intrinsics, depth_scale,
                         ee_pose, object_name, angle_label, index)
            break
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
