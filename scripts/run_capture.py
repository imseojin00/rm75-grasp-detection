# scripts/run_capture.py
from capture_utils import init_camera, get_frame, save_capture

pipeline, align, depth_scale = init_camera()

# 내일은 이 부분만 바꿔가며 여러 번 실행
depth, color, intrinsics = get_frame(pipeline, align)

# ee_pose는 범열 쪽 pose 함수 연결 필요 (21일 예정이라 지금은 더미값)
ee_pose = {"placeholder": True}

save_capture(depth, intrinsics, ee_pose, "block", "front", 1)

pipeline.stop()
