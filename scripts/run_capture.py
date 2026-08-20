from capture_utils import init_camera, get_frame, save_capture

pipeline, align, depth_scale = init_camera()

def capture_one(object_name, angle_label, index):
    depth, color, intrinsics = get_frame(pipeline, align)
    ee_pose = {"placeholder": True}  # TODO: 21일 실제 pose 함수로 교체
    save_capture(depth, color, intrinsics, depth_scale, ee_pose, object_name, angle_label, index)
    input(f"{object_name}_{angle_label} 저장 완료. 다음 물체/각도로 바꾸고 Enter...")

# ===== 블록 =====
capture_one("block_1x1", "front", 1)

capture_one("block_1x2", "long_axis", 1)
capture_one("block_1x2", "short_axis", 2)

capture_one("block_L3", "front", 1)
capture_one("block_L3", "top", 2)

capture_one("block_2x2", "front", 1)
capture_one("block_2x2", "top", 2)

# ===== 캔 =====
capture_one("can", "front", 1)
capture_one("can", "side", 2)

# ===== 수세미 =====
capture_one("sponge", "front", 1)
capture_one("sponge", "short_axis", 2)

pipeline.stop()
