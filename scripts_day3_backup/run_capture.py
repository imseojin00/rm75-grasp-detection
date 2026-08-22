from capture_utils import init_camera, get_frame, save_capture

pipeline, align, depth_scale = init_camera()

def capture_one(object_name, angle_label, index):
    depth, color, intrinsics = get_frame(pipeline, align)
    ee_pose = {"placeholder": True}  # TODO: 21일 실제 pose 함수로 교체
    save_capture(depth, color, intrinsics, depth_scale, ee_pose, object_name, angle_label, index)
    input(f"[{index}] {object_name}_{angle_label} 저장 완료. 다음으로 바꾸고 Enter...")

# ===== block_1x1 (8장) =====
for angle in ["front", "side", "top", "diagonal"]:
    for i in [1, 2]:
        capture_one("block_1x1", angle, i)

# ===== block_1x2 (10장) =====
for angle in ["long_axis_front", "long_axis_diagonal", "short_axis_front", "short_axis_diagonal", "top"]:
    for i in [1, 2]:
        capture_one("block_1x2", angle, i)

# ===== block_L3 (12장) =====
for angle in ["front", "back", "top", "rot90_1", "rot90_2", "diagonal"]:
    for i in [1, 2]:
        capture_one("block_L3", angle, i)

# ===== block_2x2 (8장) =====
for angle in ["front", "top", "side", "diagonal"]:
    for i in [1, 2]:
        capture_one("block_2x2", angle, i)

# ===== can (10장) =====
for angle in ["standing_top", "standing_side", "lying_front", "lying_end", "lying_diagonal"]:
    for i in [1, 2]:
        capture_one("can", angle, i)

# ===== sponge (8장) =====
for angle in ["wide_face_front", "wide_face_back", "short_axis", "diagonal"]:
    for i in [1, 2]:
        capture_one("sponge", angle, i)

pipeline.stop()
print("전체 촬영 완료! 총 56장")
