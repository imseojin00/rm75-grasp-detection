with open('capture_utils.py', 'r') as f:
    content = f.read()

# get_frame 함수 뒤에 새 함수 추가
insert_after = '''    return depth_image, color_image, intrinsics'''

new_function = '''    return depth_image, color_image, intrinsics


def get_median_frame(pipeline, align, num_frames=5):
    """
    여러 장의 depth 프레임을 픽셀별 median으로 합쳐서 노이즈를 줄임.
    회전되거나 모서리가 많은 물체에서 depth 노이즈가 심할 때 사용.
    color, intrinsics는 마지막 프레임 것을 사용.
    """
    depth_frames = []
    color_image = None
    intrinsics = None

    for i in range(num_frames):
        depth, color, intr = get_frame(pipeline, align)
        if depth is None:
            continue
        depth_frames.append(depth)
        color_image = color
        intrinsics = intr

    if len(depth_frames) == 0:
        return None, None, None

    depth_stack = np.stack(depth_frames, axis=0)
    median_depth = np.median(depth_stack, axis=0).astype(depth_frames[0].dtype)

    return median_depth, color_image, intrinsics'''

assert insert_after in content, "매칭 실패"
content = content.replace(insert_after, new_function, 1)

with open('capture_utils.py', 'w') as f:
    f.write(content)

print("완료")
