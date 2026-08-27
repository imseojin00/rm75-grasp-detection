import sys
sys.path.insert(0, '.')
from capture_utils import init_camera, get_frame, init_robot, get_ee_pose, save_capture

N = 8  # 몇 장 찍을지

print("카메라 초기화...")
pipeline, align, depth_scale = init_camera()

print("로봇 연결...")
arm = init_robot()

input("스펀지를 놓고 Enter를 누르세요 (그 뒤로 촬영 자동 진행)...")

for i in range(1, N + 1):
    depth_image, color_image, intrinsics = get_frame(pipeline, align)
    ee_pose = get_ee_pose(arm)
    save_capture(depth_image, color_image, intrinsics, depth_scale, ee_pose, "sponge", f"live{i}", 0)
    print(f"  {i}/{N} 저장 완료: sponge_00_live{i}")
    if i < N:
        input(f"  다음 촬영하려면 Enter (물체 위치·각도 살짝 바꿔도 좋음)...")

pipeline.stop()
print("촬영 완료")
