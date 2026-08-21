import open3d as o3d

# Point Cloud 불러오기
pcd = o3d.io.read_point_cloud("block_1x1_01_front.ply")

print("원본 point 수:", len(pcd.points))

# 계산량 감소 + 잡음 완화를 위한 다운샘플링
pcd = pcd.voxel_down_sample(voxel_size=0.002)

print("다운샘플링 후:", len(pcd.points))

# RANSAC으로 가장 큰 평면 탐색
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.003,
    ransac_n=3,
    num_iterations=1000
)

a, b, c, d = plane_model

print("평면 방정식:")
print(f"{a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
print("평면 point 수:", len(inliers))

# 평면 / 나머지 점 분리
plane = pcd.select_by_index(inliers)
objects = pcd.select_by_index(inliers, invert=True)

# 어떤 면이 RANSAC으로 잡혔는지 확인하기 위해 빨간색 표시
plane.paint_uniform_color([1.0, 0.0, 0.0])

# 평면 제거 결과 저장
o3d.io.write_point_cloud(
    "block_1x1_01_front_no_plane.ply",
    objects
)

# 빨간색 = RANSAC이 평면이라고 판단한 부분
o3d.visualization.draw_geometries([plane, objects])
