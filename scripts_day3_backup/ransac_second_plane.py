import open3d as o3d

# 첫 번째 평면(배경판)이 제거된 Point Cloud
pcd = o3d.io.read_point_cloud(
    "block_1x1_01_front_no_plane.ply"
)

print("입력 point 수:", len(pcd.points))

# 두 번째로 큰 평면 탐색
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.003,
    ransac_n=3,
    num_iterations=1000
)

a, b, c, d = plane_model

print("두 번째 평면:")
print(f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
print("평면 point 수:", len(inliers))

# 두 번째 평면과 나머지 분리
plane2 = pcd.select_by_index(inliers)
objects = pcd.select_by_index(inliers, invert=True)

# 확인하기 쉽게 두 번째 평면을 초록색으로 표시
plane2.paint_uniform_color([0.0, 1.0, 0.0])

# 두 번째 평면까지 제거된 데이터 저장
o3d.io.write_point_cloud(
    "block_1x1_01_front_no_planes.ply",
    objects
)

print("저장 완료: block_1x1_01_front_no_planes.ply")

o3d.visualization.draw_geometries([plane2, objects])
