import json
import numpy as np
import open3d as o3d

# 사용할 데이터
base_name = "block_1x1_01_front"

depth_path = f"data/{base_name}.npy"
color_path = f"data/{base_name}_color.npy"
meta_path = f"data/{base_name}_meta.json"

# 데이터 불러오기
depth = np.load(depth_path)
color = np.load(color_path)

with open(meta_path, "r") as f:
    meta = json.load(f)

# 카메라 내부 파라미터
intr = meta["intrinsics"]

fx = intr["fx"]
fy = intr["fy"]
cx = intr["cx"]
cy = intr["cy"]

depth_scale = meta["depth_scale"]

print("fx, fy:", fx, fy)
print("cx, cy:", cx, cy)
print("depth scale:", depth_scale)

# Depth → meter
depth_m = depth.astype(np.float32) * depth_scale

# 이미지 좌표 생성
v, u = np.indices(depth.shape)

# Depth 거리 + 중앙 실험 영역 ROI
valid = (
    (depth_m >= 0.15) &
    (depth_m <= 0.50) &
    (u >= 150) &
    (u <= 500) &
    (v >= 180) &
    (v <= 430)
)

z = depth_m[valid]
x = (u[valid] - cx) * z / fx
y = (v[valid] - cy) * z / fy



points = np.stack((x, y, z), axis=1)

# Color도 같은 픽셀만 가져오기
colors = color[valid].astype(np.float32) / 255.0

print("생성된 Point 수:", len(points))
print("Point Cloud shape:", points.shape)

# Open3D Point Cloud 생성
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 저장
o3d.io.write_point_cloud(
    f"{base_name}.ply",
    pcd
)

print(f"저장 완료: {base_name}.ply")

# 화면 표시
o3d.visualization.draw_geometries([pcd])
