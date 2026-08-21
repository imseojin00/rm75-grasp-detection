import sys
import numpy as np
from find_grasp import load_pointcloud

base_name = sys.argv[1] if len(sys.argv) > 1 else "block_1x1_01_front"

pcd, meta = load_pointcloud(base_name)
points = np.asarray(pcd.points)

print(f"전체 점 개수: {len(points)}")
print(f"z값 최소: {points[:,2].min():.3f}")
print(f"z값 최대: {points[:,2].max():.3f}")

mask = points[:,2] < 0.5
close_points = points[mask]
print(f"\n0.5m 이내 점 개수: {len(close_points)}")

hist, bin_edges = np.histogram(close_points[:,2], bins=20)
for i in range(len(hist)):
    print(f"{bin_edges[i]:.3f} ~ {bin_edges[i+1]:.3f} : {hist[i]}개")
