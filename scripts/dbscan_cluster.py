import open3d as o3d
import numpy as np

# 1차 RANSAC까지만 적용된 데이터
pcd = o3d.io.read_point_cloud(
    "block_1x1_01_front_no_plane.ply"
)

print("입력 point 수:", len(pcd.points))

# DBSCAN clustering
labels = np.array(
    pcd.cluster_dbscan(
        eps=0.008,
        min_points=20,
        print_progress=True
    )
)

if len(labels) == 0:
    print("클러스터를 찾지 못했습니다.")
    exit()

max_label = labels.max()

print("찾은 cluster 수:", max_label + 1)
print("noise point 수:", np.sum(labels == -1))

# 클러스터별 point 수 출력
for i in range(max_label + 1):
    count = np.sum(labels == i)
    print(f"cluster {i}: {count} points")

# 클러스터를 서로 다른 색으로 표시
colors = np.zeros((len(labels), 3))

palette = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
]

for i in range(max_label + 1):
    colors[labels == i] = palette[i % len(palette)]

# noise는 검정색
colors[labels == -1] = [0, 0, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])
# 가장 큰 cluster를 물체 후보로 선택
cluster_sizes = []

for i in range(max_label + 1):
    count = np.sum(labels == i)
    cluster_sizes.append((i, count))

target_label = max(cluster_sizes, key=lambda x: x[1])[0]

print("선택된 물체 cluster:", target_label)

# 해당 cluster의 point만 선택
object_indices = np.where(labels == target_label)[0].tolist()
object_pcd = pcd.select_by_index(object_indices)

print("물체 point 수:", len(object_pcd.points))

# 물체만 저장
o3d.io.write_point_cloud(
    "block_1x1_01_front_object.ply",
    object_pcd
)

print("저장 완료: block_1x1_01_front_object.ply")

# 물체만 확인
o3d.visualization.draw_geometries([object_pcd])
