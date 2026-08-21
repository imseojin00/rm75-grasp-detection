import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d


# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("base_name", help="예: block_1x1_01_front")

# 현재 block 데이터에서 확인한 초기 파라미터
parser.add_argument("--depth-min", type=float, default=0.15)
parser.add_argument("--depth-max", type=float, default=0.50)

parser.add_argument("--u-min", type=int, default=150)
parser.add_argument("--u-max", type=int, default=500)
parser.add_argument("--v-min", type=int, default=180)
parser.add_argument("--v-max", type=int, default=430)

parser.add_argument("--voxel", type=float, default=0.002)
parser.add_argument("--plane-threshold", type=float, default=0.003)

parser.add_argument("--dbscan-eps", type=float, default=0.008)
parser.add_argument("--dbscan-min-points", type=int, default=20)

parser.add_argument(
    "--no-vis",
    action="store_true",
    help="Open3D 시각화 창을 띄우지 않음"
)

parser.add_argument(
    "--skip-ransac",
    action="store_true",
    help="RANSAC 평면 제거를 건너뜀"
)

args = parser.parse_args()
base_name = args.base_name


# --------------------------------------------------
# 1. 데이터 불러오기
# --------------------------------------------------

depth_path = DATA_DIR / f"{base_name}.npy"
color_path = DATA_DIR / f"{base_name}_color.npy"
meta_path = DATA_DIR / f"{base_name}_meta.json"

for path in [depth_path, color_path, meta_path]:
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

depth = np.load(depth_path)
color = np.load(color_path)

with open(meta_path, "r") as f:
    meta = json.load(f)

intr = meta["intrinsics"]

fx = intr["fx"]
fy = intr["fy"]
cx = intr["cx"]
cy = intr["cy"]
depth_scale = meta["depth_scale"]

print("\n===== 데이터 =====")
print("base_name:", base_name)
print("depth shape:", depth.shape)
print("color shape:", color.shape)
print("depth scale:", depth_scale)


# --------------------------------------------------
# 2. Depth → meter + ROI
# --------------------------------------------------

depth_m = depth.astype(np.float32) * depth_scale

v, u = np.indices(depth.shape)

valid = (
    (depth_m >= args.depth_min) &
    (depth_m <= args.depth_max) &
    (u >= args.u_min) &
    (u <= args.u_max) &
    (v >= args.v_min) &
    (v <= args.v_max)
)

z = depth_m[valid]
x = (u[valid] - cx) * z / fx
y = (v[valid] - cy) * z / fy

points = np.stack((x, y, z), axis=1)
colors = color[valid].astype(np.float64) / 255.0

print("\n===== ROI / Point Cloud =====")
print("ROI point 수:", len(points))

if len(points) == 0:
    raise RuntimeError("ROI 안에 유효한 Point가 없습니다.")


# --------------------------------------------------
# 3. Point Cloud 생성
# --------------------------------------------------

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

roi_output = OUTPUT_DIR / f"{base_name}_roi.ply"
o3d.io.write_point_cloud(str(roi_output), pcd)

print("ROI Point Cloud 저장:", roi_output)


# --------------------------------------------------
# 4. Voxel Downsampling
# --------------------------------------------------

pcd_down = pcd.voxel_down_sample(voxel_size=args.voxel)

print("\n===== Downsampling =====")
print("원본:", len(pcd.points))
print("다운샘플링 후:", len(pcd_down.points))

if len(pcd_down.points) < 3:
    raise RuntimeError("RANSAC을 수행하기에 Point가 너무 적습니다.")


# --------------------------------------------------
# 5. RANSAC - 필요할 때만 평면 제거
# --------------------------------------------------

if args.skip_ransac:
    print("\n===== RANSAC =====")
    print("RANSAC 건너뜀")

    # 평면을 제거하지 않고 다운샘플링 결과 그대로 사용
    remaining = pcd_down
    plane = None

else:
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=args.plane_threshold,
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, d = plane_model

    plane = pcd_down.select_by_index(inliers)
    remaining = pcd_down.select_by_index(inliers, invert=True)

    print("\n===== RANSAC =====")
    print(
        f"평면: {a:.4f}x + {b:.4f}y + "
        f"{c:.4f}z + {d:.4f} = 0"
    )
    print("평면 point 수:", len(plane.points))
    print("남은 point 수:", len(remaining.points))


# RANSAC 결과 또는 skip 결과 저장
no_plane_output = OUTPUT_DIR / f"{base_name}_no_plane.ply"

o3d.io.write_point_cloud(
    str(no_plane_output),
    remaining
)

print("평면 제거 결과 저장:", no_plane_output)


if len(remaining.points) == 0:
    raise RuntimeError(
        "전처리 이후 Point가 남지 않았습니다."
    )
# --------------------------------------------------
# 6. DBSCAN
# --------------------------------------------------

labels = np.array(
    remaining.cluster_dbscan(
        eps=args.dbscan_eps,
        min_points=args.dbscan_min_points,
        print_progress=False
    )
)

print("\n===== DBSCAN =====")

if len(labels) == 0 or labels.max() < 0:
    print("유효한 cluster를 찾지 못했습니다.")
    raise RuntimeError("DBSCAN 실패")

max_label = labels.max()

print("cluster 수:", max_label + 1)
print("noise point 수:", np.sum(labels == -1))

cluster_sizes = []

for i in range(max_label + 1):
    count = int(np.sum(labels == i))
    cluster_sizes.append((i, count))
    print(f"cluster {i}: {count} points")


# --------------------------------------------------
# 7. 가장 큰 cluster를 물체 후보로 선택
# --------------------------------------------------

target_label, target_count = max(
    cluster_sizes,
    key=lambda x: x[1]

)

# --------------------------------------------------
# 7. 물체 후보 선택
#    충분히 큰 cluster 중 카메라 중심에 가까운 것을 선택
# --------------------------------------------------

candidates = []

for i in range(max_label + 1):

    indices = np.where(labels == i)[0]

    # 너무 작은 cluster는 제외
    if len(indices) < 100:
        continue

    cluster_pcd = remaining.select_by_index(indices.tolist())

    center = np.asarray(
        cluster_pcd.get_axis_aligned_bounding_box().get_center()
    )

    # 카메라 광축(x=0, y=0)에 가까울수록 작은 값
    center_distance = np.sqrt(center[0]**2 + center[1]**2)

    candidates.append(
        (i, len(indices), center_distance, center)
    )

    print(
        f"candidate {i}: "
        f"{len(indices)} points, "
        f"center={center}, "
        f"center_distance={center_distance:.4f}"
    )


if len(candidates) == 0:
    raise RuntimeError("물체 후보 cluster가 없습니다.")


# 중심에 가장 가까운 cluster 선택
# 너무 작은 cluster가 중심에 가깝다는 이유만으로
# 물체로 선택되는 것을 방지
# 물체 종류에 따라 cluster 선택 기준을 다르게 적용
if base_name.startswith("sponge_"):
    # 수세미는 얇은 자세에서 point 수가 적을 수 있으므로
    # 중심에 가장 가까운 cluster 선택
    target_label, target_count, _, target_center = min(
        candidates,
        key=lambda x: x[2]
    )

else:
    # 블록 / 캔은 너무 작은 cluster가
    # 중심에 가깝다는 이유만으로 선택되는 것을 방지
    max_count = max(c[1] for c in candidates)

    filtered_candidates = [
        c for c in candidates
        if c[1] >= max(100, max_count * 0.25)
    ]

    target_label, target_count, _, target_center = min(
        filtered_candidates,
        key=lambda x: x[2]
    )

print("선택된 물체 cluster:", target_label)
print("선택된 cluster 중심:", target_center)

object_indices = np.where(labels == target_label)[0].tolist()
object_pcd = remaining.select_by_index(object_indices)

# 선택된 물체 cluster의 성긴 이상치 제거

object_pcd = remaining.select_by_index(object_indices)

print("물체 point 수:", len(object_pcd.points))

print("물체 point 수:", len(object_pcd.points))

# --------------------------------------------------
# 8. 시각화
# --------------------------------------------------

if not args.no_vis:

    # RANSAC 결과
    # DBSCAN 결과
    cluster_vis = copy.deepcopy(remaining)

    palette = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    cluster_colors = np.zeros((len(labels), 3))

    for i in range(max_label + 1):
        cluster_colors[labels == i] = palette[i % len(palette)]

    # noise = 검정
    cluster_colors[labels == -1] = [0.0, 0.0, 0.0]

    cluster_vis.colors = o3d.utility.Vector3dVector(
        cluster_colors
    )

    print("[창 2] DBSCAN cluster 결과")
    o3d.visualization.draw_geometries(
        [cluster_vis]
    )

    print("[창 3] 자동 선택된 물체 후보")
    o3d.visualization.draw_geometries(
        [object_pcd]
    )


print("\n===== 전처리 완료 =====")
print(base_name)
