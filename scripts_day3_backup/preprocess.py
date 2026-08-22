import json
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def preprocess(base_name,
                depth_min=0.1, depth_max=0.5,
                u_min=200, u_max=330, v_min=300, v_max=420,
                voxel=0.001, plane_threshold=0.003,
                dbscan_eps=0.015, dbscan_min_points=20,
                min_cluster_size=20,
                skip_ransac=True):
    """
    depth+color+meta를 읽어서 배경 제거 + 클러스터링 후,
    카메라 중심에 가장 가까운 물체 클러스터의 point cloud를 반환.
    """
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
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    depth_scale = meta["depth_scale"]

    # --- Depth → meter + ROI ---
    depth_m = depth.astype(np.float32) * depth_scale
    v, u = np.indices(depth.shape)
    valid = (
        (depth_m >= depth_min) & (depth_m <= depth_max) &
        (u >= u_min) & (u <= u_max) &
        (v >= v_min) & (v <= v_max)
    )
    z = depth_m[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack((x, y, z), axis=1)
    colors = color[valid].astype(np.float64) / 255.0

    print(f"[디버그] ROI 통과 점 개수: {len(points)}")

    if len(points) == 0:
        raise RuntimeError("ROI 안에 유효한 Point가 없습니다.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- Voxel Downsampling ---
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel)
    print(f"[디버그] voxel downsample 후 점 개수: {len(pcd_down.points)}")
    if len(pcd_down.points) < 3:
        raise RuntimeError("RANSAC을 수행하기에 Point가 너무 적습니다.")

    # --- RANSAC (평면 제거, 선택적) ---
    if skip_ransac:
        remaining = pcd_down
        print("[디버그] RANSAC 건너뜀")
    else:
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=plane_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        remaining = pcd_down.select_by_index(inliers, invert=True)
        print(f"[디버그] RANSAC 후 남은 점 개수: {len(remaining.points)}")

    if len(remaining.points) == 0:
        raise RuntimeError("전처리 이후 Point가 남지 않았습니다.")

    # --- DBSCAN ---
    labels = np.array(
        remaining.cluster_dbscan(
            eps=dbscan_eps,
            min_points=dbscan_min_points,
            print_progress=False
        ))

    if len(labels) == 0 or labels.max() < 0:
        raise RuntimeError("DBSCAN 실패 — 유효한 cluster를 찾지 못했습니다.")

    max_label = labels.max()
    print(f"[디버그] DBSCAN 클러스터 개수: {max_label + 1}")
    for i in range(max_label + 1):
        print(f"[디버그]   클러스터 {i}: {np.sum(labels == i)}개")

    # --- 물체 후보 선택 ---
    candidates = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        if len(indices) < min_cluster_size:
            continue
        cluster_pcd = remaining.select_by_index(indices.tolist())
        center = np.asarray(
            cluster_pcd.get_axis_aligned_bounding_box().get_center()
        )
        center_distance = np.sqrt(center[0]**2 + center[1]**2)
        candidates.append((i, len(indices), center_distance, center))

    if len(candidates) == 0:
        raise RuntimeError("물체 후보 cluster가 없습니다.")

    # 물체 종류에 따라 선택 기준을 다르게 적용 (어진 로직)
    if base_name.startswith("sponge_"):
        # 수세미는 얇은 자세에서 point 수가 적을 수 있으므로
        # 중심에 가장 가까운 cluster 선택
        target_label, target_count, _, target_center = min(
            candidates, key=lambda x: x[2]
        )
    else:
        # 블록/캔은 너무 작은 cluster가 중심에 가깝다는 이유만으로
        # 선택되는 것을 방지
        max_count = max(c[1] for c in candidates)
        filtered_candidates = [
            c for c in candidates
            if c[1] >= max(100, max_count * 0.25)
        ]
        target_label, target_count, _, target_center = min(
            filtered_candidates, key=lambda x: x[2]
        )

    print(f"[디버그] 선택된 클러스터: {target_label} ({target_count}개)")

    object_indices = np.where(labels == target_label)[0].tolist()
    object_pcd = remaining.select_by_index(object_indices)

    return object_pcd
