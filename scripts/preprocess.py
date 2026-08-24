from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ============================================================
# DEPTH / ROI (넉넉한 1차 필터, 나머지는 RANSAC+DBSCAN이 처리)
# ============================================================
DEPTH_MIN = 0.15
DEPTH_MAX = 0.50
ROI_U_MIN = 150
ROI_U_MAX = 500
ROI_V_MIN = 180
ROI_V_MAX = 430

VOXEL_SIZE = 0.002

RANSAC_DISTANCE_THRESHOLD = 0.003
RANSAC_N = 3
RANSAC_ITERATIONS = 1000
PLANE_CLEARANCE_CANDIDATES = [0.003, 0.004, 0.005, 0.006]

DBSCAN_MIN_POINTS = 20
DBSCAN_EPS_BY_TYPE = {
    "block": [0.006, 0.008, 0.010],
    "can": [0.004, 0.005, 0.006],
    "sponge": [0.004, 0.005, 0.006],
    "unknown": [0.006, 0.008, 0.010],
}


def get_object_type(base_name):
    if base_name.startswith("sponge_"):
        return "sponge"
    if base_name.startswith("can_"):
        return "can"
    if base_name.startswith("block_"):
        return "block"
    return "unknown"


def load_sample(base_name, data_dir):
    import json
    depth_path = data_dir / f"{base_name}.npy"
    color_path = data_dir / f"{base_name}_color.npy"
    meta_path = data_dir / f"{base_name}_meta.json"

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth 파일 없음: {depth_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta 파일 없음: {meta_path}")

    depth = np.load(depth_path)
    color = np.load(color_path) if color_path.exists() else None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return depth, color, meta


def depth_to_pointcloud(depth, color, meta):
    if depth.ndim != 2:
        raise ValueError(f"Depth shape가 2D가 아닙니다: {depth.shape}")

    intr = meta["intrinsics"]
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    depth_scale = float(meta["depth_scale"])

    depth_m = depth.astype(np.float32) * depth_scale
    height, width = depth.shape
    v, u = np.indices((height, width))

    valid = (
        np.isfinite(depth_m)
        & (depth_m >= DEPTH_MIN) & (depth_m <= DEPTH_MAX)
        & (u >= ROI_U_MIN) & (u <= min(ROI_U_MAX, width - 1))
        & (v >= ROI_V_MIN) & (v <= min(ROI_V_MAX, height - 1))
    )

    z = depth_m[valid]
    if len(z) == 0:
        raise RuntimeError("ROI/Depth 범위 적용 후 유효한 point가 없습니다.")

    x = (u[valid].astype(np.float32) - cx) * z / fx
    y = (v[valid].astype(np.float32) - cy) * z / fy
    points = np.stack((x, y, z), axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if color is not None and color.shape[:2] == depth.shape:
        colors = color[valid].astype(np.float64)
        if colors.ndim == 2 and colors.shape[1] >= 3:
            colors = colors[:, :3]
            if colors.size and colors.max() > 1.0:
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def downsample_pointcloud(pcd):
    if len(pcd.points) == 0:
        raise RuntimeError("다운샘플링할 point가 없습니다.")
    downsampled = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if len(downsampled.points) == 0:
        raise RuntimeError("Voxel downsample 후 point가 없습니다.")
    return downsampled


def fit_plane_ransac(pcd):
    if len(pcd.points) < RANSAC_N:
        raise RuntimeError("RANSAC 수행에 필요한 point 수가 부족합니다.")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERATIONS,
    )
    if len(inliers) == 0:
        raise RuntimeError("RANSAC 평면을 찾지 못했습니다.")
    plane = pcd.select_by_index(inliers)
    return plane_model, inliers, plane


def remove_plane_with_clearance(pcd, plane_model, clearance):
    a, b, c, d = [float(v) for v in plane_model]
    points = np.asarray(pcd.points)
    denom = np.sqrt(a * a + b * b + c * c)
    if denom <= 1e-12:
        raise RuntimeError("잘못된 RANSAC plane model입니다.")
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / denom
    keep = np.where(distances > clearance)[0].tolist()
    return pcd.select_by_index(keep)


def run_dbscan(pcd, eps, min_points=DBSCAN_MIN_POINTS):
    if len(pcd.points) == 0:
        raise RuntimeError("DBSCAN 입력 point가 없습니다.")
    labels = np.asarray(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
    if len(labels) == 0 or not np.any(labels >= 0):
        raise RuntimeError("DBSCAN에서 유효한 cluster를 찾지 못했습니다.")
    return labels


def cluster_geometry(points):
    center = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    sorted_extent = np.sort(extent)
    smallest, middle, largest = float(sorted_extent[0]), float(sorted_extent[1]), float(sorted_extent[2])
    flatness = smallest / max(largest, 1e-9)
    table_like = largest > 0.14 and smallest < 0.010 and flatness < 0.08
    return {
        "center": center, "extent": extent,
        "smallest_extent": smallest, "middle_extent": middle, "largest_extent": largest,
        "flatness": float(flatness), "table_like": bool(table_like),
    }


def build_cluster_candidates(pcd, labels):
    points = np.asarray(pcd.points)
    candidates = []
    for cluster_id in sorted(int(v) for v in np.unique(labels) if v >= 0):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        cluster_points = points[indices]
        geometry = cluster_geometry(cluster_points)
        center = geometry["center"]
        center_distance = float(np.linalg.norm(center[:2]))
        candidates.append({"cluster_id": cluster_id, "count": int(len(indices)),
                            "center": center, "center_distance": center_distance, **geometry})
    return candidates


def choose_object_candidate(candidates, object_type):
    if not candidates:
        raise RuntimeError("물체 후보 cluster가 없습니다.")

    non_table = [c for c in candidates if not c["table_like"]]
    pool = non_table if non_table else candidates

    max_count = max(c["count"] for c in pool)

    if object_type == "sponge":
        min_count = max(12, int(max_count * 0.05))
    else:
        min_count = max(30, int(max_count * 0.20))

    filtered = [c for c in pool if c["count"] >= min_count]
    if not filtered:
        filtered = pool

    max_filtered_count = max(c["count"] for c in filtered)

    def score(c):
        size_ratio = c["count"] / max(max_filtered_count, 1)
        size_penalty = (1.0 - size_ratio) * 0.03
        shape_penalty = 0.0
        if c["largest_extent"] > 0.16:
            shape_penalty += 0.20
        if c["flatness"] < 0.02 and c["largest_extent"] > 0.10:
            shape_penalty += 0.20
        return c["center_distance"] + size_penalty + shape_penalty

    selected = min(filtered, key=score)
    selected = dict(selected)
    selected["selection_score"] = float(score(selected))
    return selected


def evaluate_attempt(pcd, labels, object_type, downsampled_count, ransac_removed_ratio):
    candidates = build_cluster_candidates(pcd, labels)
    selected = choose_object_candidate(candidates, object_type)

    noise_ratio = float(np.mean(labels == -1))
    cluster_count = len(candidates)
    object_fraction = selected["count"] / max(downsampled_count, 1)
    quality_score = selected["selection_score"]

    quality_score += noise_ratio * 0.05
    if cluster_count > 8:
        quality_score += (cluster_count - 8) * 0.005
    if object_fraction < 0.01:
        quality_score += 0.20
    if object_fraction > 0.50:
        quality_score += 0.30
    if selected["table_like"]:
        quality_score += 1.0
    if ransac_removed_ratio is not None and ransac_removed_ratio > 0.97:
        quality_score += 0.15

    return {"score": float(quality_score), "selected": selected, "candidates": candidates,
            "noise_ratio": noise_ratio, "cluster_count": cluster_count,
            "object_fraction": float(object_fraction)}


def search_best_preprocessing(pcd, base_name):
    object_type = get_object_type(base_name)
    eps_candidates = DBSCAN_EPS_BY_TYPE[object_type]
    downsampled_count = len(pcd.points)
    attempts = []

    if object_type == "sponge":
        for eps in eps_candidates:
            try:
                remaining = pcd
                labels = run_dbscan(remaining, eps)
                evaluation = evaluate_attempt(remaining, labels, object_type, downsampled_count, None)
                attempts.append({"remaining": remaining, "labels": labels,
                                  "dbscan_eps": float(eps), "ransac_removed_ratio": None,
                                  "plane_model": None, **evaluation})
            except Exception:
                continue
    else:
        plane_model, _, plane = fit_plane_ransac(pcd)
        for clearance in PLANE_CLEARANCE_CANDIDATES:
            remaining = remove_plane_with_clearance(pcd, plane_model, clearance)
            if len(remaining.points) < DBSCAN_MIN_POINTS:
                continue
            removed_ratio = 1.0 - (len(remaining.points) / max(downsampled_count, 1))
            for eps in eps_candidates:
                try:
                    labels = run_dbscan(remaining, eps)
                    evaluation = evaluate_attempt(remaining, labels, object_type, downsampled_count, removed_ratio)
                    attempts.append({"remaining": remaining, "labels": labels,
                                      "dbscan_eps": float(eps), "ransac_removed_ratio": float(removed_ratio),
                                      "plane_model": list(plane_model), **evaluation})
                except Exception:
                    continue

    if not attempts:
        raise RuntimeError("사용 가능한 전처리 파라미터 조합을 찾지 못했습니다.")

    return min(attempts, key=lambda item: item["score"])


def select_object_pcd(remaining, labels, selected_cluster_id):
    indices = np.where(labels == selected_cluster_id)[0].tolist()
    return remaining.select_by_index(indices)


# ============================================================
# find_grasp.py에서 쓰는 진입점
# ============================================================
def preprocess(base_name, data_dir=DATA_DIR, verbose=True):
    """
    어진 v2 자동 탐색 로직을 감싼 함수.
    depth+color+meta를 읽어서, RANSAC+DBSCAN 파라미터를 자동으로 탐색하고
    최적으로 선택된 물체의 point cloud를 반환.
    """
    depth, color, meta = load_sample(base_name, data_dir)
    raw_pcd = depth_to_pointcloud(depth, color, meta)
    pcd = downsample_pointcloud(raw_pcd)

    best = search_best_preprocessing(pcd, base_name)

    if verbose:
        print(f"[디버그] downsample 후 점 개수: {len(pcd.points)}")
        print(f"[디버그] 선택 dbscan_eps: {best['dbscan_eps']}")
        print(f"[디버그] cluster 수: {best['cluster_count']}, noise 비율: {best['noise_ratio']*100:.1f}%")
        print(f"[디버그] 선택된 cluster: {best['selected']['cluster_id']} ({best['selected']['count']}개)")

    object_pcd = select_object_pcd(best["remaining"], best["labels"], best["selected"]["cluster_id"])
    return object_pcd, best.get("plane_model")
