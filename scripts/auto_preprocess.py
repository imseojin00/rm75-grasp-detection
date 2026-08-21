from pathlib import Path
import argparse
import csv
import json
import traceback

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# ============================================================
# 경로
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "preprocessed"


# ============================================================
# 기본 전처리 파라미터
# ============================================================

# Depth 범위 [meter]
DEPTH_MIN = 0.15
DEPTH_MAX = 0.50

# 기존 depth_to_pointcloud.py에서 사용하던 ROI
ROI_U_MIN = 150
ROI_U_MAX = 500

ROI_V_MIN = 180
ROI_V_MAX = 430


# Point Cloud downsample
VOXEL_SIZE = 0.002


# RANSAC
RANSAC_DISTANCE_THRESHOLD = 0.003
RANSAC_N = 3
RANSAC_ITERATIONS = 1000


# DBSCAN
DEFAULT_DBSCAN_EPS = 0.008
DEFAULT_DBSCAN_MIN_POINTS = 20


# ============================================================
# 물체별 파라미터
# ============================================================

def get_preprocess_params(base_name):
    """
    물체 종류별 전처리 설정.

    block_*:
        RANSAC 사용
        DBSCAN eps = 0.008

    can_*:
        RANSAC 사용
        DBSCAN eps = 0.005

    sponge_*:
        RANSAC 생략
        DBSCAN eps = 0.005

    수세미는 얇아서 RANSAC 평면 제거 시
    테이블과 함께 제거될 가능성이 있으므로 RANSAC 생략.
    """

    params = {
        "skip_ransac": False,
        "dbscan_eps": DEFAULT_DBSCAN_EPS,
        "dbscan_min_points": DEFAULT_DBSCAN_MIN_POINTS,
    }

    if base_name.startswith("sponge_"):

        params["skip_ransac"] = True
        params["dbscan_eps"] = 0.005

    elif base_name.startswith("can_"):

        params["dbscan_eps"] = 0.005

    return params


# ============================================================
# 전체 데이터 자동 탐색
# ============================================================

def find_all_samples(data_dir):
    """
    *_meta.json 하나를 촬영 데이터 1세트로 간주.

    예:
        block_1x1_01_front.npy
        block_1x1_01_front_color.npy
        block_1x1_01_front_meta.json

    -> base_name:
        block_1x1_01_front
    """

    base_names = sorted(
        path.name.removesuffix("_meta.json")
        for path in data_dir.glob("*_meta.json")
    )

    return base_names


# ============================================================
# 데이터 로드
# ============================================================

def load_sample(base_name, data_dir):

    depth_path = data_dir / f"{base_name}.npy"
    color_path = data_dir / f"{base_name}_color.npy"
    meta_path = data_dir / f"{base_name}_meta.json"

    if not depth_path.exists():
        raise FileNotFoundError(
            f"Depth 파일이 없습니다: {depth_path}"
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta 파일이 없습니다: {meta_path}"
        )

    depth = np.load(depth_path)

    color = None

    if color_path.exists():
        color = np.load(color_path)

    with open(
        meta_path,
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    return depth, color, meta


# ============================================================
# Depth → Point Cloud
# ============================================================

def depth_to_pointcloud(depth, color, meta):

    if depth.ndim != 2:
        raise ValueError(
            f"Depth shape가 2D가 아닙니다: {depth.shape}"
        )

    intr = meta["intrinsics"]

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    depth_scale = float(meta["depth_scale"])

    # Depth → meter
    depth_m = (
        depth.astype(np.float32)
        * depth_scale
    )

    height, width = depth.shape

    # 이미지 좌표
    v, u = np.indices(
        (height, width)
    )

    # --------------------------------------------------------
    # Depth range + ROI
    # --------------------------------------------------------

    valid = (
        np.isfinite(depth_m)
        & (depth_m >= DEPTH_MIN)
        & (depth_m <= DEPTH_MAX)
        & (u >= ROI_U_MIN)
        & (u <= min(ROI_U_MAX, width - 1))
        & (v >= ROI_V_MIN)
        & (v <= min(ROI_V_MAX, height - 1))
    )

    z = depth_m[valid]

    if len(z) == 0:
        raise RuntimeError(
            "ROI/Depth 범위 적용 후 유효한 point가 없습니다."
        )

    x = (
        (u[valid].astype(np.float32) - cx)
        * z
        / fx
    )

    y = (
        (v[valid].astype(np.float32) - cy)
        * z
        / fy
    )

    points = np.stack(
        (x, y, z),
        axis=1,
    )

    # --------------------------------------------------------
    # Open3D PointCloud
    # --------------------------------------------------------

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            points.astype(np.float64)
        )
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    if (
        color is not None
        and color.shape[:2] == depth.shape
    ):

        colors = color[valid].astype(
            np.float64
        )

        if colors.ndim == 2 and colors.shape[1] >= 3:

            colors = colors[:, :3]

            if colors.max() > 1.0:
                colors /= 255.0

            pcd.colors = (
                o3d.utility.Vector3dVector(
                    colors
                )
            )

    return pcd


# ============================================================
# Voxel downsample
# ============================================================

def downsample_pointcloud(pcd):

    if len(pcd.points) == 0:
        raise RuntimeError(
            "다운샘플링할 point가 없습니다."
        )

    downsampled = pcd.voxel_down_sample(
        voxel_size=VOXEL_SIZE
    )

    if len(downsampled.points) == 0:
        raise RuntimeError(
            "Voxel downsample 후 point가 없습니다."
        )

    return downsampled


# ============================================================
# RANSAC 평면 제거
# ============================================================

def remove_table_ransac(pcd):

    if len(pcd.points) < RANSAC_N:

        raise RuntimeError(
            "RANSAC 수행에 필요한 point 수가 부족합니다."
        )

    plane_model, inliers = (
        pcd.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_ITERATIONS,
        )
    )

    if len(inliers) == 0:

        raise RuntimeError(
            "RANSAC 평면을 찾지 못했습니다."
        )

    plane = pcd.select_by_index(
        inliers
    )

    remaining = pcd.select_by_index(
        inliers,
        invert=True,
    )

    return (
        remaining,
        plane,
        plane_model,
    )


# ============================================================
# DBSCAN
# ============================================================

def run_dbscan(
    pcd,
    eps,
    min_points,
):

    if len(pcd.points) == 0:

        raise RuntimeError(
            "DBSCAN 입력 point가 없습니다."
        )

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
    )

    if len(labels) == 0:

        raise RuntimeError(
            "DBSCAN 결과가 없습니다."
        )

    cluster_labels = labels[
        labels >= 0
    ]

    if len(cluster_labels) == 0:

        raise RuntimeError(
            "DBSCAN에서 유효한 cluster를 찾지 못했습니다."
        )

    return labels


# ============================================================
# Cluster 후보 계산
# ============================================================

def build_cluster_candidates(
    pcd,
    labels,
):

    points = np.asarray(
        pcd.points
    )

    cluster_ids = sorted(
        int(label)
        for label in np.unique(labels)
        if label >= 0
    )

    candidates = []

    for cluster_id in cluster_ids:

        indices = np.where(
            labels == cluster_id
        )[0]

        if len(indices) == 0:
            continue

        cluster_points = points[
            indices
        ]

        center = np.mean(
            cluster_points,
            axis=0,
        )

        # 카메라 광축 기준으로
        # 화면 중심과 가까운 정도
        center_distance = float(
            np.linalg.norm(
                center[:2]
            )
        )

        candidates.append(
            (
                cluster_id,
                len(indices),
                center_distance,
                center,
            )
        )

    return candidates


# ============================================================
# 물체 Cluster 선택
# ============================================================

def select_object_cluster(
    pcd,
    labels,
    base_name,
):

    candidates = (
        build_cluster_candidates(
            pcd,
            labels,
        )
    )

    if not candidates:

        raise RuntimeError(
            "물체 후보 cluster가 없습니다."
        )

    # ========================================================
    # 수세미
    # ========================================================
    #
    # 수세미는 얇아서 point 수가 상대적으로 작을 수 있음.
    # 따라서 cluster 크기 필터를 강하게 걸지 않고
    # 카메라 중심에 가까운 cluster 선택.
    #

    if base_name.startswith("sponge_"):

        selected = min(
            candidates,
            key=lambda x: x[2],
        )

    # ========================================================
    # 블록 / 캔
    # ========================================================
    #
    # 너무 작은 잡음 cluster가 중심에 있다는 이유만으로
    # 선택되는 것을 방지.
    #

    else:

        max_count = max(
            candidate[1]
            for candidate in candidates
        )

        min_allowed_count = max(
            100,
            max_count * 0.25,
        )

        filtered_candidates = [
            candidate
            for candidate in candidates
            if candidate[1]
            >= min_allowed_count
        ]

        # 혹시 필터 결과가 비면 원래 후보 사용
        if not filtered_candidates:
            filtered_candidates = candidates

        selected = min(
            filtered_candidates,
            key=lambda x: x[2],
        )

    (
        target_label,
        target_count,
        target_distance,
        target_center,
    ) = selected

    object_indices = np.where(
        labels == target_label
    )[0].tolist()

    object_pcd = (
        pcd.select_by_index(
            object_indices
        )
    )

    return (
        target_label,
        object_pcd,
        candidates,
    )


# ============================================================
# Cluster 색상 Point Cloud 생성
# ============================================================

def make_cluster_colored_pcd(
    pcd,
    labels,
):

    colored_pcd = o3d.geometry.PointCloud(
        pcd
    )

    colors = np.zeros(
        (len(labels), 3),
        dtype=np.float64,
    )

    palette = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0],
        ]
    )

    for label in np.unique(labels):

        if label == -1:
            colors[
                labels == label
            ] = [0.0, 0.0, 0.0]

        else:

            colors[
                labels == label
            ] = palette[
                int(label)
                % len(palette)
            ]

    colored_pcd.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    return colored_pcd


# ============================================================
# PNG 시각화 저장
# ============================================================

def save_pointcloud_png(
    pcd,
    save_path,
    title,
):

    points = np.asarray(
        pcd.points
    )

    if len(points) == 0:
        return

    # 너무 많으면 시각화용으로만 샘플링
    max_points = 30000

    if len(points) > max_points:

        rng = np.random.default_rng(
            0
        )

        indices = rng.choice(
            len(points),
            size=max_points,
            replace=False,
        )

        points = points[
            indices
        ]

    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    plt.tight_layout()

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 한 촬영 데이터 전처리
# ============================================================

def preprocess_one(
    base_name,
    data_dir,
    output_root,
    visualize=False,
):

    print()
    print("-" * 70)
    print(f"전처리: {base_name}")
    print("-" * 70)

    params = get_preprocess_params(
        base_name
    )

    output_dir = (
        output_root
        / base_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 데이터 로드
    # ========================================================

    depth, color, meta = load_sample(
        base_name,
        data_dir,
    )

    # ========================================================
    # 2. Depth → Point Cloud
    # ========================================================

    raw_pcd = depth_to_pointcloud(
        depth,
        color,
        meta,
    )

    raw_count = len(
        raw_pcd.points
    )

    print(
        f"원본 point 수       : {raw_count}"
    )

    raw_path = (
        output_dir
        / f"{base_name}_raw.ply"
    )

    o3d.io.write_point_cloud(
        str(raw_path),
        raw_pcd,
    )

    # ========================================================
    # 3. Downsample
    # ========================================================

    pcd = downsample_pointcloud(
        raw_pcd
    )

    downsample_count = len(
        pcd.points
    )

    print(
        f"다운샘플 point 수   : {downsample_count}"
    )

    # ========================================================
    # 4. RANSAC
    # ========================================================

    plane_count = 0
    plane_model = None

    if params["skip_ransac"]:

        # ----------------------------------------------------
        # sponge
        # ----------------------------------------------------
        #
        # 수세미는 얇아서 테이블 평면과 같이 제거될
        # 가능성이 있으므로 RANSAC을 생략한다.
        #

        remaining = pcd

        print(
            "RANSAC             : 생략 (sponge)"
        )

    else:

        # ----------------------------------------------------
        # block / can
        # ----------------------------------------------------

        (
            remaining,
            plane,
            plane_model,
        ) = remove_table_ransac(
            pcd
        )

        plane_count = len(
            plane.points
        )

        print(
            f"RANSAC 평면 point  : {plane_count}"
        )

        print(
            f"평면 제거 후       : {len(remaining.points)}"
        )

        no_plane_path = (
            output_dir
            / f"{base_name}_no_plane.ply"
        )

        o3d.io.write_point_cloud(
            str(no_plane_path),
            remaining,
        )

        # RANSAC 확인용 저장
        plane_check = o3d.geometry.PointCloud(
            plane
        )

        plane_check.paint_uniform_color(
            [1.0, 0.0, 0.0]
        )

        o3d.io.write_point_cloud(
            str(
                output_dir
                / f"{base_name}_plane.ply"
            ),
            plane_check,
        )

    # ========================================================
    # 5. DBSCAN
    # ========================================================

    labels = run_dbscan(
        remaining,
        eps=params["dbscan_eps"],
        min_points=params[
            "dbscan_min_points"
        ],
    )

    cluster_ids = sorted(
        int(label)
        for label in np.unique(labels)
        if label >= 0
    )

    noise_count = int(
        np.sum(labels == -1)
    )

    print(
        f"DBSCAN eps         : {params['dbscan_eps']}"
    )

    print(
        f"DBSCAN cluster 수  : {len(cluster_ids)}"
    )

    print(
        f"noise point 수     : {noise_count}"
    )

    for cluster_id in cluster_ids:

        cluster_count = int(
            np.sum(
                labels == cluster_id
            )
        )

        print(
            f"  cluster {cluster_id}: "
            f"{cluster_count} points"
        )

    # ========================================================
    # 6. 물체 cluster 선택
    # ========================================================

    (
        selected_cluster,
        object_pcd,
        candidates,
    ) = select_object_cluster(
        remaining,
        labels,
        base_name,
    )

    object_count = len(
        object_pcd.points
    )

    print(
        f"선택된 물체 cluster: "
        f"{selected_cluster}"
    )

    print(
        f"물체 point 수      : "
        f"{object_count}"
    )

    # ========================================================
    # 7. 결과 저장
    # ========================================================

    object_path = (
        output_dir
        / f"{base_name}_object.ply"
    )

    o3d.io.write_point_cloud(
        str(object_path),
        object_pcd,
    )

    # DBSCAN labels
    np.save(
        output_dir
        / f"{base_name}_dbscan_labels.npy",
        labels,
    )

    # Cluster 색상 PLY
    cluster_pcd = (
        make_cluster_colored_pcd(
            remaining,
            labels,
        )
    )

    cluster_path = (
        output_dir
        / f"{base_name}_clusters.ply"
    )

    o3d.io.write_point_cloud(
        str(cluster_path),
        cluster_pcd,
    )

    # PNG
    save_pointcloud_png(
        remaining,
        output_dir
        / f"{base_name}_after_preprocess.png",
        f"{base_name} - DBSCAN input",
    )

    save_pointcloud_png(
        object_pcd,
        output_dir
        / f"{base_name}_object.png",
        f"{base_name} - selected object",
    )

    # ========================================================
    # 8. 필요하면 GUI 시각화
    # ========================================================

    if visualize:

        if not params["skip_ransac"]:

            plane_vis = (
                o3d.geometry.PointCloud(
                    plane
                )
            )

            plane_vis.paint_uniform_color(
                [1.0, 0.0, 0.0]
            )

            o3d.visualization.draw_geometries(
                [
                    plane_vis,
                    remaining,
                ],
                window_name=(
                    f"{base_name} - RANSAC"
                ),
            )

        o3d.visualization.draw_geometries(
            [cluster_pcd],
            window_name=(
                f"{base_name} - DBSCAN"
            ),
        )

        o3d.visualization.draw_geometries(
            [object_pcd],
            window_name=(
                f"{base_name} - Object"
            ),
        )

    # ========================================================
    # 9. 결과 JSON
    # ========================================================

    result = {

        "base_name":
            base_name,

        "status":
            "OK",

        "raw_points":
            int(raw_count),

        "downsampled_points":
            int(downsample_count),

        "skip_ransac":
            bool(
                params[
                    "skip_ransac"
                ]
            ),

        "ransac_plane_points":
            int(plane_count),

        "remaining_points":
            int(
                len(
                    remaining.points
                )
            ),

        "plane_model":
            (
                [
                    float(value)
                    for value
                    in plane_model
                ]
                if plane_model
                is not None
                else None
            ),

        "dbscan_eps":
            float(
                params[
                    "dbscan_eps"
                ]
            ),

        "dbscan_min_points":
            int(
                params[
                    "dbscan_min_points"
                ]
            ),

        "cluster_count":
            len(cluster_ids),

        "noise_points":
            noise_count,

        "selected_cluster":
            int(
                selected_cluster
            ),

        "object_points":
            int(
                object_count
            ),

        "clusters": [],
    }

    for (
        cluster_id,
        count,
        center_distance,
        center,
    ) in candidates:

        result["clusters"].append(
            {
                "cluster_id":
                    int(cluster_id),

                "points":
                    int(count),

                "center_distance":
                    float(
                        center_distance
                    ),

                "center": [
                    float(value)
                    for value in center
                ],
            }
        )

    with open(
        output_dir
        / "preprocess_result.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return result


# ============================================================
# 전체 자동 전처리
# ============================================================

def preprocess_all(
    data_dir,
    output_dir,
    visualize=False,
):

    data_dir = Path(
        data_dir
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 데이터 자동 탐색
    # --------------------------------------------------------

    base_names = find_all_samples(
        data_dir
    )

    if not base_names:

        raise RuntimeError(
            f"{data_dir}에서 *_meta.json 데이터를 찾지 못했습니다."
        )

    print()
    print("=" * 70)
    print("전체 자동 전처리 시작")
    print("=" * 70)

    print(
        f"데이터 폴더 : {data_dir}"
    )

    print(
        f"출력 폴더   : {output_dir}"
    )

    print(
        f"전체 데이터 : {len(base_names)}"
    )

    print("=" * 70)

    results = []

    # ========================================================
    # 모든 데이터 자동 처리
    # ========================================================

    for index, base_name in enumerate(
        base_names,
        start=1,
    ):

        print()
        print("=" * 70)

        print(
            f"[{index}/{len(base_names)}] "
            f"{base_name}"
        )

        print("=" * 70)

        try:

            result = preprocess_one(
                base_name=base_name,
                data_dir=data_dir,
                output_root=output_dir,
                visualize=visualize,
            )

            results.append(
                result
            )

        except Exception as error:

            print()
            print(
                f"[FAIL] {base_name}"
            )

            print(
                f"원인: {error}"
            )

            traceback.print_exc()

            results.append(
                {
                    "base_name":
                        base_name,

                    "status":
                        "FAIL",

                    "selected_cluster":
                        "",

                    "object_points":
                        "",

                    "error":
                        str(error),
                }
            )

    # ========================================================
    # CSV 결과 저장
    # ========================================================

    csv_path = (
        output_dir
        / "preprocess_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_name",
                "status",
                "selected_cluster",
                "object_points",
            ],
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    "base_name":
                        result.get(
                            "base_name",
                            "",
                        ),

                    "status":
                        result.get(
                            "status",
                            "",
                        ),

                    "selected_cluster":
                        result.get(
                            "selected_cluster",
                            "",
                        ),

                    "object_points":
                        result.get(
                            "object_points",
                            "",
                        ),
                }
            )

    # ========================================================
    # 전체 JSON 결과
    # ========================================================

    json_path = (
        output_dir
        / "preprocess_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # 결과 요약
    # ========================================================

    ok_count = sum(
        result.get("status")
        == "OK"
        for result in results
    )

    fail_count = (
        len(results)
        - ok_count
    )

    print()
    print("=" * 70)

    print(
        "전체 자동 전처리 완료"
    )

    print("=" * 70)

    print(
        f"전체 : {len(results)}"
    )

    print(
        f"성공 : {ok_count}"
    )

    print(
        f"실패 : {fail_count}"
    )

    print(
        f"CSV  : {csv_path}"
    )

    print(
        f"JSON : {json_path}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Depth → PointCloud → RANSAC → "
            "DBSCAN → Object Extraction "
            "전체 자동 전처리"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="원본 data 폴더",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="전처리 결과 폴더",
    )

    parser.add_argument(
        "--vis",
        action="store_true",
        help=(
            "각 단계 Open3D GUI 시각화. "
            "전체 자동 처리에서는 사용하지 않는 것을 권장"
        ),
    )

    args = parser.parse_args()

    preprocess_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        visualize=args.vis,
    )
