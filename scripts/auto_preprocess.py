from pathlib import Path
import argparse
import csv
import json
import traceback

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "preprocessed"


# ============================================================
# DEPTH / ROI
# ============================================================

DEPTH_MIN = 0.15
DEPTH_MAX = 0.50

ROI_U_MIN = 150
ROI_U_MAX = 500

ROI_V_MIN = 180
ROI_V_MAX = 430


# ============================================================
# POINT CLOUD
# ============================================================

VOXEL_SIZE = 0.002


# ============================================================
# RANSAC
# ============================================================

RANSAC_DISTANCE_THRESHOLD = 0.003

RANSAC_N = 3

RANSAC_ITERATIONS = 1000


# RANSAC으로 평면을 찾은 뒤
# 평면 주변 몇 mm까지 제거할지 자동 시험
PLANE_CLEARANCE_CANDIDATES = [
    0.003,
    0.004,
    0.005,
    0.006,
]


# ============================================================
# DBSCAN
# ============================================================

DBSCAN_MIN_POINTS = 20


# 물체별 eps 후보
DBSCAN_EPS_BY_TYPE = {

    "block": [
        0.006,
        0.008,
        0.010,
    ],

    "can": [
        0.004,
        0.005,
        0.006,
    ],

    "sponge": [
        0.004,
        0.005,
        0.006,
    ],

    "unknown": [
        0.006,
        0.008,
        0.010,
    ],
}


# ============================================================
# 물체 종류 판별
# ============================================================

def get_object_type(base_name):

    if base_name.startswith("sponge_"):
        return "sponge"

    if base_name.startswith("can_"):
        return "can"

    if base_name.startswith("block_"):
        return "block"

    return "unknown"


# ============================================================
# 데이터 자동 검색
# ============================================================

def find_all_samples(data_dir):

    """
    *_meta.json 하나 = 촬영 데이터 1세트

    예:
        block_1x1_01_front.npy
        block_1x1_01_front_color.npy
        block_1x1_01_front_meta.json

    -> block_1x1_01_front
    """

    base_names = sorted(

        path.name.removesuffix(
            "_meta.json"
        )

        for path
        in data_dir.glob("*_meta.json")
    )

    return base_names


# ============================================================
# 데이터 로드
# ============================================================

def load_sample(
    base_name,
    data_dir,
):

    depth_path = (
        data_dir
        / f"{base_name}.npy"
    )

    color_path = (
        data_dir
        / f"{base_name}_color.npy"
    )

    meta_path = (
        data_dir
        / f"{base_name}_meta.json"
    )

    if not depth_path.exists():

        raise FileNotFoundError(
            f"Depth 파일 없음: {depth_path}"
        )

    if not meta_path.exists():

        raise FileNotFoundError(
            f"Meta 파일 없음: {meta_path}"
        )

    depth = np.load(
        depth_path
    )

    color = None

    if color_path.exists():

        color = np.load(
            color_path
        )

    with meta_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        meta = json.load(f)

    return (
        depth,
        color,
        meta,
    )


# ============================================================
# Depth → Point Cloud
# ============================================================

def depth_to_pointcloud(
    depth,
    color,
    meta,
):

    if depth.ndim != 2:

        raise ValueError(
            f"Depth shape가 2D가 아닙니다: "
            f"{depth.shape}"
        )

    intr = meta[
        "intrinsics"
    ]

    fx = float(
        intr["fx"]
    )

    fy = float(
        intr["fy"]
    )

    cx = float(
        intr["cx"]
    )

    cy = float(
        intr["cy"]
    )

    depth_scale = float(
        meta["depth_scale"]
    )

    # --------------------------------------------------------
    # Depth → meter
    # --------------------------------------------------------

    depth_m = (
        depth.astype(
            np.float32
        )
        * depth_scale
    )

    height, width = (
        depth.shape
    )

    v, u = np.indices(
        (
            height,
            width,
        )
    )

    # --------------------------------------------------------
    # Depth 범위 + ROI
    # --------------------------------------------------------

    valid = (

        np.isfinite(
            depth_m
        )

        & (
            depth_m
            >= DEPTH_MIN
        )

        & (
            depth_m
            <= DEPTH_MAX
        )

        & (
            u
            >= ROI_U_MIN
        )

        & (
            u
            <= min(
                ROI_U_MAX,
                width - 1,
            )
        )

        & (
            v
            >= ROI_V_MIN
        )

        & (
            v
            <= min(
                ROI_V_MAX,
                height - 1,
            )
        )
    )

    z = depth_m[
        valid
    ]

    if len(z) == 0:

        raise RuntimeError(
            "ROI/Depth 범위 적용 후 "
            "유효한 point가 없습니다."
        )

    x = (
        (
            u[valid].astype(
                np.float32
            )
            - cx
        )
        * z
        / fx
    )

    y = (
        (
            v[valid].astype(
                np.float32
            )
            - cy
        )
        * z
        / fy
    )

    points = np.stack(
        (
            x,
            y,
            z,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Open3D PointCloud
    # --------------------------------------------------------

    pcd = (
        o3d.geometry.PointCloud()
    )

    pcd.points = (
        o3d.utility.Vector3dVector(
            points.astype(
                np.float64
            )
        )
    )

    # --------------------------------------------------------
    # Color
    # --------------------------------------------------------

    if (
        color is not None
        and color.shape[:2]
        == depth.shape
    ):

        colors = (
            color[valid]
            .astype(
                np.float64
            )
        )

        if (
            colors.ndim == 2
            and colors.shape[1]
            >= 3
        ):

            colors = colors[
                :,
                :3,
            ]

            if (
                colors.size
                and colors.max()
                > 1.0
            ):

                colors /= 255.0

            pcd.colors = (
                o3d.utility
                .Vector3dVector(
                    colors
                )
            )

    return pcd


# ============================================================
# Downsample
# ============================================================

def downsample_pointcloud(
    pcd,
):

    if len(pcd.points) == 0:

        raise RuntimeError(
            "다운샘플링할 "
            "point가 없습니다."
        )

    downsampled = (
        pcd.voxel_down_sample(
            voxel_size=VOXEL_SIZE
        )
    )

    if (
        len(
            downsampled.points
        )
        == 0
    ):

        raise RuntimeError(
            "Voxel downsample 후 "
            "point가 없습니다."
        )

    return downsampled


# ============================================================
# RANSAC 평면 탐색
# ============================================================

def fit_plane_ransac(
    pcd,
):

    if (
        len(pcd.points)
        < RANSAC_N
    ):

        raise RuntimeError(
            "RANSAC 수행에 필요한 "
            "point 수가 부족합니다."
        )

    (
        plane_model,
        inliers,
    ) = pcd.segment_plane(

        distance_threshold=(
            RANSAC_DISTANCE_THRESHOLD
        ),

        ransac_n=RANSAC_N,

        num_iterations=(
            RANSAC_ITERATIONS
        ),
    )

    if len(inliers) == 0:

        raise RuntimeError(
            "RANSAC 평면을 "
            "찾지 못했습니다."
        )

    plane = (
        pcd.select_by_index(
            inliers
        )
    )

    return (
        plane_model,
        inliers,
        plane,
    )


# ============================================================
# RANSAC 평면 주변까지 제거
# ============================================================

def remove_plane_with_clearance(
    pcd,
    plane_model,
    clearance,
):

    a, b, c, d = [

        float(value)

        for value
        in plane_model
    ]

    points = np.asarray(
        pcd.points
    )

    denominator = np.sqrt(
        a * a
        + b * b
        + c * c
    )

    if denominator <= 1e-12:

        raise RuntimeError(
            "잘못된 RANSAC "
            "plane model입니다."
        )

    # 각 point와 평면 사이 거리
    distances = np.abs(

        a * points[:, 0]

        + b * points[:, 1]

        + c * points[:, 2]

        + d

    ) / denominator

    # --------------------------------------------------------
    # 핵심
    #
    # 평면에서 clearance보다
    # 멀리 떨어진 점만 유지
    # --------------------------------------------------------

    keep_indices = np.where(

        distances
        > clearance

    )[0].tolist()

    remaining = (
        pcd.select_by_index(
            keep_indices
        )
    )

    return remaining


# ============================================================
# DBSCAN
# ============================================================

def run_dbscan(
    pcd,
    eps,
    min_points=DBSCAN_MIN_POINTS,
):

    if len(pcd.points) == 0:

        raise RuntimeError(
            "DBSCAN 입력 point가 없습니다."
        )

    labels = np.asarray(

        pcd.cluster_dbscan(

            eps=float(eps),

            min_points=int(
                min_points
            ),

            print_progress=False,
        )
    )

    if (
        len(labels) == 0
        or not np.any(
            labels >= 0
        )
    ):

        raise RuntimeError(
            "DBSCAN에서 유효한 "
            "cluster를 찾지 못했습니다."
        )

    return labels


# ============================================================
# Cluster 형태 분석
# ============================================================

def cluster_geometry(
    points,
):

    center = (
        points.mean(
            axis=0
        )
    )

    extent = (
        points.max(
            axis=0
        )
        -
        points.min(
            axis=0
        )
    )

    sorted_extent = (
        np.sort(
            extent
        )
    )

    smallest = float(
        sorted_extent[0]
    )

    middle = float(
        sorted_extent[1]
    )

    largest = float(
        sorted_extent[2]
    )

    flatness = (
        smallest
        / max(
            largest,
            1e-9,
        )
    )

    # --------------------------------------------------------
    # 지나치게 넓고 얇으면
    # 테이블 잔여점으로 판단
    # --------------------------------------------------------

    table_like = (

        largest > 0.14

        and smallest < 0.010

        and flatness < 0.08
    )

    return {

        "center":
            center,

        "extent":
            extent,

        "smallest_extent":
            smallest,

        "middle_extent":
            middle,

        "largest_extent":
            largest,

        "flatness":
            float(
                flatness
            ),

        "table_like":
            bool(
                table_like
            ),
    }


# ============================================================
# Cluster 후보 생성
# ============================================================

def build_cluster_candidates(
    pcd,
    labels,
):

    points = np.asarray(
        pcd.points
    )

    candidates = []

    cluster_ids = sorted(

        int(value)

        for value
        in np.unique(labels)

        if value >= 0
    )

    for cluster_id in (
        cluster_ids
    ):

        indices = np.where(

            labels
            == cluster_id

        )[0]

        if len(indices) == 0:
            continue

        cluster_points = (
            points[
                indices
            ]
        )

        geometry = (
            cluster_geometry(
                cluster_points
            )
        )

        center = (
            geometry[
                "center"
            ]
        )

        # 카메라 중심에서
        # 얼마나 떨어져 있는지
        center_distance = float(

            np.linalg.norm(
                center[:2]
            )
        )

        candidates.append(
            {

                "cluster_id":
                    cluster_id,

                "count":
                    int(
                        len(indices)
                    ),

                "center":
                    center,

                "center_distance":
                    center_distance,

                **geometry,
            }
        )

    return candidates


# ============================================================
# 물체 후보 선택
# ============================================================

def choose_object_candidate(
    candidates,
    object_type,
):

    if not candidates:

        raise RuntimeError(
            "물체 후보 cluster가 없습니다."
        )

    # --------------------------------------------------------
    # 먼저 테이블처럼 생긴 cluster 제외
    # --------------------------------------------------------

    non_table = [

        candidate

        for candidate
        in candidates

        if not candidate[
            "table_like"
        ]
    ]

    if non_table:

        pool = non_table

    else:

        pool = candidates

    max_count = max(

        candidate["count"]

        for candidate
        in pool
    )

    # --------------------------------------------------------
    # 수세미는 point 수가 작을 수 있으므로
    # 강한 크기 필터를 적용하지 않음
    # --------------------------------------------------------

    if object_type == "sponge":

        min_count = max(

            12,

            int(
                max_count
                * 0.05
            ),
        )

    else:

        min_count = max(

            30,

            int(
                max_count
                * 0.20
            ),
        )

    filtered = [

        candidate

        for candidate
        in pool

        if candidate[
            "count"
        ]
        >= min_count
    ]

    if not filtered:

        filtered = pool

    max_filtered_count = max(

        candidate["count"]

        for candidate
        in filtered
    )

    # --------------------------------------------------------
    # 점수
    #
    # 중심과 가까울수록 좋음
    # 너무 작은 cluster는 약간 패널티
    # 지나치게 길고 평평하면 패널티
    # --------------------------------------------------------

    def score(
        candidate,
    ):

        size_ratio = (

            candidate["count"]

            / max(
                max_filtered_count,
                1,
            )
        )

        size_penalty = (

            1.0
            - size_ratio

        ) * 0.03

        shape_penalty = 0.0

        if (
            candidate[
                "largest_extent"
            ]
            > 0.16
        ):

            shape_penalty += 0.20

        if (
            candidate[
                "flatness"
            ]
            < 0.02

            and candidate[
                "largest_extent"
            ]
            > 0.10
        ):

            shape_penalty += 0.20

        return (

            candidate[
                "center_distance"
            ]

            + size_penalty

            + shape_penalty
        )

    selected = min(

        filtered,

        key=score,
    )

    selected = dict(
        selected
    )

    selected[
        "selection_score"
    ] = float(
        score(selected)
    )

    return selected


# ============================================================
# 한 파라미터 조합 평가
# ============================================================

def evaluate_attempt(
    pcd,
    labels,
    object_type,
    downsampled_count,
    ransac_removed_ratio,
):

    candidates = (
        build_cluster_candidates(
            pcd,
            labels,
        )
    )

    selected = (
        choose_object_candidate(
            candidates,
            object_type,
        )
    )

    noise_ratio = float(

        np.mean(
            labels == -1
        )
    )

    cluster_count = (
        len(candidates)
    )

    object_fraction = (

        selected["count"]

        / max(
            downsampled_count,
            1,
        )
    )

    quality_score = (
        selected[
            "selection_score"
        ]
    )

    # noise가 많으면 패널티
    quality_score += (
        noise_ratio
        * 0.05
    )

    # cluster가 지나치게 많으면 패널티
    if cluster_count > 8:

        quality_score += (

            cluster_count
            - 8

        ) * 0.005

    # 물체가 지나치게 작으면 패널티
    if object_fraction < 0.01:

        quality_score += 0.20

    # 물체가 전체의 절반 이상이면
    # 테이블이 붙었을 가능성
    if object_fraction > 0.50:

        quality_score += 0.30

    if selected[
        "table_like"
    ]:

        quality_score += 1.0

    if (
        ransac_removed_ratio
        is not None

        and ransac_removed_ratio
        > 0.97
    ):

        quality_score += 0.15

    return {

        "score":
            float(
                quality_score
            ),

        "selected":
            selected,

        "candidates":
            candidates,

        "noise_ratio":
            noise_ratio,

        "cluster_count":
            cluster_count,

        "object_fraction":
            float(
                object_fraction
            ),
    }


# ============================================================
# 자동 파라미터 탐색
# ============================================================

def search_best_preprocessing(
    pcd,
    base_name,
):

    object_type = (
        get_object_type(
            base_name
        )
    )

    eps_candidates = (
        DBSCAN_EPS_BY_TYPE[
            object_type
        ]
    )

    downsampled_count = (
        len(
            pcd.points
        )
    )

    attempts = []

    # ========================================================
    # SPONGE
    #
    # RANSAC 생략
    # DBSCAN eps만 자동 시험
    # ========================================================

    if object_type == "sponge":

        for eps in (
            eps_candidates
        ):

            try:

                remaining = pcd

                labels = run_dbscan(
                    remaining,
                    eps,
                )

                evaluation = (
                    evaluate_attempt(

                        remaining,
                        labels,

                        object_type,

                        downsampled_count,

                        ransac_removed_ratio=None,
                    )
                )

                attempts.append(
                    {

                        "remaining":
                            remaining,

                        "labels":
                            labels,

                        "plane":
                            None,

                        "plane_model":
                            None,

                        "clearance":
                            None,

                        "dbscan_eps":
                            float(eps),

                        "ransac_removed_ratio":
                            None,

                        **evaluation,
                    }
                )

            except Exception:

                continue

    # ========================================================
    # BLOCK / CAN
    #
    # RANSAC 사용
    # clearance + DBSCAN eps 자동 시험
    # ========================================================

    else:

        (
            plane_model,
            _,
            plane,
        ) = fit_plane_ransac(
            pcd
        )

        for clearance in (
            PLANE_CLEARANCE_CANDIDATES
        ):

            remaining = (
                remove_plane_with_clearance(

                    pcd,

                    plane_model,

                    clearance,
                )
            )

            if (
                len(
                    remaining.points
                )
                < DBSCAN_MIN_POINTS
            ):

                continue

            removed_ratio = (

                1.0

                - (
                    len(
                        remaining.points
                    )

                    / max(
                        downsampled_count,
                        1,
                    )
                )
            )

            for eps in (
                eps_candidates
            ):

                try:

                    labels = run_dbscan(

                        remaining,

                        eps,
                    )

                    evaluation = (
                        evaluate_attempt(

                            remaining,

                            labels,

                            object_type,

                            downsampled_count,

                            ransac_removed_ratio=(
                                removed_ratio
                            ),
                        )
                    )

                    attempts.append(
                        {

                            "remaining":
                                remaining,

                            "labels":
                                labels,

                            "plane":
                                plane,

                            "plane_model":
                                plane_model,

                            "clearance":
                                float(
                                    clearance
                                ),

                            "dbscan_eps":
                                float(
                                    eps
                                ),

                            "ransac_removed_ratio":
                                float(
                                    removed_ratio
                                ),

                            **evaluation,
                        }
                    )

                except Exception:

                    continue

    if not attempts:

        raise RuntimeError(
            "사용 가능한 전처리 "
            "파라미터 조합을 "
            "찾지 못했습니다."
        )

    # 점수가 가장 작은 결과 선택
    best = min(

        attempts,

        key=lambda item:
            item["score"],
    )

    return best


# ============================================================
# 선택된 Cluster → Object PCD
# ============================================================

def select_object_pcd(
    remaining,
    labels,
    selected_cluster_id,
):

    indices = np.where(

        labels
        == selected_cluster_id

    )[0].tolist()

    object_pcd = (
        remaining.select_by_index(
            indices
        )
    )

    return object_pcd


# ============================================================
# Cluster 색상 PCD
# ============================================================

def make_cluster_colored_pcd(
    pcd,
    labels,
):

    colored = (
        o3d.geometry.PointCloud(
            pcd
        )
    )

    colors = np.zeros(

        (
            len(labels),
            3,
        ),

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

    for label in (
        np.unique(
            labels
        )
    ):

        if label == -1:

            colors[
                labels == label
            ] = [
                0.0,
                0.0,
                0.0,
            ]

        else:

            colors[
                labels == label
            ] = (

                palette[
                    int(label)
                    % len(palette)
                ]
            )

    colored.colors = (
        o3d.utility
        .Vector3dVector(
            colors
        )
    )

    return colored


# ============================================================
# DBSCAN PNG 저장
# ============================================================

def save_cluster_png(
    pcd,
    labels,
    save_path,
):

    points = np.asarray(
        pcd.points
    )

    if len(points) == 0:
        return

    fig = plt.figure(
        figsize=(
            8,
            7,
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    cmap = plt.get_cmap(
        "tab20"
    )

    for label in (
        np.unique(labels)
    ):

        mask = (
            labels
            == label
        )

        if label == -1:

            ax.scatter(

                points[
                    mask,
                    0,
                ],

                points[
                    mask,
                    1,
                ],

                points[
                    mask,
                    2,
                ],

                s=1,

                c="black",

                alpha=0.15,
            )

        else:

            ax.scatter(

                points[
                    mask,
                    0,
                ],

                points[
                    mask,
                    1,
                ],

                points[
                    mask,
                    2,
                ],

                s=2,

                color=cmap(
                    int(label)
                    % 20
                ),
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        save_path.stem
    )

    plt.tight_layout()

    fig.savefig(

        save_path,

        dpi=150,

        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Object PNG 저장
# ============================================================

def save_object_png(
    pcd,
    save_path,
):

    points = np.asarray(
        pcd.points
    )

    if len(points) == 0:
        return

    fig = plt.figure(
        figsize=(
            8,
            7,
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.scatter(

        points[:, 0],

        points[:, 1],

        points[:, 2],

        s=2,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        save_path.stem
    )

    plt.tight_layout()

    fig.savefig(

        save_path,

        dpi=150,

        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# WARN 판정
# ============================================================

def build_warnings(
    object_type,
    downsampled_count,
    best,
):

    warnings = []

    selected = (
        best["selected"]
    )

    object_points = (
        selected["count"]
    )

    object_fraction = (
        best[
            "object_fraction"
        ]
    )

    noise_ratio = (
        best[
            "noise_ratio"
        ]
    )

    if object_type == "sponge":

        minimum_points = 40

    else:

        minimum_points = 100

    # --------------------------------------------------------
    # 물체 point 너무 적음
    # --------------------------------------------------------

    if (
        object_points
        < minimum_points
    ):

        warnings.append(
            "LOW_OBJECT_POINTS"
        )

    # --------------------------------------------------------
    # 전체 point 대비 너무 작은 물체
    # --------------------------------------------------------

    if (
        object_fraction
        < 0.01
    ):

        warnings.append(
            "OBJECT_TOO_SMALL"
        )

    # --------------------------------------------------------
    # 물체가 지나치게 큼
    # --------------------------------------------------------

    if (
        object_fraction
        > 0.50
    ):

        warnings.append(
            "OBJECT_TOO_LARGE"
        )

    # --------------------------------------------------------
    # noise 지나치게 많음
    # --------------------------------------------------------

    if (
        noise_ratio
        > 0.50
    ):

        warnings.append(
            "HIGH_NOISE_RATIO"
        )

    # --------------------------------------------------------
    # cluster 너무 많음
    # --------------------------------------------------------

    if (
        best[
            "cluster_count"
        ]
        > 10
    ):

        warnings.append(
            "MANY_CLUSTERS"
        )

    # --------------------------------------------------------
    # 최종 물체가 테이블처럼 생김
    # --------------------------------------------------------

    if (
        selected[
            "table_like"
        ]
    ):

        warnings.append(
            "TABLE_LIKE_OBJECT"
        )

    # --------------------------------------------------------
    # RANSAC이 거의 다 제거
    # --------------------------------------------------------

    if (
        best[
            "ransac_removed_ratio"
        ]
        is not None

        and best[
            "ransac_removed_ratio"
        ]
        > 0.97
    ):

        warnings.append(
            "RANSAC_REMOVED_TOO_MUCH"
        )

    if (
        downsampled_count
        < 100
    ):

        warnings.append(
            "VERY_FEW_INPUT_POINTS"
        )

    return warnings


# ============================================================
# 한 샘플 전처리
# ============================================================

def preprocess_one(
    base_name,
    data_dir,
    output_root,
    visualize=False,
):

    print()

    print(
        "=" * 70
    )

    print(
        f"전처리: {base_name}"
    )

    print(
        "=" * 70
    )

    sample_dir = (

        output_root
        / base_name
    )

    sample_dir.mkdir(

        parents=True,

        exist_ok=True,
    )

    object_type = (
        get_object_type(
            base_name
        )
    )

    # ========================================================
    # LOAD
    # ========================================================

    (
        depth,
        color,
        meta,
    ) = load_sample(

        base_name,

        data_dir,
    )

    # ========================================================
    # DEPTH → POINT CLOUD
    # ========================================================

    raw_pcd = (
        depth_to_pointcloud(

            depth,

            color,

            meta,
        )
    )

    raw_count = (
        len(
            raw_pcd.points
        )
    )

    print(
        f"원본 point 수       : "
        f"{raw_count}"
    )

    # ========================================================
    # DOWNSAMPLE
    # ========================================================

    pcd = (
        downsample_pointcloud(
            raw_pcd
        )
    )

    downsampled_count = (
        len(
            pcd.points
        )
    )

    print(
        f"다운샘플 point 수   : "
        f"{downsampled_count}"
    )

    # ========================================================
    # 자동 파라미터 검색
    # ========================================================

    best = (
        search_best_preprocessing(

            pcd,

            base_name,
        )
    )

    remaining = (
        best["remaining"]
    )

    labels = (
        best["labels"]
    )

    selected = (
        best["selected"]
    )

    # ========================================================
    # Object 추출
    # ========================================================

    object_pcd = (
        select_object_pcd(

            remaining,

            labels,

            selected[
                "cluster_id"
            ],
        )
    )

    object_count = (
        len(
            object_pcd.points
        )
    )

    # ========================================================
    # 로그
    # ========================================================

    if object_type == "sponge":

        print(
            "RANSAC              : "
            "생략 (sponge)"
        )

    else:

        print(
            f"선택 clearance      : "
            f"{best['clearance']:.3f} m"
        )

        print(
            f"RANSAC 제거율       : "
            f"{best['ransac_removed_ratio'] * 100:.1f}%"
        )

    print(
        f"선택 DBSCAN eps     : "
        f"{best['dbscan_eps']:.3f}"
    )

    print(
        f"DBSCAN cluster 수   : "
        f"{best['cluster_count']}"
    )

    print(
        f"noise 비율          : "
        f"{best['noise_ratio'] * 100:.1f}%"
    )

    for candidate in (
        best["candidates"]
    ):

        print(

            f"  cluster "
            f"{candidate['cluster_id']}: "

            f"{candidate['count']} points, "

            f"center="
            f"{candidate['center_distance']:.4f}, "

            f"extent="
            f"{np.round(candidate['extent'], 4).tolist()}, "

            f"table_like="
            f"{candidate['table_like']}"
        )

    print(
        f"선택된 물체 cluster : "
        f"{selected['cluster_id']}"
    )

    print(
        f"물체 point 수       : "
        f"{object_count}"
    )

    # ========================================================
    # WARN
    # ========================================================

    warnings = (
        build_warnings(

            object_type,

            downsampled_count,

            best,
        )
    )

    if warnings:

        status = "WARN"

        print(
            "경고                : "
            + ", ".join(
                warnings
            )
        )

    else:

        status = "OK"

        print(
            "경고                : 없음"
        )

    # ========================================================
    # 저장
    # ========================================================

    o3d.io.write_point_cloud(

        str(
            sample_dir
            / f"{base_name}_raw.ply"
        ),

        raw_pcd,
    )

    o3d.io.write_point_cloud(

        str(
            sample_dir
            / f"{base_name}_remaining.ply"
        ),

        remaining,
    )

    o3d.io.write_point_cloud(

        str(
            sample_dir
            / f"{base_name}_object.ply"
        ),

        object_pcd,
    )

    # --------------------------------------------------------
    # DBSCAN 색상 PLY
    # --------------------------------------------------------

    cluster_pcd = (
        make_cluster_colored_pcd(

            remaining,

            labels,
        )
    )

    o3d.io.write_point_cloud(

        str(
            sample_dir
            / f"{base_name}_clusters.ply"
        ),

        cluster_pcd,
    )

    # --------------------------------------------------------
    # RANSAC plane 저장
    # --------------------------------------------------------

    if (
        best["plane"]
        is not None
    ):

        plane_vis = (
            o3d.geometry.PointCloud(
                best["plane"]
            )
        )

        plane_vis.paint_uniform_color(
            [
                1.0,
                0.0,
                0.0,
            ]
        )

        o3d.io.write_point_cloud(

            str(
                sample_dir
                / f"{base_name}_ransac_plane.ply"
            ),

            plane_vis,
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    np.save(

        sample_dir
        / f"{base_name}_dbscan_labels.npy",

        labels,
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    save_cluster_png(

        remaining,

        labels,

        sample_dir
        / f"{base_name}_clusters.png",
    )

    save_object_png(

        object_pcd,

        sample_dir
        / f"{base_name}_object.png",
    )

    # ========================================================
    # GUI
    # ========================================================

    if visualize:

        # 기존처럼 RANSAC/DBSCAN/Object 창을
        # 계속 열면 X11 BadWindow가 생길 수 있어서
        # 최종 Object만 표시

        o3d.visualization.draw_geometries(

            [
                object_pcd
            ],

            window_name=(
                f"{base_name} - Object"
            ),
        )

    # ========================================================
    # RESULT JSON
    # ========================================================

    result = {

        "base_name":
            base_name,

        "object_type":
            object_type,

        "status":
            status,

        "warnings":
            warnings,

        "raw_points":
            int(
                raw_count
            ),

        "downsampled_points":
            int(
                downsampled_count
            ),

        "skip_ransac":
            object_type
            == "sponge",

        "selected_clearance":
            best[
                "clearance"
            ],

        "selected_dbscan_eps":
            float(
                best[
                    "dbscan_eps"
                ]
            ),

        "ransac_removed_ratio":
            best[
                "ransac_removed_ratio"
            ],

        "remaining_points":
            int(
                len(
                    remaining.points
                )
            ),

        "cluster_count":
            int(
                best[
                    "cluster_count"
                ]
            ),

        "noise_ratio":
            float(
                best[
                    "noise_ratio"
                ]
            ),

        "selected_cluster":
            int(
                selected[
                    "cluster_id"
                ]
            ),

        "object_points":
            int(
                object_count
            ),

        "object_fraction":
            float(
                best[
                    "object_fraction"
                ]
            ),

        "object_center":
            [
                float(value)

                for value
                in selected[
                    "center"
                ]
            ],

        "object_extent":
            [
                float(value)

                for value
                in selected[
                    "extent"
                ]
            ],

        "object_table_like":
            bool(
                selected[
                    "table_like"
                ]
            ),

        "search_score":
            float(
                best[
                    "score"
                ]
            ),
    }

    with (
        sample_dir
        / "preprocess_result.json"
    ).open(

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

    base_names = (
        find_all_samples(
            data_dir
        )
    )

    if not base_names:

        raise RuntimeError(

            f"{data_dir}에서 "
            "*_meta.json 데이터를 "
            "찾지 못했습니다."
        )

    print()

    print(
        "=" * 70
    )

    print(
        "전체 자동 전처리 시작"
    )

    print(
        f"데이터 수 : "
        f"{len(base_names)}"
    )

    print(
        "=" * 70
    )

    results = []

    # ========================================================
    # 전체 데이터 순회
    # ========================================================

    for index, base_name in enumerate(

        base_names,

        start=1,
    ):

        print()

        print(
            f"[{index}/{len(base_names)}] "
            f"{base_name}"
        )

        try:

            result = (
                preprocess_one(

                    base_name=base_name,

                    data_dir=data_dir,

                    output_root=output_dir,

                    visualize=visualize,
                )
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"[FAIL] "
                f"{base_name}: "
                f"{error}"
            )

            traceback.print_exc()

            results.append(
                {

                    "base_name":
                        base_name,

                    "object_type":
                        get_object_type(
                            base_name
                        ),

                    "status":
                        "FAIL",

                    "warnings":
                        [],

                    "error":
                        str(
                            error
                        ),

                    "selected_clearance":
                        "",

                    "selected_dbscan_eps":
                        "",

                    "selected_cluster":
                        "",

                    "object_points":
                        "",

                    "noise_ratio":
                        "",

                    "object_fraction":
                        "",
                }
            )

    # ========================================================
    # CSV
    # ========================================================

    csv_path = (
        output_dir
        / "preprocess_summary.csv"
    )

    csv_fields = [

        "base_name",

        "object_type",

        "status",

        "warnings",

        "selected_clearance",

        "selected_dbscan_eps",

        "selected_cluster",

        "object_points",

        "noise_ratio",

        "object_fraction",
    ]

    with csv_path.open(

        "w",

        newline="",

        encoding="utf-8-sig",

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=csv_fields,
        )

        writer.writeheader()

        for result in (
            results
        ):

            row = {

                key:
                    result.get(
                        key,
                        "",
                    )

                for key
                in csv_fields
            }

            if isinstance(

                row["warnings"],

                list,
            ):

                row[
                    "warnings"
                ] = "|".join(

                    row[
                        "warnings"
                    ]
                )

            writer.writerow(
                row
            )

    # ========================================================
    # 전체 JSON
    # ========================================================

    json_path = (
        output_dir
        / "preprocess_summary.json"
    )

    with json_path.open(

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
    # 결과
    # ========================================================

    ok_count = sum(

        result["status"]
        == "OK"

        for result
        in results
    )

    warn_count = sum(

        result["status"]
        == "WARN"

        for result
        in results
    )

    fail_count = sum(

        result["status"]
        == "FAIL"

        for result
        in results
    )

    print()

    print(
        "=" * 70
    )

    print(
        "전체 자동 전처리 완료"
    )

    print(
        f"전체 : "
        f"{len(results)}"
    )

    print(
        f"OK   : "
        f"{ok_count}"
    )

    print(
        f"WARN : "
        f"{warn_count}"
    )

    print(
        f"FAIL : "
        f"{fail_count}"
    )

    print(
        f"CSV  : "
        f"{csv_path}"
    )

    print(
        f"JSON : "
        f"{json_path}"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description=(

            "전체 Depth 데이터 자동 전처리: "

            "Depth → PointCloud → "

            "RANSAC/skip → "

            "DBSCAN → Object"
        )
    )

    parser.add_argument(

        "--data-dir",

        type=Path,

        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(

        "--vis",

        action="store_true",

        help=(

            "각 샘플의 최종 Object를 "
            "Open3D 창으로 표시"
        ),
    )

    args = parser.parse_args()

    preprocess_all(

        data_dir=args.data_dir,

        output_dir=args.output_dir,

        visualize=args.vis,
    )
