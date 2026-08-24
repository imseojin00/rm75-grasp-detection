import sys
import numpy as np
import open3d as o3d

from preprocess import preprocess


# ============================================================
# 설정
# ============================================================

GRIPPER_MIN = 0.005   # 0.5 cm
GRIPPER_MAX = 0.065   # 6.5 cm

NORMAL_RADIUS = 0.005
NORMAL_MAX_NN = 30

NUM_CONTACT_CANDIDATES = 80

ANTIPODAL_THRESHOLD = -0.8


# ============================================================
# 1. PCA
# ============================================================

def compute_pca_axes(pcd):
    """
    수직 파지 전용: z(카메라 깊이) 무시, x-y 평면에서만 PCA.
    범열 2026-08-21 제안 반영.
    """
    points = np.asarray(pcd.points)
    if len(points) < 10:
        return None, None, None
    center = points.mean(axis=0)
    xy = points[:, :2] - center[:2]
    cov_xy = np.cov(xy.T)
    eigvals_xy, eigvecs_xy = np.linalg.eigh(cov_xy)
    print("\n[2D PCA 고유값 (x-y 평면)]")
    for k in range(2):
        print(f"  축 {k}: {eigvals_xy[k]:.8f}")
    short_axis = np.array([eigvecs_xy[0, 0], eigvecs_xy[1, 0], 0.0])
    long_axis = np.array([eigvecs_xy[0, 1], eigvecs_xy[1, 1], 0.0])
    approach_axis = np.array([0.0, 0.0, 1.0])
    eigvecs = np.column_stack([short_axis, long_axis, approach_axis])
    print("\n[PCA 축 방향]")
    print(f"  축 0 (짧은축): {eigvecs[:, 0]}")
    print(f"  축 1 (긴축): {eigvecs[:, 1]}")
    print(f"  축 2 (approach): {eigvecs[:, 2]}")
    return center, eigvecs, points
# ============================================================
# 2. Surface Normal
# ============================================================

def estimate_normals(pcd, center):

    points = np.asarray(pcd.points)

    temp_pcd = o3d.geometry.PointCloud()

    temp_pcd.points = o3d.utility.Vector3dVector(points)

    temp_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN
        )
    )

    normals = np.asarray(
        temp_pcd.normals
    ).copy()

    # --------------------------------------------------------
    # normal을 물체 바깥 방향으로 정렬
    # --------------------------------------------------------

    vectors = points - center

    dots = np.sum(
        normals * vectors,
        axis=1
    )

    flip_mask = dots < 0

    normals[flip_mask] *= -1

    # --------------------------------------------------------
    # normalize
    # --------------------------------------------------------

    lengths = np.linalg.norm(
        normals,
        axis=1,
        keepdims=True
    )

    lengths[lengths < 1e-8] = 1.0

    normals /= lengths

    return normals


# ============================================================
# 3. 특정 축의 전체 폭
# ============================================================

def compute_axis_width(
    points,
    center,
    axis
):

    axis = axis / np.linalg.norm(axis)

    projections = (
        (points - center) @ axis
    )

    return (
        projections.max()
        - projections.min()
    )


# ============================================================
# 4. 접촉 후보 추출
# ============================================================

def get_contact_candidates(
    points,
    center,
    grasp_axis,
    num_candidates=NUM_CONTACT_CANDIDATES
):

    grasp_axis = (
        grasp_axis
        / np.linalg.norm(grasp_axis)
    )

    projections = (
        (points - center)
        @ grasp_axis
    )

    sorted_indices = np.argsort(
        projections
    )

    num_candidates = min(
        num_candidates,
        len(points) // 2
    )

    low_indices = (
        sorted_indices[:num_candidates]
    )

    high_indices = (
        sorted_indices[-num_candidates:]
    )

    return low_indices, high_indices


# ============================================================
# 5. Antipodal grasp 후보 검색
# ============================================================

def search_antipodal_pairs(
    points,
    normals,
    center,
    grasp_axis,
    gripper_min,
    gripper_max,
    axis_width=None,
    num_candidates=NUM_CONTACT_CANDIDATES
):

    low_indices, high_indices = (
        get_contact_candidates(
            points,
            center,
            grasp_axis,
            num_candidates
        )
    )

    candidates = []

    # --------------------------------------------------------
    # 양쪽 contact 후보 조합
    # --------------------------------------------------------

    for idx_a in low_indices:

        point_a = points[idx_a]
        normal_a = normals[idx_a]

        for idx_b in high_indices:

            point_b = points[idx_b]
            normal_b = normals[idx_b]

            difference = (
                point_b - point_a
            )

            width = np.linalg.norm(
                difference
            )

            if width < 1e-8:
                continue

            # ------------------------------------------------
            # gripper 범위
            # ------------------------------------------------

            if not (
                gripper_min
                <= width
                <= gripper_max
            ):
                continue

            # ------------------------------------------------
            # contact A → B 방향
            # ------------------------------------------------

            pair_axis = (
                difference / width
            )

            # ------------------------------------------------
            # normal 방향
            # ------------------------------------------------

            normal_dot = np.dot(
                normal_a,
                normal_b
            )

            # ------------------------------------------------
            # normal과 grasp 방향 정렬 정도
            # ------------------------------------------------

            alignment_a = abs(
                np.dot(
                    normal_a,
                    pair_axis
                )
            )

            alignment_b = abs(
                np.dot(
                    normal_b,
                    pair_axis
                )
            )

            alignment = (
                alignment_a
                + alignment_b
            ) / 2.0

            # ------------------------------------------------
            # antipodal score
            # ------------------------------------------------

            antipodal_score = (
                -normal_dot
            )

            # ------------------------------------------------
            # width가 물체 전체 폭(axis_width)에 가까울수록 높은 점수
            # (단일 시점 depth 노이즈로 우연히 좁은 지점이
            #  antipodal처럼 보이는 것을 방지, 범열 제안 반영)
            # ------------------------------------------------

            if axis_width is not None and axis_width > 1e-6:
                width_similarity = 1.0 - min(
                    abs(width - axis_width) / axis_width,
                    1.0
                )
            else:
                width_similarity = 0.0

            # ------------------------------------------------
            # 최종 score
            # ------------------------------------------------

            score = (
                0.5 * antipodal_score
                + 0.2 * alignment
                + 0.3 * width_similarity
            )

            candidates.append({

                "index_a":
                    int(idx_a),

                "index_b":
                    int(idx_b),

                "point_a":
                    point_a.copy(),

                "point_b":
                    point_b.copy(),

                "normal_a":
                    normal_a.copy(),

                "normal_b":
                    normal_b.copy(),

                "width":
                    float(width),

                "pair_axis":
                    pair_axis.copy(),

                "normal_dot":
                    float(normal_dot),

                "alignment_a":
                    float(alignment_a),

                "alignment_b":
                    float(alignment_b),

                "alignment":
                    float(alignment),

                "antipodal_score":
                    float(antipodal_score),

                "width_similarity":
                    float(width_similarity),

                "score":
                    float(score)
            })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates


# ============================================================
# 6. PCA 축 하나 평가
# ============================================================

def evaluate_axis(
    points,
    normals,
    center,
    axis,
    axis_index,
    gripper_min,
    gripper_max
):

    axis = (
        axis
        / np.linalg.norm(axis)
    )

    width = compute_axis_width(
        points,
        center,
        axis
    )

    print("\n--------------------------------")
    print(
        f"PCA 축 {axis_index} 평가"
    )
    print("--------------------------------")

    print(
        f"전체 projection width: "
        f"{width:.6f} m "
        f"({width * 100:.2f} cm)"
    )

    candidates = search_antipodal_pairs(
        points,
        normals,
        center,
        axis,
        gripper_min,
        gripper_max,
        axis_width=width
    )

    if len(candidates) == 0:

        print(
            "그리퍼 범위 내 후보 pair 없음"
        )

        return {
            "axis_index": axis_index,
            "axis": axis.copy(),
            "axis_width": float(width),
            "candidates": [],
            "best": None,
            "antipodal": False,
            "graspable": False
        }

    print(
        f"gripper 범위 내 후보: "
        f"{len(candidates)}개"
    )

    print("\n상위 후보 5개")

    for rank, candidate in enumerate(
        candidates[:5],
        start=1
    ):

        print(
            f"{rank}. "
            f"width={candidate['width'] * 100:.2f}cm, "
            f"dot={candidate['normal_dot']:.4f}, "
            f"align={candidate['alignment']:.4f}, "
            f"score={candidate['score']:.4f}"
        )

    best = candidates[0]

    print("\n[최고 후보]")

    print(
        f"contact A: "
        f"{best['point_a']}"
    )

    print(
        f"contact B: "
        f"{best['point_b']}"
    )

    print(
        f"width: "
        f"{best['width'] * 100:.2f} cm"
    )

    print(
        f"normal A: "
        f"{best['normal_a']}"
    )

    print(
        f"normal B: "
        f"{best['normal_b']}"
    )

    print(
        f"normal dot: "
        f"{best['normal_dot']:.4f}"
    )

    print(
        f"alignment A: "
        f"{best['alignment_a']:.4f}"
    )

    print(
        f"alignment B: "
        f"{best['alignment_b']:.4f}"
    )

    print(
        f"score: "
        f"{best['score']:.4f}"
    )

    antipodal = (
        best["normal_dot"]
        <= ANTIPODAL_THRESHOLD
    )

    graspable = (
        antipodal
        and
        gripper_min
        <= best["width"]
        <= gripper_max
    )

    print(
        f"antipodal: "
        f"{'YES' if antipodal else 'NO'}"
    )

    print(
        f"최종 grasp 가능: "
        f"{'YES' if graspable else 'NO'}"
    )

    return {
        "axis_index": axis_index,
        "axis": axis.copy(),
        "axis_width": float(width),
        "candidates": candidates,
        "best": best,
        "antipodal": antipodal,
        "graspable": graspable
    }


# ============================================================
# 7. Grasp Pose 계산
# ============================================================

def compute_grasp_pose(best_result):

    if best_result is None:
        return None

    pair = best_result["best"]

    point_a = np.asarray(pair["point_a"], dtype=float)
    point_b = np.asarray(pair["point_b"], dtype=float)

    normal_a = np.asarray(pair["normal_a"], dtype=float)
    normal_b = np.asarray(pair["normal_b"], dtype=float)

    # ========================================================
    # 1. Grasp position
    # ========================================================

    position = (point_a + point_b) / 2.0

    # ========================================================
    # 2. Closing axis
    # ========================================================

    closing_axis = point_b - point_a
    closing_norm = np.linalg.norm(closing_axis)

    if closing_norm < 1e-8:
        return None

    closing_axis /= closing_norm

    # ========================================================
    # 3. Approach axis
    # ========================================================

    approach_axis = -(normal_a + normal_b)
    approach_norm = np.linalg.norm(approach_axis)

    if approach_norm < 1e-8:
        approach_axis = -normal_a
        approach_norm = np.linalg.norm(approach_axis)

    if approach_norm < 1e-8:
        return None

    approach_axis /= approach_norm

    # closing axis와 직교하도록 보정
    approach_axis = (
        approach_axis
        - np.dot(approach_axis, closing_axis) * closing_axis
    )

    approach_norm = np.linalg.norm(approach_axis)

    if approach_norm < 1e-8:
        return None

    approach_axis /= approach_norm

    # ========================================================
    # 4. Third axis
    #
    # right-handed coordinate system
    # ========================================================

    third_axis = np.cross(
        approach_axis,
        closing_axis
    )

    third_norm = np.linalg.norm(third_axis)

    if third_norm < 1e-8:
        return None

    third_axis /= third_norm

    # ========================================================
    # 5. 다시 직교화
    # ========================================================

    approach_axis = np.cross(
        closing_axis,
        third_axis
    )

    approach_axis /= np.linalg.norm(approach_axis)

    # ========================================================
    # 6. Rotation matrix
    # ========================================================

    rotation_matrix = np.column_stack([
        closing_axis,
        third_axis,
        approach_axis
    ])

    det_R = np.linalg.det(rotation_matrix)

    print(f"[DEBUG] det(R) = {det_R:.6f}")

    if det_R < 0:
        third_axis *= -1.0

        rotation_matrix = np.column_stack([
            closing_axis,
            third_axis,
            approach_axis
        ])

        det_R = np.linalg.det(rotation_matrix)

        print(f"[DEBUG] corrected det(R) = {det_R:.6f}")

    return {
        "position": position.tolist(),
        "closing_axis": closing_axis.tolist(),
        "approach_axis": approach_axis.tolist(),
        "third_axis": third_axis.tolist(),
        "rotation_matrix": rotation_matrix.tolist(),
        "width": float(pair["width"]),
        "contact_a": point_a.tolist(),
        "contact_b": point_b.tolist(),
        "normal_a": normal_a.tolist(),
        "normal_b": normal_b.tolist()
    }

# ============================================================
# 8. Grasp 시각화
# ============================================================

def visualize_grasp(points, grasp_pose):

    """
    grasp 계산에 실제로 사용된 점군만 시각화
    + contact A/B
    + grasp center
    + contact 연결선
    + grasp coordinate frame
    """

    if grasp_pose is None:

        print(
            "[시각화] grasp pose가 없습니다."
        )

        return

    # --------------------------------------------------------
    # 1. 실제 grasp 계산에 사용된 점군
    # --------------------------------------------------------

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    # 회색 점군
    pcd.paint_uniform_color(
        [0.65, 0.65, 0.65]
    )

    # --------------------------------------------------------
    # 2. Contact A / B
    # --------------------------------------------------------

    contact_a = np.asarray(
        grasp_pose["contact_a"],
        dtype=float
    )

    contact_b = np.asarray(
        grasp_pose["contact_b"],
        dtype=float
    )

    contacts = o3d.geometry.PointCloud()

    contacts.points = (
        o3d.utility.Vector3dVector(
            np.array([
                contact_a,
                contact_b
            ])
        )
    )

    # 빨간색
    contacts.paint_uniform_color(
        [1.0, 0.0, 0.0]
    )

    # --------------------------------------------------------
    # 3. Grasp 중심
    # --------------------------------------------------------

    position = np.asarray(
        grasp_pose["position"],
        dtype=float
    )

    grasp_point = o3d.geometry.PointCloud()

    grasp_point.points = (
        o3d.utility.Vector3dVector(
            np.array([
                position
            ])
        )
    )

    # 분홍색
    grasp_point.paint_uniform_color(
        [1.0, 0.0, 1.0]
    )

    # --------------------------------------------------------
    # 4. Contact A-B 연결선
    # --------------------------------------------------------

    contact_line = o3d.geometry.LineSet()

    contact_line.points = (
        o3d.utility.Vector3dVector(
            np.array([
                contact_a,
                contact_b
            ])
        )
    )

    contact_line.lines = (
        o3d.utility.Vector2iVector([
            [0, 1]
        ])
    )

    # 빨간색
    contact_line.colors = (
        o3d.utility.Vector3dVector([
            [1.0, 0.0, 0.0]
        ])
    )

    # --------------------------------------------------------
    # 5. Grasp 좌표계
    # --------------------------------------------------------

    frame = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=0.03,
            origin=position
        )
    )

    # --------------------------------------------------------
    # 6. 시각화
    # --------------------------------------------------------

    print("\n================================")
    print("GRASP VISUALIZATION")
    print("================================")

    print(
        f"시각화 점 개수: {len(points)}"
    )

    print(
        "회색 = grasp 계산에 사용된 점군"
    )

    print(
        "빨간 점 = contact A / B"
    )

    print(
        "분홍 점 = grasp center"
    )

    print(
        "좌표축: X=빨강, Y=초록, Z=파랑"
    )

    print(
        "\nOpen3D 창을 닫으면 프로그램이 종료됩니다."
    )

    o3d.visualization.draw_geometries([
        pcd,
        contacts,
        grasp_point,
        contact_line,
        frame
    ])


# ============================================================
# 9. 전체 grasp 계산
# ============================================================

def find_grasp(
    base_name,
    gripper_min=GRIPPER_MIN,
    gripper_max=GRIPPER_MAX,
    **preprocess_kwargs
):

    # --------------------------------------------------------
    # preprocess
    # --------------------------------------------------------

    object_pcd, plane_model = preprocess(
        base_name,
        **preprocess_kwargs
    )

    if object_pcd is None:

        print(
            "preprocess 실패"
        )

        return None

    # 높이 계산은 table_z_from_tf.py(TF 기반 고정 테이블 위치)에서
    # 별도로 처리. RANSAC 기반 방식(plane_model 활용)은 물체가 크거나
    # 복잡하면(block_2x2, block_L3, can) 잘못된 평면을 잡는 문제가 있어 폐기.

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    center, eigvecs, points = (
        compute_pca_axes(
            object_pcd
        )
    )

    if center is None:

        print(
            "PCA 계산 실패"
        )

        return None

    # --------------------------------------------------------
    # Surface normal
    # --------------------------------------------------------

    print("\n==============================")
    print("Surface normal 계산")
    print("==============================")

    normals = estimate_normals(
        object_pcd,
        center
    )

    print(
        f"normal 계산 완료: "
        f"{len(normals)}개"
    )

    # --------------------------------------------------------
    # PCA 3축 평가
    # --------------------------------------------------------

    results = []

    for i in range(2):  # approach축(2) 제외

        result = evaluate_axis(
            points,
            normals,
            center,
            eigvecs[:, i],
            i,
            gripper_min,
            gripper_max
        )

        results.append(result)

    # --------------------------------------------------------
    # 유효 grasp만 선택
    # --------------------------------------------------------

    valid = [
        r
        for r in results
        if r["graspable"]
    ]

    print("\n================================")
    print("최종 후보")
    print("================================")

    if len(valid) == 0:

        print(
            "현재 조건을 만족하는 "
            "antipodal grasp 후보가 없습니다."
        )

        best = None

    else:

        for result in valid:

            pair = result["best"]

            print(
                f"축 {result['axis_index']} "
                f"→ width="
                f"{pair['width'] * 100:.2f} cm "
                f"→ dot="
                f"{pair['normal_dot']:.4f} "
                f"→ score="
                f"{pair['score']:.4f}"
            )

        best = max(
            valid,
            key=lambda r:
            r["best"]["score"]
        )

        pair = best["best"]

        print("\n================================")
        print("선택된 grasp")
        print("================================")

        print(
            f"축: "
            f"{best['axis_index']}"
        )

        print(
            f"width: "
            f"{pair['width'] * 100:.2f} cm"
        )

        print(
            f"normal dot: "
            f"{pair['normal_dot']:.4f}"
        )

        print(
            f"score: "
            f"{pair['score']:.4f}"
        )

    # --------------------------------------------------------
    # Grasp pose
    # --------------------------------------------------------

    grasp_pose = compute_grasp_pose(
        best
    )

    if grasp_pose is not None:

        print("\n================================")
        print("GRASP POSE")
        print("================================")

        print(
            f"position: "
            f"{np.array(grasp_pose['position'])}"
        )

        print(
            f"closing axis: "
            f"{np.array(grasp_pose['closing_axis'])}"
        )

        print(
            f"approach axis: "
            f"{np.array(grasp_pose['approach_axis'])}"
        )

        print(
            f"third axis: "
            f"{np.array(grasp_pose['third_axis'])}"
        )

        print("\nrotation matrix:")

        print(
            np.array(
                grasp_pose["rotation_matrix"]
            )
        )

        print(
            f"\nwidth: "
            f"{grasp_pose['width'] * 100:.2f} cm"
        )

        print(
            f"contact A: "
            f"{np.array(grasp_pose['contact_a'])}"
        )

        print(
            f"contact B: "
            f"{np.array(grasp_pose['contact_b'])}"
        )

    # --------------------------------------------------------
    # 반환
    # --------------------------------------------------------

    return {

        "center":
            center.tolist(),

        "axis_0":
            eigvecs[:, 0].tolist(),

        "axis_1":
            eigvecs[:, 1].tolist(),

        "axis_2":
            eigvecs[:, 2].tolist(),

        "results":
            results,

        "best_grasp":
            best,

        "grasp_pose":
            grasp_pose,

        # ★ 핵심 수정
        # grasp 계산에 실제 사용된 점군 저장
        "points":
            points.tolist(),

        "num_points":
            len(points)
    }


# ============================================================
# 10. 실행
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "사용법:"
        )

        print(
            "python3 find_grasp.py "
            "<object_name>"
        )

        sys.exit(1)

    base_name = sys.argv[1]

    result = find_grasp(
        base_name,
    )

    print("\n================================")
    print("실행 완료")
    print("================================")

    if result is None:

        print(
            "grasp 계산 실패"
        )

    else:

        print(
            f"입력: "
            f"{base_name}"
        )

        print(
            f"점 개수: "
            f"{result['num_points']}"
        )

        if result["best_grasp"] is None:

            print(
                "선택된 grasp가 없습니다."
            )

        else:

            best = result["best_grasp"]

            pair = best["best"]

            print(
                f"선택된 축: "
                f"{best['axis_index']}"
            )

            print(
                f"grasp width: "
                f"{pair['width'] * 100:.2f} cm"
            )

            print(
                "antipodal: True"
            )

            print(
                f"score: "
                f"{pair['score']:.4f}"
            )

            pose = result["grasp_pose"]

            if pose is not None:

                print(
                    f"grasp position: "
                    f"{pose['position']}"
                )

                print(
                    f"closing axis: "
                    f"{pose['closing_axis']}"
                )

                print(
                    f"approach axis: "
                    f"{pose['approach_axis']}"
                )

                # ★ 계산에 사용된 점군만 시각화
                visualize_grasp(
                    np.asarray(result["points"]),
                    pose
                )
