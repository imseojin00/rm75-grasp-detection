import numpy as np
import open3d as o3d

from preprocess import preprocess


# ============================================================
# 1. PCA
# ============================================================

def compute_pca_axes(pcd):

    points = np.asarray(pcd.points)

    if len(points) < 10:
        return None, None, None

    center = points.mean(axis=0)

    centered = points - center

    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    print("\n[PCA 고유값]")

    for i in range(3):
        print(f"  축 {i}: {eigvals[i]:.8f}")

    print("\n[PCA 축 방향]")

    for i in range(3):
        print(f"  축 {i}: {eigvecs[:, i]}")

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
            radius=0.01,
            max_nn=30
        )
    )

    normals = np.asarray(
        temp_pcd.normals
    ).copy()

    # --------------------------------------------------------
    # normal 방향을 물체 바깥쪽으로 정렬
    # --------------------------------------------------------

    vectors = points - center

    dots = np.sum(
        normals * vectors,
        axis=1
    )

    flip_mask = dots < 0

    normals[flip_mask] *= -1

    # normalize

    lengths = np.linalg.norm(
        normals,
        axis=1,
        keepdims=True
    )

    lengths[lengths < 1e-8] = 1.0

    normals /= lengths

    return normals


# ============================================================
# 3. 특정 축 방향 폭
# ============================================================

def compute_axis_width(
    points,
    center,
    axis
):

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
    num_candidates=80
):

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
# 5. Antipodal 후보 검색
# ============================================================

def search_antipodal_pairs(
    points,
    normals,
    center,
    grasp_axis,
    gripper_min,
    gripper_max,
    num_candidates=80
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
    # 모든 점 쌍 검사
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
            # gripper width
            # ------------------------------------------------

            if not (
                gripper_min
                <= width
                <= gripper_max
            ):
                continue

            pair_axis = (
                difference / width
            )

            # ------------------------------------------------
            # normal끼리 얼마나 반대인지
            # ------------------------------------------------

            normal_dot = np.dot(
                normal_a,
                normal_b
            )

            # ------------------------------------------------
            # 각각의 normal이 grasp 방향과
            # 얼마나 평행한지
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
            # antipodal 정도
            #
            # dot = -1 → 완벽하게 반대
            # dot =  0 → 직각
            # dot = +1 → 같은 방향
            # ------------------------------------------------

            antipodal_score = (
                -normal_dot
            )

            # ------------------------------------------------
            # 종합 score
            # ------------------------------------------------

            score = (
                0.7 * antipodal_score
                + 0.3 * alignment
            )

            candidates.append({

                "index_a": int(idx_a),

                "index_b": int(idx_b),

                "point_a": point_a.copy(),

                "point_b": point_b.copy(),

                "normal_a": normal_a.copy(),

                "normal_b": normal_b.copy(),

                "width": float(width),

                "pair_axis": pair_axis.copy(),

                "normal_dot": float(
                    normal_dot
                ),

                "alignment_a": float(
                    alignment_a
                ),

                "alignment_b": float(
                    alignment_b
                ),

                "alignment": float(
                    alignment
                ),

                "antipodal_score": float(
                    antipodal_score
                ),

                "score": float(score)
            })

    # score 높은 순으로 정렬

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates


# ============================================================
# 6. PCA 축 평가
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

    axis = axis / np.linalg.norm(axis)

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

    # --------------------------------------------------------
    # 후보 검색
    # --------------------------------------------------------

    candidates = search_antipodal_pairs(
        points,
        normals,
        center,
        axis,
        gripper_min,
        gripper_max
    )

    if len(candidates) == 0:

        print(
            "그리퍼 범위 내 후보 pair 없음"
        )

        return {
            "axis_index": axis_index,
            "axis": axis.tolist(),
            "axis_width": float(width),
            "candidates": [],
            "best": None
        }

    # --------------------------------------------------------
    # 상위 후보 출력
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # antipodal 판정
    #
    # dot <= -0.8
    # --------------------------------------------------------

    antipodal = (
        best["normal_dot"] <= -0.8
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
        "axis": axis.tolist(),
        "axis_width": float(width),
        "candidates": candidates,
        "best": best,
        "antipodal": antipodal,
        "graspable": graspable
    }


# ============================================================
# 7. Grasp Pose 계산
# ============================================================

def compute_grasp_pose(
    best_result
):

    if best_result is None:
        return None

    pair = best_result["best"]

    point_a = np.asarray(
        pair["point_a"],
        dtype=float
    )

    point_b = np.asarray(
        pair["point_b"],
        dtype=float
    )

    normal_a = np.asarray(
        pair["normal_a"],
        dtype=float
    )

    normal_b = np.asarray(
        pair["normal_b"],
        dtype=float
    )

    # --------------------------------------------------------
    # 1. Grasp position
    #
    # 두 접촉점의 중간
    # --------------------------------------------------------

    position = (
        point_a + point_b
    ) / 2.0

    # --------------------------------------------------------
    # 2. Closing axis
    #
    # 그리퍼 손가락이 서로 닫히는 방향
    # --------------------------------------------------------

    closing_axis = (
        point_b - point_a
    )

    closing_norm = np.linalg.norm(
        closing_axis
    )

    if closing_norm < 1e-8:
        return None

    closing_axis /= closing_norm

    # --------------------------------------------------------
    # 3. Approach axis
    #
    # 두 접촉면의 바깥쪽 normal을 이용
    #
    # normal_a와 normal_b는 서로 반대이므로
    # normal_a - normal_b 방향을 사용
    # --------------------------------------------------------

    approach_axis = (
        normal_a - normal_b
    )

    approach_norm = np.linalg.norm(
        approach_axis
    )

    if approach_norm < 1e-8:

        approach_axis = normal_a.copy()

    else:

        approach_axis /= approach_norm

    # --------------------------------------------------------
    # 4. Approach axis를 closing axis에 직교화
    # --------------------------------------------------------

    approach_axis = (
        approach_axis
        - np.dot(
            approach_axis,
            closing_axis
        ) * closing_axis
    )

    approach_norm = np.linalg.norm(
        approach_axis
    )

    if approach_norm < 1e-8:

        # fallback
        approach_axis = normal_a.copy()

        approach_axis = (
            approach_axis
            - np.dot(
                approach_axis,
                closing_axis
            ) * closing_axis
        )

        approach_norm = np.linalg.norm(
            approach_axis
        )

    if approach_norm < 1e-8:
        return None

    approach_axis /= approach_norm

    # --------------------------------------------------------
    # 5. Third axis
    #
    # 오른손 좌표계
    # --------------------------------------------------------

    third_axis = np.cross(
        closing_axis,
        approach_axis
    )

    third_norm = np.linalg.norm(
        third_axis
    )

    if third_norm < 1e-8:
        return None

    third_axis /= third_norm

    # --------------------------------------------------------
    # 다시 직교화
    # --------------------------------------------------------

    approach_axis = np.cross(
        third_axis,
        closing_axis
    )

    approach_axis /= np.linalg.norm(
        approach_axis
    )

    # --------------------------------------------------------
    # 6. Rotation matrix
    #
    # column:
    #   0 = closing
    #   1 = third
    #   2 = approach
    # --------------------------------------------------------

    rotation_matrix = np.column_stack([
        closing_axis,
        third_axis,
        approach_axis
    ])

    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    return {

        "position":
            position.tolist(),

        "closing_axis":
            closing_axis.tolist(),

        "approach_axis":
            approach_axis.tolist(),

        "third_axis":
            third_axis.tolist(),

        "rotation_matrix":
            rotation_matrix.tolist(),

        "width":
            float(pair["width"]),

        "contact_a":
            point_a.tolist(),

        "contact_b":
            point_b.tolist(),

        "normal_a":
            normal_a.tolist(),

        "normal_b":
            normal_b.tolist()
    }


# ============================================================
# 8. 전체 grasp 계산
# ============================================================

def find_grasp(
    base_name,
    gripper_min=0.005,
    gripper_max=0.065,
    **preprocess_kwargs
):

    # --------------------------------------------------------
    # preprocess
    # --------------------------------------------------------

    object_pcd = preprocess(
        base_name,
        **preprocess_kwargs
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    center, eigvecs, points = (
        compute_pca_axes(
            object_pcd
        )
    )

    if center is None:
        return None

    # --------------------------------------------------------
    # normals
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
    # 각 PCA 축 평가
    # --------------------------------------------------------

    results = []

    for i in range(3):

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
    # 유효 grasp
    # --------------------------------------------------------

    valid = [
        r
        for r in results
        if r.get("graspable", False)
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

        best = max(
            valid,
            key=lambda x:
            x["best"]["score"]
        )

        for result in valid:

            pair = result["best"]

            print(
                f"축 {result['axis_index']} "
                f"→ "
                f"width="
                f"{pair['width'] * 100:.2f} cm "
                f"→ "
                f"dot="
                f"{pair['normal_dot']:.4f} "
                f"→ "
                f"score="
                f"{pair['score']:.4f}"
            )

        print("\n================================")
        print("선택된 grasp")
        print("================================")

        pair = best["best"]

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
    # Grasp Pose 계산
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
    # return
    # --------------------------------------------------------

    return {

        "position":
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

        "num_points":
            len(points)
    }


# ============================================================
# 9. 실행
# ============================================================

if __name__ == "__main__":

    import sys

    base_name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "block_1x1_01_front"
    )

    result = find_grasp(
        base_name,
        u_min=200,
        u_max=330,
        v_min=300,
        v_max=420
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

            if result["grasp_pose"] is not None:

                pose = result["grasp_pose"]

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
