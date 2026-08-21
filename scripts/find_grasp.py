import numpy as np
from preprocess import preprocess


def find_short_axis(pcd):
    """PCA로 짧은 축 찾기 (논문의 Darboux frame 간소화 버전)"""
    points = np.asarray(pcd.points)
    if len(points) < 10:
        return None, None

    center = points.mean(axis=0)
    centered = points - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh는 오름차순 정렬 → 가장 작은 고유값의 축이 "짧은 축"
    short_axis = eigvecs[:, 0]

    return center, short_axis


def find_grasp(base_name, **preprocess_kwargs):
    """최소 버전: 어진의 전처리(RANSAC+DBSCAN) 결과에 PCA 적용
    antipodal은 22일에 추가 예정"""
    object_pcd = preprocess(base_name, **preprocess_kwargs)

    center, short_axis = find_short_axis(object_pcd)

    if center is None:
        return None

    return {
        "position": center.tolist(),
        "approach_axis": short_axis.tolist(),
        "width": 0.05,  # TODO: antipodal 계산 후 실제 값으로
        "score": 1.0,   # TODO: antipodal 완성 후 실제 점수
        "num_points": len(object_pcd.points),
    }


if __name__ == "__main__":
    import sys
    base_name = sys.argv[1] if len(sys.argv) > 1 else "block_1x1_01_front"

    result = find_grasp(base_name)
    print(f"입력: {base_name}")
    print(f"결과: {result}")
