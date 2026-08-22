import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# find_grasp.py에서 새로 계산된 rotation matrix
# ============================================================

R = np.array([
    [-0.54626076, -0.45631963,  0.70240414],
    [ 0.37527324, -0.88303433, -0.28181619],
    [ 0.74884523,  0.10964835,  0.65361155]
])


# ============================================================
# Rotation matrix 확인
# ============================================================

print("rotation matrix:")
print(R)

print()

det_R = np.linalg.det(R)

print(f"det(R) = {det_R:.6f}")


# ============================================================
# Quaternion 변환
# scipy 결과 순서: [x, y, z, w]
# ============================================================

q = Rotation.from_matrix(R).as_quat()

print()
print("quaternion [x, y, z, w]:")
print(q)


# ============================================================
# 다시 Rotation matrix로 복원해서 검증
# ============================================================

R_check = Rotation.from_quat(q).as_matrix()

print()
print("복원된 rotation matrix:")
print(R_check)

print()
print(
    "matrix reconstruction error:",
    np.linalg.norm(R - R_check)
)
