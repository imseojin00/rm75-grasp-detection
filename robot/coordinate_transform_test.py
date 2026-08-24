import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# Camera -> base_link TF
# ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
# ============================================================

T_BASE_CAMERA = np.array([
    [-0.022, -1.000, -0.003,  0.464],
    [-1.000,  0.022, -0.007,  0.090],
    [ 0.007,  0.003, -1.000,  0.099],
    [ 0.000,  0.000,  0.000,  1.000],
])


# ============================================================
# 실제 find_grasp.py에서 방금 얻은 GRASP POSE
# Camera frame
# ============================================================

POSITION_CAMERA = np.array([
    -0.017545758839855808,
     0.03960483521223068,
     0.20900000631809235
])


CLOSING_AXIS_CAMERA = np.array([
    -0.7599213896251932,
     0.5779125335590202,
     0.29755097906999056
])


APPROACH_AXIS_CAMERA = np.array([
    -0.44118687308457255,
    -0.7947431483182275,
     0.41681827121551324
])


THIRD_AXIS_CAMERA = np.array([
    -0.47736111,
    -0.18547353,
    -0.85890974
])


# ============================================================
# Camera grasp rotation matrix
#
# column 0 = closing axis
# column 1 = third axis
# column 2 = approach axis
# ============================================================

R_CAMERA_GRASP = np.column_stack([
    CLOSING_AXIS_CAMERA,
    THIRD_AXIS_CAMERA,
    APPROACH_AXIS_CAMERA
])


# ============================================================
# Position transform
# ============================================================

def transform_point(point_camera):

    p_camera = np.append(point_camera, 1.0)

    p_base = T_BASE_CAMERA @ p_camera

    return p_base[:3]


# ============================================================
# Rotation transform
#
# R_base_grasp
# = R_base_camera @ R_camera_grasp
# ============================================================

R_BASE_CAMERA = T_BASE_CAMERA[:3, :3]

R_BASE_GRASP = (
    R_BASE_CAMERA @ R_CAMERA_GRASP
)


# ============================================================
# Quaternion
# scipy: [x, y, z, w]
# ============================================================

quaternion_xyzw = (
    Rotation.from_matrix(
        R_BASE_GRASP
    ).as_quat()
)


# ============================================================
# Position
# ============================================================

position_base = transform_point(
    POSITION_CAMERA
)


# ============================================================
# Axes in base frame
# ============================================================

closing_axis_base = (
    R_BASE_CAMERA @ CLOSING_AXIS_CAMERA
)

approach_axis_base = (
    R_BASE_CAMERA @ APPROACH_AXIS_CAMERA
)

third_axis_base = (
    R_BASE_CAMERA @ THIRD_AXIS_CAMERA
)


# ============================================================
# 출력
# ============================================================

print("=" * 60)
print("Camera -> base_link GRASP POSE 변환")
print("=" * 60)


print("\n[Camera frame]")

print("position:")
print(POSITION_CAMERA)

print("\nclosing axis:")
print(CLOSING_AXIS_CAMERA)

print("\napproach axis:")
print(APPROACH_AXIS_CAMERA)

print("\nthird axis:")
print(THIRD_AXIS_CAMERA)

print("\nrotation matrix:")
print(R_CAMERA_GRASP)


print("\n[base_link frame]")

print("position:")
print(position_base)

print("\nclosing axis:")
print(closing_axis_base)

print("\napproach axis:")
print(approach_axis_base)

print("\nthird axis:")
print(third_axis_base)

print("\nrotation matrix:")
print(R_BASE_GRASP)

print("\ndet(R):")
print(np.linalg.det(R_BASE_GRASP))

print("\nquaternion [x, y, z, w]:")
print(quaternion_xyzw)

print("\n" + "=" * 60)
