import sys
import numpy as np
import matplotlib.pyplot as plt

from find_grasp import find_grasp


# ============================================================
# 설정
# ============================================================

base_name = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "block_1x2_01_short_axis_front"
)


# ============================================================
# Grasp 계산
# ============================================================

result = find_grasp(base_name)

if result is None:
    print("grasp 계산 실패")
    sys.exit(1)


grasp_pose = result["grasp_pose"]


if grasp_pose is None:
    print("선택된 grasp가 없습니다.")
    sys.exit(1)


# ============================================================
# 점군
# ============================================================

points = np.asarray(
    result["points"],
    dtype=float
)


# ============================================================
# Grasp 정보
# ============================================================

contact_a = np.asarray(
    grasp_pose["contact_a"],
    dtype=float
)

contact_b = np.asarray(
    grasp_pose["contact_b"],
    dtype=float
)

position = np.asarray(
    grasp_pose["position"],
    dtype=float
)

closing_axis = np.asarray(
    grasp_pose["closing_axis"],
    dtype=float
)

approach_axis = np.asarray(
    grasp_pose["approach_axis"],
    dtype=float
)

third_axis = np.asarray(
    grasp_pose["third_axis"],
    dtype=float
)


# ============================================================
# 3D Figure
# ============================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# 1. 점군
# ============================================================

ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    s=4,
    alpha=0.35,
    label="Point cloud"
)


# ============================================================
# 2. Contact A / B
# ============================================================

ax.scatter(
    contact_a[0],
    contact_a[1],
    contact_a[2],
    s=100,
    marker="o",
    label="Contact A"
)

ax.scatter(
    contact_b[0],
    contact_b[1],
    contact_b[2],
    s=100,
    marker="o",
    label="Contact B"
)


# ============================================================
# 3. Grasp center
# ============================================================

ax.scatter(
    position[0],
    position[1],
    position[2],
    s=150,
    marker="*",
    label="Grasp center"
)


# ============================================================
# 4. Contact A ↔ B
# ============================================================

ax.plot(
    [contact_a[0], contact_b[0]],
    [contact_a[1], contact_b[1]],
    [contact_a[2], contact_b[2]],
    linewidth=3,
    label="Closing direction"
)


# ============================================================
# 5. Grasp coordinate axes
# ============================================================

axis_length = 0.03


# Closing axis
ax.quiver(
    position[0],
    position[1],
    position[2],
    closing_axis[0],
    closing_axis[1],
    closing_axis[2],
    length=axis_length,
    normalize=True,
    linewidth=3,
    label="Closing axis"
)


# Approach axis
ax.quiver(
    position[0],
    position[1],
    position[2],
    approach_axis[0],
    approach_axis[1],
    approach_axis[2],
    length=axis_length,
    normalize=True,
    linewidth=3,
    label="Approach axis"
)


# Third axis
ax.quiver(
    position[0],
    position[1],
    position[2],
    third_axis[0],
    third_axis[1],
    third_axis[2],
    length=axis_length,
    normalize=True,
    linewidth=3,
    label="Third axis"
)


# ============================================================
# 축 설정
# ============================================================

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    f"Grasp Visualization\n"
    f"width = {grasp_pose['width'] * 100:.2f} cm, "
    f"score = {result['best_grasp']['best']['score']:.4f}"
)


ax.legend()


# ============================================================
# 화면 비율
# ============================================================

try:
    ax.set_box_aspect(
        [
            np.ptp(points[:, 0]),
            np.ptp(points[:, 1]),
            np.ptp(points[:, 2])
        ]
    )
except Exception:
    pass


# ============================================================
# 저장
# ============================================================

output_file = (
    f"{base_name}_grasp.png"
)

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

print("\n================================")
print("시각화 저장 완료")
print("================================")
print(
    f"파일: {output_file}"
)

plt.show()
