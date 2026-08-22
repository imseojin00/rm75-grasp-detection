with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

# compute_pca_axes 함수 교체 (25번째 줄 근처, def부터 다음 구분선 전까지)
new_content = []
i = 0
while i < len(lines):
    if lines[i].strip() == 'def compute_pca_axes(pcd):':
        # 새 함수로 교체
        new_content.append('def compute_pca_axes(pcd):\n')
        new_content.append('    """\n')
        new_content.append('    수직 파지 전용: z(카메라 깊이) 무시, x-y 평면에서만 PCA.\n')
        new_content.append('    범열 2026-08-21 제안 반영.\n')
        new_content.append('    """\n')
        new_content.append('    points = np.asarray(pcd.points)\n')
        new_content.append('    if len(points) < 10:\n')
        new_content.append('        return None, None, None\n')
        new_content.append('    center = points.mean(axis=0)\n')
        new_content.append('    xy = points[:, :2] - center[:2]\n')
        new_content.append('    cov_xy = np.cov(xy.T)\n')
        new_content.append('    eigvals_xy, eigvecs_xy = np.linalg.eigh(cov_xy)\n')
        new_content.append('    print("\\n[2D PCA 고유값 (x-y 평면)]")\n')
        new_content.append('    for k in range(2):\n')
        new_content.append('        print(f"  축 {k}: {eigvals_xy[k]:.8f}")\n')
        new_content.append('    short_axis = np.array([eigvecs_xy[0, 0], eigvecs_xy[1, 0], 0.0])\n')
        new_content.append('    long_axis = np.array([eigvecs_xy[0, 1], eigvecs_xy[1, 1], 0.0])\n')
        new_content.append('    approach_axis = np.array([0.0, 0.0, 1.0])\n')
        new_content.append('    eigvecs = np.column_stack([short_axis, long_axis, approach_axis])\n')
        new_content.append('    print("\\n[PCA 축 방향]")\n')
        new_content.append('    print(f"  축 0 (짧은축): {eigvecs[:, 0]}")\n')
        new_content.append('    print(f"  축 1 (긴축): {eigvecs[:, 1]}")\n')
        new_content.append('    print(f"  축 2 (approach): {eigvecs[:, 2]}")\n')
        new_content.append('    return center, eigvecs, points\n')
        # 원본 함수의 끝(다음 '# ====' 구분선)까지 건너뛰기
        i += 1
        while i < len(lines) and not lines[i].startswith('# ====='):
            i += 1
        continue
    elif lines[i].strip() == 'for i in range(3):' and i > 0 and 'evaluate_axis' in ''.join(lines[i:i+8]):
        new_content.append('    for i in range(2):  # approach축(2) 제외\n')
        i += 1
        continue
    else:
        new_content.append(lines[i])
        i += 1

with open('find_grasp.py', 'w') as f:
    f.writelines(new_content)

print("완료")
