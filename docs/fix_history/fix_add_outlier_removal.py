with open('preprocess.py', 'r') as f:
    content = f.read()

old = '''def downsample_pointcloud(pcd):
    if len(pcd.points) == 0:
        raise RuntimeError("다운샘플링할 point가 없습니다.")
    downsampled = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if len(downsampled.points) == 0:
        raise RuntimeError("Voxel downsample 후 point가 없습니다.")
    return downsampled'''

new = '''def downsample_pointcloud(pcd):
    if len(pcd.points) == 0:
        raise RuntimeError("다운샘플링할 point가 없습니다.")
    downsampled = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if len(downsampled.points) == 0:
        raise RuntimeError("Voxel downsample 후 point가 없습니다.")

    # 통계적 이상치 제거 (depth 노이즈로 튀는 점들을 미리 걸러냄)
    # 세운 원통형 물체 등에서 noise 비율이 높아 DBSCAN이 잘못된
    # 큰 덩어리를 선택하는 문제를 완화하기 위해 추가.
    if len(downsampled.points) >= 20:
        cleaned, _ = downsampled.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.5
        )
        if len(cleaned.points) > 0:
            downsampled = cleaned

    return downsampled'''

assert old in content, "매칭 실패"
content = content.replace(old, new)

with open('preprocess.py', 'w') as f:
    f.write(content)

print("완료")
