import numpy as np
import json

base_name = "block_1x1_01_top"

depth = np.load(f"../data/{base_name}.npy")
with open(f"../data/{base_name}_meta.json") as f:
    meta = json.load(f)

depth_scale = meta["depth_scale"]
depth_m = depth.astype(np.float32) * depth_scale

# 유효한 depth 값만 (0 제외, 너무 먼 값 제외)
valid_mask = (depth_m > 0.05) & (depth_m < 1.0)
valid_depths = depth_m[valid_mask]

print(f"유효 depth 픽셀 수: {len(valid_depths)}")

# 히스토그램으로 최빈값 찾기 (1mm 단위로 구간 나눔)
bin_size = 0.001  # 1mm
bins = np.arange(valid_depths.min(), valid_depths.max() + bin_size, bin_size)
hist, bin_edges = np.histogram(valid_depths, bins=bins)

max_bin_idx = np.argmax(hist)
table_depth = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2

print(f"\n[테이블로 추정된 depth]")
print(f"  값: {table_depth:.4f} m")
print(f"  해당 구간 픽셀 수: {hist[max_bin_idx]} (전체의 {hist[max_bin_idx]/len(valid_depths)*100:.1f}%)")

# 히스토그램 상위 10개 구간 출력 (참고용)
print(f"\n[상위 10개 depth 구간]")
top10_idx = np.argsort(hist)[::-1][:10]
for idx in sorted(top10_idx):
    center = (bin_edges[idx] + bin_edges[idx+1]) / 2
    print(f"  {center:.4f}m : {hist[idx]}개")

# find_grasp으로 물체 중심 depth 확인
from find_grasp import find_grasp
result = find_grasp(base_name)
if result and result.get("grasp_pose"):
    object_depth = result["grasp_pose"]["position"][2]
    print(f"\n[물체 depth]: {object_depth:.4f} m")

    object_height = table_depth - object_depth
    print(f"\n[방법 E로 계산된 물체 높이]: {object_height*100:.2f} cm")
    print(f"[실제 block_1x1 높이]: 2.5 cm")
else:
    print("\nfind_grasp 결과 없음")
