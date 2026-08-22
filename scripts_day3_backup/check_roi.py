import sys
import numpy as np
import json

base_name = sys.argv[1] if len(sys.argv) > 1 else "block_1x1_01_front"

depth = np.load(f"../data/{base_name}.npy")
with open(f"../data/{base_name}_meta.json") as f:
    meta = json.load(f)

depth_scale = meta["depth_scale"]
depth_m = depth.astype(np.float32) * depth_scale

# 0.5m 이내만 잘게 히스토그램
mask = (depth_m > 0) & (depth_m < 0.5)
close = depth_m[mask]

hist, bin_edges = np.histogram(close, bins=30)
for i in range(len(hist)):
    print(f"{bin_edges[i]:.3f} ~ {bin_edges[i+1]:.3f} : {hist[i]}개")
