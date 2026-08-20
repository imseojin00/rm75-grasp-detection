import sys
import numpy as np
import cv2

if len(sys.argv) < 2:
    print("사용법: python3 check_capture.py <파일이름_prefix>")
    print("예: python3 check_capture.py block_1x1_01_front")
    sys.exit(1)

base_name = sys.argv[1]

color = np.load(f"../data/{base_name}_color.npy")
depth = np.load(f"../data/{base_name}.npy")

print(f"color shape: {color.shape}, dtype: {color.dtype}")
print(f"depth shape: {depth.shape}, dtype: {depth.dtype}, min: {depth.min()}, max: {depth.max()}")

cv2.imshow("Color", color)

# depth는 눈으로 보기 편하게 정규화
depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow("Depth", depth_vis)

print("아무 키나 누르면 창 닫힘")
cv2.waitKey(0)
cv2.destroyAllWindows()
