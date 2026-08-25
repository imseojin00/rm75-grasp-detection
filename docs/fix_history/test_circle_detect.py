from preprocess import preprocess
import numpy as np
import cv2

object_pcd, plane_model = preprocess('test_00_live1')
points = np.asarray(object_pcd.points)
print(f'원래 점 개수: {len(points)}')
print(f'원래 x범위: {points[:,0].min():.4f} ~ {points[:,0].max():.4f}')
print(f'원래 y범위: {points[:,1].min():.4f} ~ {points[:,1].max():.4f}')

# 2D 이미지로 투영
xy = points[:, :2] * 1000  # m -> mm
xy_min = xy.min(axis=0)
xy_shifted = xy - xy_min
img_size = int(xy_shifted.max()) + 20
mask = np.zeros((img_size, img_size), dtype=np.uint8)
for x, y in xy_shifted:
    cv2.circle(mask, (int(x)+10, int(y)+10), 2, 255, -1)

# minEnclosingCircle로 전체를 감싸는 원 확인 (참고용)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
(cx, cy), radius = cv2.minEnclosingCircle(largest)
print(f'전체 감싸는 원: 중심=({cx:.1f},{cy:.1f}), 반지름={radius:.1f}mm')

# HoughCircles로 실제 원형 물체 찾기 시도
mask_blur = cv2.GaussianBlur(mask, (5,5), 0)
circles = cv2.HoughCircles(
    mask_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
    param1=50, param2=15, minRadius=10, maxRadius=int(img_size/2)
)
if circles is not None:
    print(f'HoughCircles 찾은 원 개수: {len(circles[0])}')
    for c in circles[0]:
        print(f'  중심=({c[0]:.1f},{c[1]:.1f}), 반지름={c[2]:.1f}mm')
else:
    print('HoughCircles: 원을 못 찾음')
