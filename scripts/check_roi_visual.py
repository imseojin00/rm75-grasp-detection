import sys
import numpy as np
import cv2

base_name = sys.argv[1] if len(sys.argv) > 1 else "block_1x1_01_front"

color = np.load(f"../data/{base_name}_color.npy")
display = color.copy()

WINDOW_NAME = "click corners"

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"click coord: u={x}, v={y}")

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

print("Click top-left corner of block, then bottom-right corner.")
print("Press any key to quit.")

cv2.imshow(WINDOW_NAME, display)
cv2.waitKey(0)
cv2.destroyAllWindows()
