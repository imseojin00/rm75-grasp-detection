import sys
import numpy as np
import cv2

base_name = sys.argv[1] if len(sys.argv) > 1 else "block_1x1_01_front"

color = np.load(f"../data/{base_name}_color.npy")

u_min, u_max, v_min, v_max = 200, 330, 300, 420

display = color.copy()
cv2.rectangle(display, (u_min, v_min), (u_max, v_max), (0, 255, 0), 2)

cv2.imshow("roi check", display)
print("press any key to close")
cv2.waitKey(0)
cv2.destroyAllWindows()
