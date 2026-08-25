with open('table_z_from_tf.py', 'r') as f:
    lines = f.readlines()

start_idx = 27  # "def get_table_z_in_camera(self):"
end_idx = 50    # "return float(...)" 다음 (exclusive)

new_block = '''    def get_table_z_in_camera(self, object_point_cam=None):
        """
        object_point_cam: 물체의 카메라 좌표(x,y,z), None이면 카메라 바로 아래 사용.
        카메라가 완벽히 수직이 아니면(실측상 약 5도 기울어짐), 물체 위치에 따라
        테이블까지의 거리가 달라지므로, 물체의 실제 x,y 위치에서 계산해야 정확함.
        """
        transform = self.tf_buffer.lookup_transform(
            BASE_FRAME, CAMERA_FRAME, rclpy.time.Time()
        )
        t = transform.transform.translation
        q = transform.transform.rotation

        x, y, z, w = q.x, q.y, q.z, q.w
        R_base_cam = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
        ])
        t_base_cam = np.array([t.x, t.y, t.z])

        R_cam_base = R_base_cam.T
        t_cam_base = -R_cam_base @ t_base_cam

        if object_point_cam is None:
            table_point_base = np.array([t.x, t.y, TABLE_Z_BASE])
        else:
            object_point_base = R_base_cam @ np.asarray(object_point_cam) + t_base_cam
            table_point_base = np.array([object_point_base[0], object_point_base[1], TABLE_Z_BASE])

        table_point_cam = R_cam_base @ table_point_base + t_cam_base

        return float(table_point_cam[2])
'''

new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]

with open('table_z_from_tf.py', 'w') as f:
    f.writelines(new_lines)

print("완료")
