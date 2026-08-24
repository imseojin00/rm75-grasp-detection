with open('find_grasp.py', 'r') as f:
    content = f.read()

old = '''    object_height = None
    if plane_model is not None:
        pts = np.asarray(object_pcd.points)
        a, b, c, d = plane_model
        if abs(c) > 1e-9 and len(pts) > 0:
            obj_x = float(np.median(pts[:, 0]))
            obj_y = float(np.median(pts[:, 1]))
            top_z = float(np.median(pts[:, 2]))
            table_z = -(a * obj_x + b * obj_y + d) / c
            object_height = table_z - top_z
            if object_height < 0:
                object_height = None
            else:
                print(f"[디버그] 자동 계산된 물체 높이: {object_height*100:.2f} cm")
'''

new = '''    # 높이 계산은 table_z_from_tf.py(TF 기반 고정 테이블 위치)에서
    # 별도로 처리. RANSAC 기반 방식(plane_model 활용)은 물체가 크거나
    # 복잡하면(block_2x2, block_L3, can) 잘못된 평면을 잡는 문제가 있어 폐기.
'''

assert old in content, "매칭 실패"
content = content.replace(old, new)

with open('find_grasp.py', 'w') as f:
    f.write(content)

print("RANSAC 기반 높이 계산 코드 제거 완료")
