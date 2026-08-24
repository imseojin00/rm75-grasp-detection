with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

old_block = '''    object_height = None
    if plane_model is not None:
        pts = np.asarray(object_pcd.points)
        a, b, c, d = plane_model
        denom = (a**2 + b**2 + c**2) ** 0.5
        if denom > 1e-9 and len(pts) > 0:
            distances = np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / denom
            object_height = float(distances.max())
            print(f"[디버그] 자동 계산된 물체 높이: {object_height*100:.2f} cm")
'''

new_block = '''    object_height = None
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

content = ''.join(lines)
assert old_block in content, "old_block 매칭 실패"
content = content.replace(old_block, new_block)

with open('find_grasp.py', 'w') as f:
    f.write(content)

print("높이 계산 공식 수정 완료")
