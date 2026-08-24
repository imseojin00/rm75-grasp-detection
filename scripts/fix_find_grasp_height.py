with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

# object_pcd = preprocess( 를 object_pcd, plane_model = preprocess( 로 교체
for i, line in enumerate(lines):
    if 'object_pcd = preprocess(' in line:
        lines[i] = line.replace('object_pcd = preprocess(', 'object_pcd, plane_model = preprocess(')
        print(f"교체된 줄 번호: {i+1}")
        break

# "return None" 다음에 높이 계산 코드 삽입
insert_code = '''
    object_height = None
    if plane_model is not None:
        pts = np.asarray(object_pcd.points)
        a, b, c, d = plane_model
        denom = (a**2 + b**2 + c**2) ** 0.5
        if denom > 1e-9 and len(pts) > 0:
            distances = np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / denom
            object_height = float(distances.max())
            print(f"[디버그] 자동 계산된 물체 높이: {object_height*100:.2f} cm")
'''

for i, line in enumerate(lines):
    if line.strip() == 'return None' and i > 845 and i < 860:
        lines.insert(i + 1, insert_code)
        print(f"삽입 위치: {i+2}번째 줄")
        break

with open('find_grasp.py', 'w') as f:
    f.writelines(lines)

print("find_grasp.py 수정 완료")
