with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip() == 'points.tolist(),' and i > 1070 and i < 1090:
        insert_code = '        "object_height":\n            object_height,\n'
        lines.insert(i + 1, insert_code)
        print(f"삽입 위치: {i+2}번째 줄")
        break

with open('find_grasp.py', 'w') as f:
    f.writelines(lines)

print("object_height 반환값 추가 완료")
