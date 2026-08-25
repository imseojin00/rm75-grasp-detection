with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

# 877번째 줄(인덱스 876) = "        return None" 다음에 삽입
for i, line in enumerate(lines):
    if line.strip() == 'return None' and i > 870 and i < 880:
        insert_code = (
            '\n'
            '    if len(points) < MIN_RELIABLE_POINTS:\n'
            '        print(f"경고: 점 개수 부족({len(points)}개 < {MIN_RELIABLE_POINTS}), 결과 신뢰 불가. 재촬영 필요")\n'
            '        return None\n'
        )
        lines.insert(i + 1, insert_code)
        print(f"삽입 위치: {i+2}번째 줄")
        break

with open('find_grasp.py', 'w') as f:
    f.writelines(lines)

print("완료")
