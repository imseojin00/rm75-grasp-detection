with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

insert_idx = 881  # index 880("return None") 다음
insert_code = (
    '\n'
    '    if len(points) < MIN_RELIABLE_POINTS:\n'
    '        print(f"경고: 점 개수 부족({len(points)}개 < {MIN_RELIABLE_POINTS}), 결과 신뢰 불가. 재촬영 필요")\n'
    '        return None\n'
)

lines.insert(insert_idx, insert_code)

with open('find_grasp.py', 'w') as f:
    f.writelines(lines)

print("삽입 완료")
