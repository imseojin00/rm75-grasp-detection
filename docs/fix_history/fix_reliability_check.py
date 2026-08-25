with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

# 1. 상단 설정에 MIN_RELIABLE_POINTS 추가
for i, line in enumerate(lines):
    if line.strip() == 'ANTIPODAL_THRESHOLD = -0.8':
        lines.insert(i + 1, 'MIN_RELIABLE_POINTS = 150  # 이보다 적으면 결과 신뢰 불가\n')
        print(f"MIN_RELIABLE_POINTS 추가: {i+2}번째 줄")
        break

with open('find_grasp.py', 'w') as f:
    f.writelines(lines)

# 2. find_grasp() 함수 안에서, PCA 계산 직후 점 개수 체크 추가
with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'print("PCA 계산 실패")' in line:
        # 그 다음 return None 다음 줄 찾기
        for j in range(i, i + 3):
            if 'return None' in lines[j]:
                insert_idx = j + 1
                insert_code = (
                    '\n'
                    '    if len(points) < MIN_RELIABLE_POINTS:\n'
                    '        print(f"경고: 점 개수 부족({len(points)}개 < {MIN_RELIABLE_POINTS}), '
                    '결과 신뢰 불가. 재촬영 필요")\n'
                    '        return None\n'
                )
                lines.insert(insert_idx, insert_code)
                print(f"점 개수 체크 추가: {insert_idx+1}번째 줄")
                break
        break

with open('find_grasp.py', 'w') as f:
    f.writelines(lines)

print("완료")
