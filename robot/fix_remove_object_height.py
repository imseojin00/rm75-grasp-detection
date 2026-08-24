with open('find_grasp.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    if line.strip() == '"object_height":':
        skip_next = True
        continue
    new_lines.append(line)

with open('find_grasp.py', 'w') as f:
    f.writelines(new_lines)

print("object_height 반환 코드 제거 완료")
