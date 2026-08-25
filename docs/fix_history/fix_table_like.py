with open('preprocess.py', 'r') as f:
    content = f.read()

old = '    table_like = largest > 0.14 and smallest < 0.010 and flatness < 0.08'
new = '    table_like = smallest < 0.010 and flatness < 0.08'

assert old in content, "매칭 실패"
content = content.replace(old, new)

with open('preprocess.py', 'w') as f:
    f.write(content)

print("완료")
