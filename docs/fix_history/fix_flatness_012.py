with open('preprocess.py', 'r') as f:
    content = f.read()

old = '    table_like = smallest < 0.015 and flatness < 0.08'
new = '    table_like = smallest < 0.015 and flatness < 0.12'

assert old in content, "매칭 실패"
content = content.replace(old, new)

with open('preprocess.py', 'w') as f:
    f.write(content)

print("완료")
