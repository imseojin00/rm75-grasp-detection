with open('capture_utils.py', 'r') as f:
    content = f.read()

old = '''def get_frame(pipeline, align):
    """사진 한 장 찍기"""
    frames = pipeline.wait_for_frames()'''

new = '''def get_frame(pipeline, align, flush=3):
    """사진 한 장 찍기. flush만큼 오래된 프레임을 먼저 버리고 최신 프레임을 받음."""
    for _ in range(flush):
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()'''

assert old in content, "매칭 실패"
content = content.replace(old, new)

with open('capture_utils.py', 'w') as f:
    f.write(content)

print("완료")
