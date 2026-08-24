with open('preprocess.py', 'r') as f:
    content = f.read()

# 1. sponge 분기의 attempts.append에 plane_model: None 추가
old1 = '''                attempts.append({"remaining": remaining, "labels": labels,
                                  "dbscan_eps": float(eps), "ransac_removed_ratio": None, **evaluation})'''
new1 = '''                attempts.append({"remaining": remaining, "labels": labels,
                                  "dbscan_eps": float(eps), "ransac_removed_ratio": None,
                                  "plane_model": None, **evaluation})'''
assert old1 in content, "old1 매칭 실패"
content = content.replace(old1, new1)

# 2. block/can 분기의 attempts.append에 plane_model 추가
old2 = '''                    attempts.append({"remaining": remaining, "labels": labels,
                                      "dbscan_eps": float(eps), "ransac_removed_ratio": float(removed_ratio),
                                      **evaluation})'''
new2 = '''                    attempts.append({"remaining": remaining, "labels": labels,
                                      "dbscan_eps": float(eps), "ransac_removed_ratio": float(removed_ratio),
                                      "plane_model": list(plane_model), **evaluation})'''
assert old2 in content, "old2 매칭 실패"
content = content.replace(old2, new2)

# 3. preprocess() 함수가 plane_model도 같이 반환하도록 수정
old3 = '''    object_pcd = select_object_pcd(best["remaining"], best["labels"], best["selected"]["cluster_id"])
    return object_pcd'''
new3 = '''    object_pcd = select_object_pcd(best["remaining"], best["labels"], best["selected"]["cluster_id"])
    return object_pcd, best.get("plane_model")'''
assert old3 in content, "old3 매칭 실패"
content = content.replace(old3, new3)

with open('preprocess.py', 'w') as f:
    f.write(content)

print("preprocess.py 수정 완료")
