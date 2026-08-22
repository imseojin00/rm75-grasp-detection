with open('find_grasp.py', 'r') as f:
    content = f.read()

old_sig = '''    grasp_axis,
    gripper_min,
    gripper_max,
    num_candidates=NUM_CONTACT_CANDIDATES
):'''
new_sig = '''    grasp_axis,
    gripper_min,
    gripper_max,
    axis_width=None,
    num_candidates=NUM_CONTACT_CANDIDATES
):'''
assert old_sig in content, "sig 매칭 실패"
content = content.replace(old_sig, new_sig)

old_score = '''            antipodal_score = (
                -normal_dot
            )

            # ------------------------------------------------
            # 최종 score
            # ------------------------------------------------

            score = (
                0.7 * antipodal_score
                + 0.3 * alignment
            )'''

new_score = '''            antipodal_score = (
                -normal_dot
            )

            # ------------------------------------------------
            # width가 물체 전체 폭(axis_width)에 가까울수록 높은 점수
            # (단일 시점 depth 노이즈로 우연히 좁은 지점이
            #  antipodal처럼 보이는 것을 방지, 범열 제안 반영)
            # ------------------------------------------------

            if axis_width is not None and axis_width > 1e-6:
                width_similarity = 1.0 - min(
                    abs(width - axis_width) / axis_width,
                    1.0
                )
            else:
                width_similarity = 0.0

            # ------------------------------------------------
            # 최종 score
            # ------------------------------------------------

            score = (
                0.5 * antipodal_score
                + 0.2 * alignment
                + 0.3 * width_similarity
            )'''

assert old_score in content, "score 매칭 실패"
content = content.replace(old_score, new_score)

old_dict_end = '''                "antipodal_score":
                    float(antipodal_score),

                "score":
                    float(score)
            })'''

new_dict_end = '''                "antipodal_score":
                    float(antipodal_score),

                "width_similarity":
                    float(width_similarity),

                "score":
                    float(score)
            })'''

assert old_dict_end in content, "dict 매칭 실패"
content = content.replace(old_dict_end, new_dict_end)

old_call = '''    candidates = search_antipodal_pairs(
        points,
        normals,
        center,
        axis,
        gripper_min,
        gripper_max
    )'''

new_call = '''    candidates = search_antipodal_pairs(
        points,
        normals,
        center,
        axis,
        gripper_min,
        gripper_max,
        axis_width=width
    )'''

assert old_call in content, "call 매칭 실패"
content = content.replace(old_call, new_call)

with open('find_grasp.py', 'w') as f:
    f.write(content)

print("모든 수정 완료")
