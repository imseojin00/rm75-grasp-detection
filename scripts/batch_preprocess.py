from pathlib import Path
import subprocess
import csv
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "batch_logs"

PREPROCESS = ROOT / "scripts" / "preprocess.py"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# *_meta.json 하나를 촬영 데이터 1세트로 사용
base_names = sorted(
    p.name.replace("_meta.json", "")
    for p in DATA_DIR.glob("*_meta.json")
)

print(f"전체 데이터 수: {len(base_names)}")

results = []

for i, base_name in enumerate(base_names, start=1):

    print()
    print("=" * 60)
    print(f"[{i}/{len(base_names)}] {base_name}")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(PREPROCESS),
        base_name,
        "--no-vis",
    ]

    # 수세미: RANSAC 생략
    if base_name.startswith("sponge_"):
        cmd += [
            "--dbscan-eps", "0.005",
            "--skip-ransac",
        ]

    # 캔
    elif base_name.startswith("can_"):
        cmd += [
            "--dbscan-eps", "0.005",
        ]

    # 블록: preprocess.py 기본값 사용

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    log_text = result.stdout + "\n" + result.stderr

    log_path = LOG_DIR / f"{base_name}.log"
    log_path.write_text(log_text, encoding="utf-8")

    # 최종 물체 point 수 추출
    point_match = re.search(
        r"물체 point 수\s*:\s*(\d+)",
        log_text
    )

    object_points = (
        int(point_match.group(1))
        if point_match
        else ""
    )

    # 선택 cluster 번호
    cluster_match = re.search(
        r"선택된 물체 cluster\s*:\s*(\d+)",
        log_text
    )

    selected_cluster = (
        int(cluster_match.group(1))
        if cluster_match
        else ""
    )

    if result.returncode == 0:
        status = "RUN_OK"
    else:
        status = "RUN_FAIL"

    print(f"상태: {status}")
    print(f"물체 points: {object_points}")

    results.append({
        "base_name": base_name,
        "status": status,
        "selected_cluster": selected_cluster,
        "object_points": object_points,
    })


csv_path = OUTPUT_DIR / "batch_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "base_name",
            "status",
            "selected_cluster",
            "object_points",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


ok_count = sum(r["status"] == "RUN_OK" for r in results)
fail_count = sum(r["status"] == "RUN_FAIL" for r in results)

print()
print("=" * 60)
print("전체 자동 전처리 완료")
print(f"전체 : {len(results)}")
print(f"RUN_OK : {ok_count}")
print(f"RUN_FAIL : {fail_count}")
print(f"결과 파일 : {csv_path}")
print("=" * 60)
