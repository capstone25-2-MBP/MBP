import os
import json
import shutil
import pandas as pd

BASE_DIR = os.getcwd()

IMAGES_DIR = os.path.join(BASE_DIR, "images")
JSON_DIR = os.path.join(BASE_DIR, "vqa_json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

PART_SIZE = 2000

# =====================
# CSV 파일 자동 탐색
# =====================
excel_files = [
    f for f in os.listdir(BASE_DIR)
    if f.endswith(".csv")
]

print("엑셀 파일:", excel_files)

# =====================
# json 캐시
# =====================
json_cache = {}

def load_json(part_num):
    if part_num in json_cache:
        return json_cache[part_num]

    part_name = f"GMAI_mm_bench_TEST_part_{part_num}"
    part_path = os.path.join(JSON_DIR, part_name)

    if not os.path.exists(part_path):
        return None

    files = os.listdir(part_path)
    if not files:
        return None

    with open(os.path.join(part_path, files[0]), "r", encoding="utf-8") as f:
        data = json.load(f)

    json_cache[part_num] = data
    return data

# =====================
# 메인 루프
# =====================
total_copied = 0

for excel in excel_files:
    domain = os.path.splitext(excel)[0]
    print(f"\n▶ 도메인 처리 중: {domain}")

    df = pd.read_csv(os.path.join(BASE_DIR, excel))

    assert "index" in df.columns, f"{excel}에 index 열 없음"

    out_dir = os.path.join(OUTPUT_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)

    copied = 0

    for idx in df["index"]:
        if pd.isna(idx):
            continue

        global_idx = int(idx)
        part_num = global_idx // PART_SIZE + 1

        part_json = load_json(part_num)
        if part_json is None:
            continue

        local_idx = global_idx % PART_SIZE
        if local_idx >= len(part_json):
            continue

        json_item = part_json[local_idx]
        image_name = json_item.get("image")
        if not image_name:
            continue

        src = os.path.join(
            IMAGES_DIR,
            f"GMAI_mm_bench_TEST_part_{part_num}",
            image_name
        )

        if not os.path.exists(src):
            continue

        dst = os.path.join(out_dir, f"{global_idx}.jpg")
        shutil.copy(src, dst)
        copied += 1

    print(f"  ✔ {copied}개 이미지 복사됨")
    total_copied += copied

print(f"\n✅ 전체 완료: {total_copied}개 이미지")
