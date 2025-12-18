import os
import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

BASE_DIR = os.getcwd()

CSV_PATH = "Pulmonary_Medicine.csv"
VQA_JSON_DIR = "vqa_json"
IMAGES_DIR = "images"
OUTPUT_XLSX = "Pulmonary_Medicine.xlsx"

# ---------------------------
# 1. CSV 로드
# ---------------------------
df = pd.read_csv(CSV_PATH)
print(f"CSV 행 수: {len(df)}")

# ---------------------------
# 2. 엑셀 생성
# ---------------------------
wb = Workbook()
ws = wb.active
ws.title = "result"

headers = list(df.columns) + ["image"]
ws.append(headers)

# ---------------------------
# 3. part별 json 캐시
# ---------------------------
json_cache = {}

def load_part_json(part_name):
    if part_name in json_cache:
        return json_cache[part_name]

    json_path = os.path.join(
        VQA_JSON_DIR,
        part_name,
        "vqa_shard_000.json"
    )

    if not os.path.exists(json_path):
        print(f"[경고] json 없음: {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # ⭐ 리스트

    json_cache[part_name] = data
    return data

# ---------------------------
# 4. CSV 순회
# ---------------------------
row_idx = 2

for _, row in df.iterrows():
    raw = str(row["index"]).strip()

    if raw.lower() == "nan":
        continue

    try:
        global_idx = int(float(raw))
    except:
        continue

    # part / local index 계산
    part = global_idx // 2000 + 1
    local_idx = global_idx % 2000

    part_name = f"GMAI_mm_bench_TEST_part_{part}"

    part_json = load_part_json(part_name)
    if part_json is None:
        continue

    if local_idx >= len(part_json):
        print(f"[범위 초과] {part_name} index {local_idx}")
        continue

    item = part_json[local_idx]
    image_name = item.get("image")

    # CSV 내용 기록
    ws.append(list(row.values) + [""])

    # 이미지 삽입
    if image_name:
        img_path = os.path.join(
            IMAGES_DIR,
            part_name,
            image_name
        )

        if os.path.exists(img_path):
            img = XLImage(img_path)
            img.width = 150
            img.height = 150
            ws.add_image(img, f"{chr(65+len(headers)-1)}{row_idx}")
        else:
            print(f"[이미지 없음] {img_path}")

    row_idx += 1

# ---------------------------
# 5. 저장
# ---------------------------
wb.save(OUTPUT_XLSX)
print(f"\n완료: {OUTPUT_XLSX}")
