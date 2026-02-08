import os
import json
import math
import csv
import numpy as np

# ============================================================
# 設定（程式同層有 outputs/lines/）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "lines")
OUTPUT_CSV = os.path.join(BASE_DIR, "geom_features.csv")


# ============================================================
# 幾何工具
# ============================================================
def angle_of_line(line):
    (x1, y1), (x2, y2) = line
    theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
    theta = abs(theta)
    if theta > 180:
        theta -= 180
    return theta


def normalize_and_round(x, base=90.0, digits=5):
    """正規化並取到小數後 digits 位（你原本做法：除以 90）"""
    return round(float(x) / base, digits)


# ============================================================
# JSON 處理
# ============================================================
def collect_lines(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for shape in data.get("shapes", []):
        for line in shape.get("lines", []):
            # line: [[x1,y1],[x2,y2]]
            lines.append(line)
    return lines


# ============================================================
# 幾何特徵計算（角度）
# ============================================================
def compute_mean_angle_diff(angles):
    if len(angles) < 2:
        return 0.0

    diffs = []
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            d = abs(angles[i] - angles[j])
            d = min(d, 180 - d)
            diffs.append(d)

    return float(np.mean(diffs))


def compute_top_bottom_diff(lines, angles):
    y_mids = [(l[0][1] + l[1][1]) / 2 for l in lines]
    split_y = np.median(y_mids)

    top_angles, bottom_angles = [], []
    for y_mid, angle in zip(y_mids, angles):
        if y_mid < split_y:
            top_angles.append(angle)
        else:
            bottom_angles.append(angle)

    if not top_angles or not bottom_angles:
        return 0.0

    d = abs(np.mean(top_angles) - np.mean(bottom_angles))
    return float(min(d, 180 - d))


# ============================================================
# 批次處理
# ============================================================
def process_outputs_folder(outputs_dir, output_csv):
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    rows = []

    for fname in sorted(os.listdir(outputs_dir)):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(outputs_dir, fname)
        base = os.path.splitext(fname)[0]  # 作為 image 欄位

        lines = collect_lines(json_path)

        if not lines:
            mean_diff = 0.0
            top_bottom_diff = 0.0
        else:
            angles = [angle_of_line(line) for line in lines]
            mean_diff = compute_mean_angle_diff(angles)
            top_bottom_diff = compute_top_bottom_diff(lines, angles)

        # 依你原本：除以 90 正規化 + 取小數後 5 位
        mean_diff = normalize_and_round(mean_diff)
        top_bottom_diff = normalize_and_round(top_bottom_diff)

        rows.append([base, mean_diff, top_bottom_diff])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mean_diff", "top_bottom_diff"])
        writer.writerows(rows)

    print(f"[DONE] {len(rows)} json processed → {output_csv}")


# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    process_outputs_folder(
        outputs_dir=OUTPUTS_DIR,
        output_csv=OUTPUT_CSV
    )
