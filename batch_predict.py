import os
import csv
import math
import json
import argparse
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class RebarNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()  # 512-d
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512 + 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, img, geom):
        f_img = self.backbone(img)           # (B, 512)
        f = torch.cat([f_img, geom], dim=1)  # (B, 514)
        return self.classifier(f)            # (B, 2)

def angle_of_line(line: List[List[float]]) -> float:
    (x1, y1), (x2, y2) = line
    theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
    theta = abs(theta)
    if theta > 180:
        theta -= 180
    return theta


def compute_mean_angle_diff(angles: List[float]) -> float:
    if len(angles) < 2:
        return 0.0
    diffs = []
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            d = abs(angles[i] - angles[j])
            d = min(d, 180 - d)
            diffs.append(d)
    return float(np.mean(diffs))


def compute_top_bottom_diff(lines: List[List[List[float]]], angles: List[float]) -> float:
    if not lines:
        return 0.0
    y_mids = [(l[0][1] + l[1][1]) / 2.0 for l in lines]
    split_y = float(np.median(y_mids))

    top_angles, bottom_angles = [], []
    for y_mid, angle in zip(y_mids, angles):
        if y_mid < split_y:
            top_angles.append(angle)
        else:
            bottom_angles.append(angle)

    if not top_angles or not bottom_angles:
        return 0.0

    d = abs(float(np.mean(top_angles)) - float(np.mean(bottom_angles)))
    return float(min(d, 180 - d))


# ============================ Line generation from nodes (PCA + neighbor) ============================
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def get_pc_direction(pc: np.ndarray) -> str:
    x, y = pc
    if abs(x) >= abs(y):
        return "right" if x >= 0 else "left"
    else:
        return "down" if y >= 0 else "up"


def horizontal_or_vertical(direction: str) -> str:
    return "horizontal" if direction in ("left", "right") else "vertical"


def vector_aligned_with_pc(p: np.ndarray, q: np.ndarray, pc: np.ndarray, tolerance_deg: float) -> bool:
    v = q - p
    if np.linalg.norm(v) < 1e-6:
        return False
    v_u = _unit(v)
    pc_u = _unit(pc)
    cosv = float(np.clip(np.dot(v_u, pc_u), -1.0, 1.0))
    ang = math.degrees(math.acos(abs(cosv)))  # allow +/- pc direction
    return ang <= tolerance_deg


def mode_angle_outlier_filter(lines: List[List[List[float]]], threshold_deg: float = 10.0) -> List[List[List[float]]]:
    """Keep lines near dominant angle mode (simple robust pruning)."""
    if not lines or threshold_deg <= 0:
        return lines

    radians = []
    for l in lines:
        p1 = np.array(l[0], dtype=np.float32)
        p2 = np.array(l[1], dtype=np.float32)
        v = p2 - p1
        radians.append(float(np.arctan2(v[1], v[0])))

    bins = 36
    hist, bin_edges = np.histogram(radians, bins=bins, range=(-np.pi, np.pi))
    max_bin = int(np.argmax(hist))
    mode = float(bin_edges[max_bin] + (np.pi / bins))

    thr = threshold_deg * np.pi / 180.0

    def norm_radian(r: float) -> float:
        while r <= -np.pi:
            r += 2 * np.pi
        while r > np.pi:
            r -= 2 * np.pi
        return r

    keep = []
    for l, r in zip(lines, radians):
        if abs(norm_radian(r - mode)) <= thr:
            keep.append(l)
    return keep


def dedup_lines(lines: List[List[List[float]]], round_ndigits: int = 2) -> List[List[List[float]]]:
    seen = set()
    out = []
    for l in lines:
        a = (round(l[0][0], round_ndigits), round(l[0][1], round_ndigits))
        b = (round(l[1][0], round_ndigits), round(l[1][1], round_ndigits))
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        out.append([[float(l[0][0]), float(l[0][1])], [float(l[1][0]), float(l[1][1])]])
    return out


def get_bounding_boxes_yolo(model: Any, image_path: str, conf: float = 0.25) -> List[List[float]]:
    """Return boxes as [x1,y1,x2,y2]."""
    res = model.predict(source=image_path, conf=conf, verbose=False)
    if not res:
        return []
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    return xyxy.astype(np.float32).tolist()


def get_vertice_from_box(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def generate_lines_from_nodes(
    image_path: str,
    yolo_model: Any,
    conf: float = 0.25,
    tolerance_deg: float = 30.0,
    prune_angle_deg: float = 10.0,
) -> Dict[str, Any]:
    boxes = get_bounding_boxes_yolo(yolo_model, image_path, conf=conf)
    if len(boxes) < 2:
        return {"shapes": [], "num_nodes": len(boxes)}

    vertices = np.array([get_vertice_from_box(b) for b in boxes], dtype=np.float32)
    mean = np.mean(vertices, axis=0)
    vc = vertices - mean

    _, _, Vh = np.linalg.svd(vc, full_matrices=False)
    pc1 = Vh[0]
    pc2 = Vh[1] if Vh.shape[0] > 1 else np.array([0.0, 1.0], dtype=np.float32)

    dir1 = get_pc_direction(pc1)
    dir2 = get_pc_direction(pc2)

    pc1_lines, pc2_lines = [], []
    n = len(vertices)

    for i in range(n):
        best1_j, best1_d = None, float("inf")
        best2_j, best2_d = None, float("inf")
        for j in range(n):
            if i == j:
                continue
            p = vertices[i]
            q = vertices[j]
            d = float(np.linalg.norm(q - p))
            if d < 1e-6:
                continue

            if vector_aligned_with_pc(p, q, pc1, tolerance_deg) and d < best1_d:
                best1_j, best1_d = j, d
            if vector_aligned_with_pc(p, q, pc2, tolerance_deg) and d < best2_d:
                best2_j, best2_d = j, d

        if best1_j is not None:
            pc1_lines.append([vertices[i].tolist(), vertices[best1_j].tolist()])
        if best2_j is not None:
            pc2_lines.append([vertices[i].tolist(), vertices[best2_j].tolist()])

    pc1_lines = mode_angle_outlier_filter(dedup_lines(pc1_lines), threshold_deg=prune_angle_deg)
    pc2_lines = mode_angle_outlier_filter(dedup_lines(pc2_lines), threshold_deg=prune_angle_deg)

    shapes = []
    if pc1_lines:
        shapes.append({"lines": pc1_lines, "orientation": horizontal_or_vertical(dir1), "shape_type": "group of lines"})
    if pc2_lines:
        shapes.append({"lines": pc2_lines, "orientation": horizontal_or_vertical(dir2), "shape_type": "group of lines"})

    return {"shapes": shapes, "num_nodes": len(boxes)}


def flatten_lines(line_json: Dict[str, Any]) -> List[List[List[float]]]:
    lines = []
    for s in line_json.get("shapes", []):
        lines.extend(s.get("lines", []))
    return lines


# ============================ Classification inference ============================
def preprocess_rgb(img_path: str, img_size: int = 224) -> torch.Tensor:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # CHW


def safe_torch_load_state_dict(path: str, device: str):
    """
    For your Rebar_2d3d.pt (state_dict), torch.load should return a dict of tensors.
    If user accidentally points to YOLO best.pt, it will error -> show clear message.
    """
    try:
        return torch.load(path, map_location=device)  # weights_only default ok for state_dict
    except Exception as e:
        raise RuntimeError(
            f"Failed to load classifier weights: {path}\n"
            f"請確認這是『Rebar_2d3d.pt（分類模型 state_dict）』，不要拿 YOLO 的 best.pt。\n"
            f"Original error: {e}"
        )


def main():
    ap = argparse.ArgumentParser()

    # ✅ default = same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--folder", default=script_dir, help="資料夾（預設=程式同一層）")

    ap.add_argument("--node_weights", default="best.pt", help="節點偵測 YOLO 權重（預設 best.pt）")
    ap.add_argument("--cls_weights", default="Rebar_2d3d.pt", help="2D/3D 分類權重（state_dict）")
    ap.add_argument("--out_csv", default="pred_2d3d.csv", help="輸出 CSV 檔名")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf threshold")
    ap.add_argument("--tol_deg", type=float, default=30.0, help="PCA neighbor tolerance degrees")
    ap.add_argument("--prune_deg", type=float, default=10.0, help="line angle prune degrees (mode filter)")
    ap.add_argument("--save_lines_json", action="store_true", help="是否輸出每張圖的 line json 到 outputs/lines")
    args = ap.parse_args()

    if YOLO is None:
        raise RuntimeError("找不到 ultralytics，請先 pip install ultralytics")

    folder = args.folder
    print("[INFO] Using folder:", folder)

    node_w = args.node_weights if os.path.isabs(args.node_weights) else os.path.join(folder, args.node_weights)
    cls_w  = args.cls_weights  if os.path.isabs(args.cls_weights)  else os.path.join(folder, args.cls_weights)
    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(folder, args.out_csv)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not os.path.exists(node_w):
        raise FileNotFoundError(f"node_weights not found: {node_w}")
    if not os.path.exists(cls_w):
        raise FileNotFoundError(f"cls_weights not found: {cls_w}")

    # load models
    node_model = YOLO(node_w)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_model = RebarNet().to(device)
    state = safe_torch_load_state_dict(cls_w, device)
    cls_model.load_state_dict(state)
    cls_model.eval()

    # output dir for json
    lines_dir = os.path.join(folder, "outputs", "lines")
    if args.save_lines_json:
        os.makedirs(lines_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    if not images:
        print("[WARN] 找不到圖片（jpg/png/bmp）")
        return

    rows = []
    for fname in images:
        img_path = os.path.join(folder, fname)

        # 1) nodes -> lines json
        line_json = generate_lines_from_nodes(
            img_path, node_model,
            conf=args.conf,
            tolerance_deg=args.tol_deg,
            prune_angle_deg=args.prune_deg,
        )
        lines = flatten_lines(line_json)

        # 2) geom features
        if lines:
            angles = [angle_of_line(l) for l in lines]
            mean_diff = compute_mean_angle_diff(angles)
            top_bottom = compute_top_bottom_diff(lines, angles)
        else:
            mean_diff = 0.0
            top_bottom = 0.0

        # normalize (跟你訓練時一致：/90)
        mean_n = mean_diff / 90.0
        top_n  = top_bottom / 90.0

        # 3) classify 2D/3D
        img_t = preprocess_rgb(img_path, args.img_size).unsqueeze(0).to(device)
        geom_t = torch.tensor([[mean_n, top_n]], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = cls_model(img_t, geom_t)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(prob))

        pred_label = "2D" if pred == 0 else "3D"

        # save json (optional)
        if args.save_lines_json:
            json_name = os.path.splitext(fname)[0] + ".json"
            with open(os.path.join(lines_dir, json_name), "w", encoding="utf-8") as f:
                json.dump({"shapes": line_json.get("shapes", [])}, f, ensure_ascii=False, indent=2)

        rows.append([
            fname,
            pred_label,
            f"{prob[0]:.5f}",  # P(2D)
            f"{prob[1]:.5f}",  # P(3D)
            f"{mean_n:.5f}",
            f"{top_n:.5f}",
            str(line_json.get("num_nodes", 0)),
            str(len(lines)),
        ])

        print(f"[OK] {fname} -> {pred_label} | P3D={prob[1]:.3f} | mean={mean_n:.5f} topbot={top_n:.5f} | nodes={line_json.get('num_nodes',0)} lines={len(lines)}")

    # write csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "pred", "p_2d", "p_3d", "mean_diff_norm", "top_bottom_norm", "num_nodes", "num_lines"])
        w.writerows(rows)

    print(f"\n[DONE] {len(rows)} images -> {out_csv}")
    if args.save_lines_json:
        print(f"[DONE] lines json saved -> {lines_dir}")


if __name__ == "__main__":
    main()