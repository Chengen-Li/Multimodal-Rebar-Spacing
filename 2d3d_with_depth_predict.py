import os
import csv
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

# 2D/3D Classification Model (RGB + 10 Geometry/Depth features)
class RebarNet(nn.Module):
    def __init__(self, feat_dim: int = 10):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512 + feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, img, feat):
        f_img = self.backbone(img)
        f = torch.cat([f_img, feat], dim=1)
        return self.classifier(f)

# Calculate the angle of a single line segment
def angle_of_line(line: List[List[float]]) -> float:
    (x1, y1), (x2, y2) = line
    theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
    theta = abs(theta)
    if theta > 180: theta -= 180
    return theta

# Calculate mean angular difference between all line pairs
def compute_mean_angle_diff(angles: List[float]) -> float:
    if len(angles) < 2: return 0.0
    diffs = []
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            d = abs(angles[i] - angles[j])
            d = min(d, 180 - d)
            diffs.append(d)
    return float(np.mean(diffs))

# Calculate angular difference between top and bottom halves of the image
def compute_top_bottom_diff(lines: List[List[List[float]]], angles: List[float]) -> float:
    if not lines: return 0.0
    y_mids = [(l[0][1] + l[1][1]) / 2.0 for l in lines]
    split_y = float(np.median(y_mids))
    top, bot = [], []
    for y_mid, angle in zip(y_mids, angles):
        if y_mid < split_y: top.append(angle)
        else: bot.append(angle)
    if not top or not bot: return 0.0
    d = abs(float(np.mean(top)) - float(np.mean(bot)))
    return float(min(d, 180 - d))

# Normalize vector to unit length
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

# Check if vector is aligned with principal component within tolerance
def vector_aligned_with_pc(p: np.ndarray, q: np.ndarray, pc: np.ndarray, tolerance_deg: float) -> bool:
    v = q - p
    if np.linalg.norm(v) < 1e-6: return False
    ang = math.degrees(math.acos(abs(float(np.clip(np.dot(_unit(v), _unit(pc)), -1.0, 1.0)))))
    return ang <= tolerance_deg

# Remove duplicate lines
def dedup_lines(lines: List[List[List[float]]], digits: int = 2) -> List[List[List[float]]]:
    seen, out = set(), []
    for l in lines:
        a = (round(l[0][0], digits), round(l[0][1], digits))
        b = (round(l[1][0], digits), round(l[1][1], digits))
        key = tuple(sorted([a, b]))
        if key not in seen:
            seen.add(key)
            out.append(l)
    return out

# Filter lines that deviate from the most common angle (mode)
def mode_angle_outlier_filter(lines: List[List[List[float]]], threshold_deg: float = 10.0) -> List[List[List[float]]]:
    if not lines or threshold_deg <= 0: return lines
    radians = [float(np.arctan2(np.array(l[1])[1]-np.array(l[0])[1], np.array(l[1])[0]-np.array(l[0])[0])) for l in lines]
    hist, bin_edges = np.histogram(radians, bins=36, range=(-np.pi, np.pi))
    mode = float(bin_edges[np.argmax(hist)] + (np.pi / 36))
    thr = threshold_deg * np.pi / 180.0
    keep = []
    for l, r in zip(lines, radians):
        diff = r - mode
        while diff <= -np.pi: diff += 2 * np.pi
        while diff > np.pi: diff -= 2 * np.pi
        if abs(diff) <= thr: keep.append(l)
    return keep

# Use YOLO to detect rebar intersection nodes
def get_bounding_boxes_yolo(model: Any, image_path: str, conf: float = 0.25) -> List[List[float]]:
    res = model.predict(source=image_path, conf=conf, verbose=False)
    if not res or res[0].boxes is None: return []
    return res[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32).tolist()

# Connect nodes into lines based on PCA directions
def generate_lines_from_nodes(image_path: str, yolo_model: Any, conf: float = 0.25, tol_deg: float = 30.0, prune_deg: float = 10.0) -> Dict[str, Any]:
    boxes = get_bounding_boxes_yolo(yolo_model, image_path, conf)
    if len(boxes) < 2: return {"shapes": [], "num_nodes": len(boxes)}
    vertices = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes], dtype=np.float32)
    _, _, Vh = np.linalg.svd(vertices - np.mean(vertices, axis=0), full_matrices=False)
    pc1, pc2 = Vh[0], Vh[1] if Vh.shape[0] > 1 else np.array([0.0, 1.0], dtype=np.float32)
    pc1_l, pc2_l = [], []
    for i in range(len(vertices)):
        b1_j, b1_d = None, float("inf"); b2_j, b2_d = None, float("inf")
        for j in range(len(vertices)):
            if i == j: continue
            d = float(np.linalg.norm(vertices[j] - vertices[i]))
            if d < 1e-6: continue
            if vector_aligned_with_pc(vertices[i], vertices[j], pc1, tol_deg) and d < b1_d: b1_j, b1_d = j, d
            if vector_aligned_with_pc(vertices[i], vertices[j], pc2, tol_deg) and d < b2_d: b2_j, b2_d = j, d
        if b1_j is not None: pc1_l.append([vertices[i].tolist(), vertices[b1_j].tolist()])
        if b2_j is not None: pc2_l.append([vertices[i].tolist(), vertices[b2_j].tolist()])
    pc1_l = mode_angle_outlier_filter(dedup_lines(pc1_l), prune_deg)
    pc2_l = mode_angle_outlier_filter(dedup_lines(pc2_l), prune_deg)
    shapes = []
    if pc1_l: shapes.append({"lines": pc1_l})
    if pc2_l: shapes.append({"lines": pc2_l})
    return {"shapes": shapes, "num_nodes": len(boxes)}

# Combine all lines into a single list
def flatten_lines(line_json: Dict[str, Any]) -> List[List[List[float]]]:
    lines = []
    for s in line_json.get("shapes", []): lines.extend(s.get("lines", []))
    return lines

# Find depth map file matching the RGB image name
def find_matching_depth(depth_dir: str, base: str) -> Optional[str]:
    for ext in [".jpg", ".jpeg", ".png", ".npy"]:
        p = os.path.join(depth_dir, base + ext)
        if os.path.exists(p): return p
    return None

# Read depth data from image or npy file
def read_depth_any(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"): return np.load(path).astype(np.float32)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None: raise FileNotFoundError(f"Cannot read depth: {path}")
    return cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32) if d.ndim == 3 else d.astype(np.float32)

# Extract global features from depth map
def depth_global_features(depth: np.ndarray) -> Tuple[float, float, float, float, float]:
    valid = depth > 0
    vr = float(valid.mean())
    med, std = (float(np.median(depth[valid])), float(np.std(depth[valid]))) if valid.any() else (0.0, 0.0)
    mid = depth.shape[0] // 2
    t_v, b_v = valid[:mid, :], valid[mid:, :]
    diff = abs(float(np.median(depth[:mid,:][t_v])) - float(np.median(depth[mid:,:][b_v]))) if t_v.any() and b_v.any() else 0.0
    g = cv2.magnitude(cv2.Sobel(depth, cv2.CV_32F, 1, 0), cv2.Sobel(depth, cv2.CV_32F, 0, 1))
    grad = float(np.mean(g[valid])) if valid.any() else float(np.mean(g))
    return vr, med, std, diff, grad

# Calculate intersection point of two line segments
def segment_intersection(p1, p2, p3, p4) -> Optional[Tuple[float, float]]:
    x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9: return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
    if min(x1,x2)-1e-6<=px<=max(x1,x2)+1e-6 and min(y1,y2)-1e-6<=py<=max(y1,y2)+1e-6: return float(px), float(py)
    return None

# Extract depth features from Regions of Interest around intersections
def roi_depth_features(depth: np.ndarray, lines: List[List[List[float]]], roi_h: int = 10, ring: int = 6) -> Tuple[float, float, float]:
    if depth is None or depth.size == 0 or not lines: return 0.0, 0.0, 0.0
    pts = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            p = segment_intersection(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
            if p: pts.append(p)
    if not pts: return 0.0, 0.0, 0.0
    h, w = depth.shape; valid = depth > 0; vr, stds, rings = [], [], []
    for cx, cy in pts:
        x1, x2, y1, y2 = max(0,int(cx-roi_h)), min(w,int(cx+roi_h+1)), max(0,int(cy-roi_h)), min(h,int(cy+roi_h+1))
        roi, roi_v = depth[y1:y2, x1:x2], valid[y1:y2, x1:x2]
        if roi.size == 0: continue
        vr.append(float(roi_v.mean()))
        roi_med = float(np.median(roi[roi_v])) if roi_v.any() else 0.0
        stds.append(float(np.std(roi[roi_v])) if roi_v.any() else 0.0)
        ox1, ox2, oy1, oy2 = max(0,int(cx-(roi_h+ring))), min(w,int(cx+roi_h+ring+1)), max(0,int(cy-(roi_h+ring))), min(h,int(cy+roi_h+ring+1))
        outer, outer_v = depth[oy1:oy2, ox1:ox2], valid[oy1:oy2, ox1:ox2]
        ring_m = outer_v.copy(); ring_m[y1-oy1:y2-oy1, x1-ox1:x2-ox1] = False
        rings.append(abs(roi_med - float(np.median(outer[ring_m]))) if ring_m.any() else 0.0)
    return float(np.mean(vr)), float(np.mean(stds)), float(np.mean(rings))

# Resize and normalize RGB image for ResNet
def preprocess_rgb(img_path: str, img_size: int = 224) -> torch.Tensor:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

# Flexible state_dict loader for different model saving formats
def load_state_dict_flexible(model: nn.Module, state: dict):
    if "state_dict" in state: state = state["state_dict"]
    try: model.load_state_dict(state)
    except:
        model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()}, strict=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=".")
    ap.add_argument("--depth_dir", default="Depth")
    ap.add_argument("--node_weights", default="weights/best.pt")
    ap.add_argument("--cls_weights", default="weights/2d3d_with_depth.pt")
    ap.add_argument("--out_csv", default="pred_strengthened.csv")
    args = ap.parse_args()

    node_model = YOLO(args.node_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_model = RebarNet(feat_dim=10).to(device)
    load_state_dict_flexible(cls_model, torch.load(args.cls_weights, map_location=device))
    cls_model.eval()

    images = sorted([f for f in os.listdir(args.folder) if f.lower().endswith((".jpg", ".png"))])
    rows = []
    for fname in images:
        img_path = os.path.join(args.folder, fname)
        base = os.path.splitext(fname)[0]
        depth_path = find_matching_depth(args.depth_dir, base)
        
        line_json = generate_lines_from_nodes(img_path, node_model)
        lines = flatten_lines(line_json)
        
        if lines:
            angles = [angle_of_line(l) for l in lines]
            m_n, t_n = compute_mean_angle_diff(angles)/90, compute_top_bottom_diff(lines, angles)/90
        else: m_n = t_n = 0.0

        if depth_path:
            depth = read_depth_any(depth_path)
            vr, med, std, db, gr = depth_global_features(depth)
            roi_vr, roi_std, roi_ring = roi_depth_features(depth, lines)
        else: vr=med=std=db=gr=roi_vr=roi_std=roi_ring=0.0

        img_t = preprocess_rgb(img_path).unsqueeze(0).to(device)
        feat_t = torch.tensor([[m_n, t_n, vr, med, std, db, gr, roi_vr, roi_std, roi_ring]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prob = torch.softmax(cls_model(img_t, feat_t), dim=1).cpu().numpy()[0]
            p3d = float(prob[1])

        # Two-stage logic with depth gate for gray zone
        if p3d >= 0.7: pred, reason = 1, "high_conf"
        elif p3d <= 0.3: pred, reason = 0, "low_conf"
        else:
            gate = (vr >= 0.2) and ((db/255 >= 0.15) or (gr/255 >= 0.2) or (roi_ring/255 >= 0.15))
            pred, reason = (1, "depth_gate_pass") if gate else (0, "gray_zone_fail")

        rows.append([fname, "3D" if pred else "2D", f"{p3d:.3f}", reason])
        print(f"[OK] {fname} -> {'3D' if pred else '2D'}")

    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows([["image", "pred", "p3d", "reason"]] + rows)

if __name__ == "__main__":
    main()
