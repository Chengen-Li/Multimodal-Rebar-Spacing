# Rebar Geometry Analysis and 2D/3D Classification

This project provides a comprehensive pipeline for detecting rebar intersections, generating grid lines using PCA and Hough Transform, and classifying imagery into 2D or 3D structures using a ResNet-based fusion model (RGB + Geometric/Depth features).

## 🚀 Features

- **Intersection Detection**: High-precision detection of rebar nodes using YOLO.
- **Automated Line Generation**: Intelligent linking of nodes into structural grids using Principal Component Analysis (PCA).
- **Noise Filtering**: Robust line pruning using Hough Transform and angular mode detection.
- **Multimodal Classification**: A ResNet-18 based classifier that fuses RGB image data with 10-dimensional geometric and depth features.
- **Two-Stage Decision Logic**: Enhanced 2D/3D prediction with a dedicated "Depth Gate" for high-reliability results in uncertain cases.

---

## 📂 Project Structure

```text
REBAR_MCAE/
├── weights/
│   ├── best.pt                # YOLO model for node detection
│   └── 2d3d_with_depth.pt      # ResNet classifier weights (LFS tracked)
├── 2d3d_with_depth_predict.py  # Advanced inference script (RGB + Depth)
├── batch_predict.py            # Batch processing for RGB 2D/3D classification
├── line_extractor.py           # Core logic for node detection and line grouping
├── gen_line.py                 # Script to generate line data in JSON format
├── compute_features.py         # Geometric feature extraction from JSON
├── advanced_lbr.py             # Optimized utility functions for rebar analysis
└── README.md
```
## 🛠️ Installation
### 1. Clone the repository
```bash
git clone [https://github.com/Chengen-Li/Rebar-Geometry-Analysis.git](https://github.com/Chengen-Li/Rebar-Geometry-Analysis.git)
cd Rebar-Geometry-Analysis
```
### 2. Install dependencies
```bash
pip install torch torchvision ultralytics opencv-python scikit-image numpy pyyaml
```
## 💻 Usage
### 1. Generate Structural LinesTo detect rebar nodes and generate line segments from your images (stored in getline_image/):Bashpython gen_line.py
### 2. Perform 2D/3D ClassificationTo classify your images using the trained ResNet model (integrating RGB and Depth data):Bashpython 2d3d_with_depth_predict.py --folder ./your_images --depth_dir ./your_depth_maps
📊 MethodologyLine Extraction PipelineNode Detection: YOLO identifies rebar intersections.PCA Alignment: Determines the dominant horizontal and vertical axes of the rebar grid.Hough Grouping: Refines line segments and groups them based on geometric consistency.Classification LogicThe system evaluates the probability of an image being 3D based on a two-stage threshold:Confident Zone: Direct classification if $P(3D) \geq 0.70$ or $P(3D) \leq 0.30$.Gray Zone: If the model is uncertain (between $0.3$ and $0.7$), a Depth Gate is triggered. This gate analyzes depth variance, gradients, and ROI ring differences to make the final determination.
