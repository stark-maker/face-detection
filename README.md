# 计算机视觉项目集（Python + OpenCV + YOLO）

一个基于 Python 的计算机视觉练习项目集，包含实时人脸检测、颜色追踪与 YOLO 目标检测。

## 项目列表

### 1. 实时人脸检测（face_detect.py）
基于 OpenCV Haar 级联分类器，实时检测并框选人脸，显示 FPS 与人数，支持截图保存。

### 2. 颜色追踪（color_track.py）
基于 HSV 颜色空间 + 轮廓检测，实时追踪指定颜色的物体并画框。

### 3. YOLO 实时物体检测（yolo_detect.py）
基于 Ultralytics YOLOv8，实时检测并框出画面中的常见物体（人、杯子、手机等 80 类），
显示类别、置信度与 FPS。

## 技术栈
- Python 3
- OpenCV
- Ultralytics YOLOv8 + PyTorch

## 环境安装
```bash
pip install opencv-python ultralytics

