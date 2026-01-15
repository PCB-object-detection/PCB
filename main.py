!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="NO0Vv5uoZbVOAefb8tS6")
project = rf.workspace("cnn-38lvj").project("team6_pcb_merge_data")
version = project.version(2)
dataset = version.download("yolov8")

#yolov8,11모델 설치
!pip install -U ultralytics

!pip install -U wandb ultralytics
!wandb login

!yolo detect train \
  model=yolov8s.pt \
  data=/content/Team6_PCB_Merge_Data-2/data.yaml \
  epochs=100 \
  batch=32 \
  patience=10 \
  imgsz=640 \
  project=/content/runs_pcb \
  name=y8s_e100_b32_es10 \
  cache=True

from google.colab import drive
drive.mount('/content/drive')

!cp /content/runs_pcb/y8s_e100_b32_es10/weights/best.pt \
   /content/drive/MyDrive/yolo_models/best.pt

!mkdir -p /content/drive/MyDrive/yolo_models

!yolo detect predict \
  model=/content/runs_pcb/y8s_e100_b32_es10/weights/best.pt \
  source=/content/Team6_PCB_Merge_Data-2/valid/images \
  imgsz=640 \
  conf=0.25 \
  save=True

if 'results' in locals() and not results.empty:
    max_map50 = results['metrics/mAP50(B)'].max()
    print(f"Maximum mAP_50 achieved during training: {max_map50:.4f}")
else:
    print("Results DataFrame is not available or empty. Please ensure 'results.csv' was loaded correctly.")

import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the training results directory in Google Drive backup
results_dir = '/content/drive/MyDrive/PCB_Backup/runs_pcb/y8s_e100_b32_es102/' # Corrected to es102
results_path = os.path.join(results_dir, 'results.csv')

# Load the results CSV file
if os.path.exists(results_path):
    # The results.csv often has a leading space in column names, so we strip them.
    results = pd.read_csv(results_path).rename(columns=lambda x: x.strip())
    print("Training results loaded successfully from Google Drive backup.")
    display(results.head())
else:
    print(f"Error: results.csv not found at {results_path}. Please ensure the backup is correct.")

pyplot as plt
import os

# Define the path to the training results directory in Google Drive backup
results_dir = '/content/drive/MyDrive/PCB_Backup/runs_pcb/y8s_e100_b32_es102/' # Corrected to es102
results_path = os.path.join(results_dir, 'results.csv')

# Load the results CSV file
if os.path.exists(results_path):
    # The results.csv often has a leading space in column names, so we strip them.
    results = pd.read_csv(results_path).rename(columns=lambda x: x.strip())
    print("Training results loaded successfully from Google Drive backup.")
    display(results.head())
else:
    print(f"Error: results.csv not found at {results_path}. Please ensure the backup is correct.")

if 'results' in locals() and not results.empty:
    # Plot mAP_50 and mAP_50-95
    plt.figure(figsize=(12, 6))
    plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP_50')
    plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP_50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP Value')
    plt.title('YOLOv8 Training mAP Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot box_loss, cls_loss, and dfl_loss
    plt.figure(figsize=(12, 6))
    plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss')
    plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss')
    plt.plot(results['epoch'], results['train/cls_loss'], label='Train Class Loss')
    plt.plot(results['epoch'], results['val/cls_loss'], label='Val Class Loss')
    plt.plot(results['epoch'], results['train/dfl_loss'], label='Train DFL Loss')
    plt.plot(results['epoch'], results['val/dfl_loss'], label='Val DFL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('YOLOv8 Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Results DataFrame is not available or empty. Please ensure 'results.csv' was loaded correctly.")

from pathlib import Path
import pandas as pd
import glob
import os

def print_pcb_performance_summary(run_dir=None):
    """
    PCB 결함 탐지 YOLOv5 모델 성능 요약 출력 (YOLOv5 results.csv 기반)
    - run_dir 미지정 시: runs/train/exp* 중 최신 폴더 자동 선택
    """

    print("=" * 70)
    print("YOLOv5 PCB 결함 탐지 모델 성능 요약")
    print("=" * 70)

    # \u2705 \ucd5c\uc2e0 exp \ud3f4\ub354 \uc790\ub3d9 \uc120\ud0dd
    if run_dir is None:
        exp_dirs = sorted(glob.glob("runs/train/exp*"), key=os.path.getmtime)
        if not exp_dirs:
            print("\u274c runs/train/exp* \ud3f4\ub354\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4.")
            print("=" * 70)
            return
        run_dir = exp_dirs[-1] # \ucd5c\uc2e0 \ud3f4\ub354 \uc120\ud0dd

    results_path = os.path.join(run_dir, 'results.csv')

    if not os.path.exists(results_path):
        print(f"\u274c results.csv not found at {results_path}")
        print("=" * 70)
        return

    results = pd.read_csv(results_path).rename(columns=lambda x: x.strip()) # \ucf6c\ub7fc\uba85 \uacf5\ubc31 \uc81c\uac70

    print(f"\n\u2702\ufe0f \ub300\uc0c1 run \ud3f4\ub354: {run_dir}")

    last = results.iloc[-1]

    print(f"\n\u2702\ufe0f \ucd5c\uc885 \uc5d0\ud3ed \uc131\ub2a5:")
    print(f"- Epoch: {int(last['epoch'])}")
    print(f"- Precision: {float(last['metrics/precision(B)']):.4f}")
    print(f"- Recall:    {float(last['metrics/recall(B)']):.4f}")
    print(f"- mAP@0.5:   {float(last['metrics/mAP50(B)']):.4f}")
    print(f"- mAP@0.5:0.95: {float(last['metrics/mAP50-95(B)']):.4f}")

    print(f"\n\u2702\ufe0f \ucd5c\uc885 \uc190\uc2e4 (Loss):")
    print(f"- Train Box Loss:    {float(last['train/box_loss']):.4f}")
    print(f"- Val Box Loss:      {float(last['val/box_loss']):.4f}")
    print(f"- Train Class Loss:  {float(last['train/cls_loss']):.4f}")
    print(f"- Val Class Loss:    {float(last['val/cls_loss']):.4f}")
    print(f"- Train DFL Loss:    {float(last['train/dfl_loss']):.4f}")
    print(f"- Val DFL Loss:      {float(last['val/dfl_loss']):.4f}")

    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")

    print(f"\n\u2702\ufe0f \uac00\uc911\uce58 \ud30c\uc77c:")
    print(f"- best.pt: {best_pt if os.path.exists(best_pt) else '\uac83\uc74c'}")
    print(f"- last.pt: {last_pt if os.path.exists(last_pt) else '\uac83\uc74c'}")

    print("=" * 70) 

======================================================================
YOLOv5 PCB 결함 탐지 모델 성능 요약
======================================================================

✂️ 대상 run 폴더: /content/drive/MyDrive/PCB_Backup/runs_pcb/y8s_e100_b32_es102

✂️ 최종 에폭 성능:
- Epoch: 100
- Precision: 0.9790
- Recall:    0.9749
- mAP@0.5:   0.9849
- mAP@0.5:0.95: 0.6014

✂️ 최종 손실 (Loss):
- Train Box Loss:    0.8828
- Val Box Loss:      1.3594
- Train Class Loss:  0.4405
- Val Class Loss:    0.4950
- Train DFL Loss:    0.8707
- Val DFL Loss:      1.0166

✂️ 가중치 파일:
- best.pt: /content/drive/MyDrive/PCB_Backup/runs_pcb/y8s_e100_b32_es102/weights/best.pt
- last.pt: /content/drive/MyDrive/PCB_Backup/runs_pcb/y8s_e100_b32_es102/weights/last.pt
======================================================================

pip install ultralytics # Ensure ultralytics is installed

import pandas as pd
import os
from ultralytics import YOLO
from IPython.display import display, Image, Markdown
import torch

# 1. Setup Paths - Pointing to Google Drive backup
run_dir = '/content/drive/MyDrive/PCB_Backup/runs_pcb/y8s_e100_b32_es102' # Corrected to es102
best_model_path = os.path.join(run_dir, 'weights', 'best.pt')

# 2. Load Model
model = YOLO(best_model_path)

# 3. Run Validation to get detailed metrics
print("Running validation to extract per-class metrics...")
metrics = model.val(data='/content/Team6_PCB_Merge_Data-2/data.yaml', split='val', verbose=False)

# --- A. Model Performance Comparison Table ---
# Get model info
model_info = model.info()
params_m = 11.14 # From summary: 11,137,922 parameters -> ~11.1M
model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)

# Metrics
map50 = metrics.box.map50
map5095 = metrics.box.map
precision = metrics.box.mp
recall = metrics.box.mr
f1_score = metrics.box.f1.mean() # Approximate average F1

# Estimate FPS (GPU) from speed dictionary
speed = metrics.speed
total_time_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
fps_gpu = 1000 / total_time_ms

# Create DataFrame
perf_data = {
    'Model': ['YOLOv8s'],
    'Size (MB)': [f"{model_size_mb:.1f}"],
    'Params (M)': [f"{params_m:.1f}"],
    'mAP@0.5': [f"{map50:.4f}"],
    'mAP@50-95': [f"{map5095:.4f}"],
    'Precision': [f"{precision:.4f}"],
    'Recall': [f"{recall:.4f}"],
    'F1': [f"{f1_score:.4f}"],
    'FPS (GPU)': [f"{fps_gpu:.1f}"],
    'FPS (CPU)': ['-'] # Requires separate benchmark
}

df_perf = pd.DataFrame(perf_data)

# --- B. Class-wise Performance Table ---
class_names = metrics.names
cls_data = []
for i, name in class_names.items():
    p = metrics.box.p[i]
    r = metrics.box.r[i]
    ap50 = metrics.box.ap50[i]

    cls_data.append({
        '결함': name,
        'Recall': f"{r:.4f}",
        'Precision': f"{p:.4f}",
        'AP@0.5': f"{ap50:.4f}"
    })

df_class = pd.DataFrame(cls_data)

# --- Display Tables ---
display(Markdown("### 1. 모델 성능 비교표 (Model Performance Comparison)"))
display(df_perf)

display(Markdown("### 2. 클래스별 성능 (Class-wise Performance - YOLOv8s)"))
display(df_class)

# --- Display Visualizations ---
display(Markdown("### 3. 시각화 자료 (Visualizations)"))

# Helper to display if exists
def show_img(filename, title):
    path = os.path.join(run_dir, filename)
    if os.path.exists(path):
        display(Markdown(f"**{title}**"))
        display(Image(filename=path, width=600))
    else:
        print(f"{title} not found at {path}")

# Confusion Matrix
show_img("confusion_matrix_normalized.png", "Confusion Matrix (Normalized)")
if not os.path.exists(os.path.join(run_dir, "confusion_matrix_normalized.png")):
     show_img("confusion_matrix.png", "Confusion Matrix")

# F1 Curve
show_img("F1_curve.png", "F1-Curve")

# Predictions (Labels vs Pred)
display(Markdown("**Labels vs Predictions (Validation Batch)**"))
col1, col2 = os.path.join(run_dir, "val_batch0_labels.jpg"), os.path.join(run_dir, "val_batch0_pred.jpg")
if os.path.exists(col1) and os.path.exists(col2):
    display(Image(filename=col1, width=400))
    print("Actual Labels (Left) vs Model Predictions (Right)")
    display(Image(filename=col2, width=400))
else:
    print("Validation batch images not found.")
