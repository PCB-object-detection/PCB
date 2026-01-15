import os
import cv2
import albumentations as A
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from src.utils.utils import setup_logger
from collections import defaultdict
from tqdm import tqdm

# Logger setup
project_root = Path(__file__).resolve().parents[2]  # PCB/
log_dir = project_root / "logs/augmentation"

logger = setup_logger(
    name="OfflineAugmentation",
    log_dir=log_dir,
    log_level="INFO"
)

class Augmentor:
    """
    Offline augmentation pipeline using Albumentations.
    Each class reaches its target count using CSV-based paths.
    """
    def __init__(self, img_size: int, save_dir: str, pcb_optimized: bool = True,
                 normal_class_id: int = -1, class_target_counts: Optional[Dict[int, int]] = None):
        self.img_size = img_size
        self.save_dir = Path(save_dir)
        self.normal_class_id = normal_class_id
        self.class_target_counts = class_target_counts or {}
        self.class_aug_counts = defaultdict(int)

        self.image_dir = self.save_dir / "aug_images"
        self.label_dir = self.save_dir / "aug_labels"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)

        self.transform = self._build_transforms(pcb_optimized)

        logger.info(f"Offline Augmentor initialized, save_dir={self.save_dir}, img_size={img_size}")

    def _build_transforms(self, pcb_optimized: bool) -> A.Compose:
        if pcb_optimized:
            transforms = [
                A.RandomScale(scale_limit=(-0.5, 0.2), p=0.7),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=lambda: np.random.choice([[255,255,255],[0,0,0],[128,128,128]]), p=0.7),
                A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
                A.RandomShadow(shadow_roi=(0,0.5,1,1), p=0.1),
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.2),
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.3),
                A.Affine(translate_percent=0.1, scale=(0.9,1.1), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
                A.GaussNoise(std_range=(0.05,0.2), p=0.2),
                A.GaussianBlur(blur_limit=(2,5), p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=10,   # 색조 변화 범위 (-10 ~ +10)
                    sat_shift_limit=30,   # 채도 변화 범위 (-30 ~ +30)
                    val_shift_limit=20,   # 명도 변화 범위 (-20 ~ +20)
                    p=0.4                 # 적용 확률
                ),
                A.Resize(self.img_size, self.img_size)
            ]
        else:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.Resize(self.img_size, self.img_size)
            ]
        return A.Compose(transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))

    def augment_image(self, image_path: str, bboxes: List[List[float]], class_labels: List[int], base_name: str, target_classes: List[int]) -> Dict[int,int]:
        created_per_class = defaultdict(int)
        unique_classes = set(class_labels)
        if unique_classes == {self.normal_class_id}:
            return created_per_class

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Only augment classes that still need images
        needed_classes = [c for c in unique_classes if c in target_classes and self.class_aug_counts[c] < self.class_target_counts.get(c,0)]
        if not needed_classes:
            return created_per_class

        # Max number of augmentation attempts based on remaining counts
        max_remaining = max([self.class_target_counts[c] - self.class_aug_counts[c] for c in needed_classes])

        aug_idx = 0
        while any(self.class_aug_counts[c] < self.class_target_counts.get(c,0) for c in needed_classes) and aug_idx < max_remaining*2:
            augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img, aug_bboxes, aug_labels = augmented["image"], augmented["bboxes"], augmented["class_labels"]
            aug_classes_in_img = set(aug_labels)

            saved = False
            for c in aug_classes_in_img:
                if c in needed_classes and c != self.normal_class_id and self.class_aug_counts[c] < self.class_target_counts[c]:
                    img_name = f"aug_{base_name}_cls{c}_idx{aug_idx}.jpg"
                    label_name = f"aug_{base_name}_cls{c}_idx{aug_idx}.txt"
                    cv2.imwrite(str(self.image_dir / img_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    self._save_yolo_label(self.label_dir / label_name, aug_bboxes, aug_labels)
                    self.class_aug_counts[c] += 1
                    created_per_class[c] += 1
                    saved = True
            if not saved:
                break
            aug_idx += 1

        return created_per_class

    def _save_yolo_label(self, label_path: Path, bboxes: List[List[float]], labels: List[int]):
        with open(label_path, "w") as f:
            for cls, bbox in zip(labels, bboxes):
                bbox_str = " ".join([f"{x:.6f}" for x in bbox])
                f.write(f"{cls} {bbox_str}\n")

if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file with image_path, label_path, class_id")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--save_dir", type=str, default="dataset/roboflow/train")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    # Set target augmentation per class
    class_target_counts = {0:100, 1:100, 2:100, 3:100, 4:100, 5:100}
    augmentor = Augmentor(img_size=args.img_size, save_dir=args.save_dir, normal_class_id=-1, class_target_counts=class_target_counts)

    original_train_count = len(df[df['split']=='train']) if 'split' in df.columns else len(df)

    for cls in class_target_counts.keys():
        cls_rows = []
        for _, row in df.iterrows():
            label_path = row['label_path']
            with open(label_path, 'r') as f:
                class_labels = [int(line.strip().split()[0]) for line in f.readlines()]
            if cls in class_labels:
                cls_rows.append(row)
        logger.info(f"Class {cls}: {len(cls_rows)} images containing this class")
        np.random.shuffle(cls_rows)
        for row in cls_rows:
            image_path = row['image_path']
            label_path = row['label_path']
            base_name = Path(image_path).stem
            with open(label_path, 'r') as f:
                lines = f.readlines()
                bboxes = [[float(x) for x in line.strip().split()[1:]] for line in lines]
                class_labels = [int(line.strip().split()[0]) for line in lines]
            augmentor.augment_image(image_path=image_path, bboxes=bboxes, class_labels=class_labels, base_name=base_name, target_classes=[cls])
            if augmentor.class_aug_counts[cls] >= class_target_counts[cls]:
                break

    # Summary log
    logger.info("\n===== Offline Augmentation Summary =====")
    total_aug = 0
    for cls, target in class_target_counts.items():
        count = augmentor.class_aug_counts.get(cls,0)
        total_aug += count
        logger.info(f"Class {cls}: augmented {count} images (target: {target})")
    logger.info(f"Original train images: {original_train_count}")
    logger.info(f"Total images after augmentation: {original_train_count + total_aug}")

    print("✅ Offline augmentation finished")