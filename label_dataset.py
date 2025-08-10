#!/usr/bin/env python3

import os
from pathlib import Path
from PIL import Image
import shutil
import yaml


# === CONFIG ===
IMG_ROOT = Path("dataset/images")
LBL_ROOT = Path("dataset/labels")
YAML_PATH = Path("dataset/dataset.yaml")

# === INIT ===
IMG_ROOT.mkdir(parents=True, exist_ok=True)
LBL_ROOT.mkdir(parents=True, exist_ok=True)

# === SUPPORTED IMAGES ===
VALID_EXT = ['.jpg', '.jpeg', '.png']

def create_yaml():
    classes = sorted([d.name for d in IMG_ROOT.iterdir() if d.is_dir()])
    with open(YAML_PATH, 'w') as f:
        yaml.dump({
            'path': 'dataset',
            'train': 'images',
            'val': 'images',  # Lazy same folder
            'names': classes
        }, f)
    print(f"[INFO] dataset.yaml updated with classes: {classes}")

def auto_label():
    classes = sorted([d.name for d in IMG_ROOT.iterdir() if d.is_dir()])
    class_map = {name: idx for idx, name in enumerate(classes)}

    for class_name in classes:
        img_dir = IMG_ROOT / class_name
        label_dir = LBL_ROOT / class_name
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in VALID_EXT:
                continue

            # Fake label center/midpoint box (just placeholder)
            with Image.open(img_file) as img:
                # w, h = img.size
                # x_center, y_center, box_w, box_h = 0.5, 0.5, 0.95, 0.95
                # Center of image (normalized)
                x_center = 0.5
                y_center = 0.5

                # Box size (adjusted to ~60% of image, looks more like real object area)
                box_w = 0.6
                box_h = 0.6

                # Safety clamp
                box_w = min(box_w, 1.0)
                box_h = min(box_h, 1.0)

            label = f"{class_map[class_name]} {x_center} {y_center} {box_w} {box_h}\n"
            label_path = label_dir / (img_file.stem + ".txt")
            with open(label_path, 'w') as f:
                f.write(label)

    print("[INFO] Auto-labeling complete.")

if __name__ == "__main__":
    auto_label()
    create_yaml()
