"""
FARO Dataset Preparation Script

This script prepares the FARO dataset for training by:
1. Using crop data (120 files) for training and testing
2. Using plane data (119 files) for validation
3. Creating train/test split from crop data
"""

import os
import shutil
import random
from pathlib import Path

def prepare_faro_data():
    """准备FARO数据集"""
    
    # 数据路径
    # data_root = "pointcept/pointcept/datasets/0329_FARO_part_1"
    # crop_dir = os.path.join(data_root, "0329_FARO_part_1_crop_")
    # plane_dir = os.path.join(data_root, "0329_FARO_part_1_plane__")

    data_root = "pointcept/pointcept/datasets/data"
    crop_dir = os.path.join(data_root, "0329_FARO_part_1_crop_")
    plane_dir = os.path.join(data_root, "0329_FARO_part_1_plane__")
    
    # 检查路径是否存在
    if not os.path.exists(crop_dir):
        print(f"Error: Crop directory not found: {crop_dir}")
        return
    if not os.path.exists(plane_dir):
        print(f"Error: Plane directory not found: {plane_dir}")
        return
    
    # 创建新的目录结构
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    
    # 创建目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取crop数据文件列表
    crop_files = [f for f in os.listdir(crop_dir) if f.endswith('.ply')]
    crop_files.sort()
    
    # 获取plane数据文件列表
    plane_files = [f for f in os.listdir(plane_dir) if f.endswith('.ply')]
    plane_files.sort()
    
    print(f"Found {len(crop_files)} crop files")
    print(f"Found {len(plane_files)} plane files")
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 将crop数据分为训练集和测试集 (80% train, 20% test)
    random.shuffle(crop_files)
    split_idx = int(0.8 * len(crop_files))
    train_files = crop_files[:split_idx]
    test_files = crop_files[split_idx:]
    
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    print(f"Val files: {len(plane_files)}")
    
    # 复制训练文件
    print("Copying training files...")
    for file in train_files:
        src = os.path.join(crop_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy2(src, dst)
    
    # 复制测试文件
    print("Copying test files...")
    for file in test_files:
        src = os.path.join(crop_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.copy2(src, dst)
    
    # 复制验证文件
    print("Copying validation files...")
    for file in plane_files:
        src = os.path.join(plane_dir, file)
        dst = os.path.join(val_dir, file)
        shutil.copy2(src, dst)
    
    print("Data preparation completed!")
    print(f"Training set: {len(train_files)} files in {train_dir}")
    print(f"Validation set: {len(plane_files)} files in {val_dir}")
    print(f"Test set: {len(test_files)} files in {test_dir}")
    
    # 创建数据分割信息文件
    split_info = {
        "train": train_files,
        "val": plane_files,
        "test": test_files
    }
    
    import json
    with open(os.path.join(data_root, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    
    print("Split information saved to split_info.json")

if __name__ == "__main__":
    prepare_faro_data()
