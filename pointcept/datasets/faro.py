"""
FARO Dataset for Point Transformer V3

Author: Custom Dataset
Please cite our work if the code is helpful to you.
"""

import os
import glob
import random
import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class FARODataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "feat",
    ]
    
    def __init__(
        self,
        data_root,
        split="train",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        use_color_labels=False,  # 是否从颜色中提取类别标签
        num_classes=2,  # 类别数量
        color_mapping_mode="kmeans",  # "kmeans", "centers", 或 "threshold"
        color_centers=None,  # 预定义的颜色中心 (RGB, 0-1范围), 例如: [[1,0,0], [0,0,1]]
        color_threshold=0.5,  # 阈值模式的阈值 (0-1范围)
        color_channel="mean",  # 阈值模式使用的颜色通道: "r", "g", "b", "mean", "max", "min"
        threshold_low_class=0,  # 低于阈值的点的类别
        threshold_high_class=1,  # 高于阈值的点的类别
        **kwargs,
    ):
        # 调用父类初始化（但需要先设置一些属性）
        self.data_root = data_root
        self.split = split
        self.test_mode = test_mode
        self.test_cfg = test_cfg
        self.cache = cache
        self.loop = loop
        self.use_color_labels = use_color_labels
        self.num_classes = num_classes
        self.color_mapping_mode = color_mapping_mode
        self.color_centers = color_centers
        self.color_threshold = color_threshold
        self.color_channel = color_channel
        self.threshold_low_class = threshold_low_class
        self.threshold_high_class = threshold_high_class
        
        # 初始化变换
        if transform is None:
            transform = []
        self.transform = Compose(transform)
        
        # 获取数据列表（必须在调用父类之前）
        self.data_list = self.get_data_list()
        
        # 初始化测试相关的变换（如果 test_mode=True）
        if self.test_mode and self.test_cfg is not None:
            from copy import deepcopy
            from pointcept.datasets.transform import TRANSFORMS
            
            # 初始化 aug_transform
            self.aug_transform = []
            if "aug_transform" in self.test_cfg:
                for aug_list in self.test_cfg["aug_transform"]:
                    # Each aug_list is already a list of transform configs;
                    # Compose will build them, so avoid building twice.
                    self.aug_transform.append(Compose(aug_list))
            
            # 初始化 test_voxelize
            self.test_voxelize = None
            if "voxelize" in self.test_cfg and self.test_cfg["voxelize"] is not None:
                self.test_voxelize = TRANSFORMS.build(self.test_cfg["voxelize"])
            
            # 初始化 test_crop
            self.test_crop = None
            if "crop" in self.test_cfg and self.test_cfg["crop"] is not None:
                self.test_crop = TRANSFORMS.build(self.test_cfg["crop"])
            
            # 初始化 post_transform
            self.post_transform = Compose([])
            if "post_transform" in self.test_cfg:
                # test_cfg already provides config dicts; let Compose build them.
                self.post_transform = Compose(self.test_cfg["post_transform"])
        else:
            self.aug_transform = []
            self.test_voxelize = None
            self.test_crop = None
            self.post_transform = Compose([])
        
        logger = get_root_logger()
        logger.info(f"FARO Dataset - {split}: {len(self.data_list)} samples")

    def get_data_list(self):
        """获取数据文件列表"""
        logger = get_root_logger()
        data_list = []
        
        # 只要存在 data 子目录，或者显式指定 use_color_labels=True，就启用颜色标签模式
        data_subdir = os.path.join(self.data_root, "data")
        use_color_mode = self.use_color_labels or os.path.exists(data_subdir)

        if use_color_mode:
            # 检查是否存在 data 子目录
            if os.path.exists(data_subdir):
                data_path = data_subdir
            else:
                # 如果没有 data 子目录，尝试从 data_root 直接读取
                data_path = self.data_root

            # 直接从 data 目录（或 data_root）读取所有 PLY 文件
            ply_files = glob.glob(os.path.join(data_path, "*.ply"))
            logger.info(f"使用颜色标签模式，从 {data_path} 读取数据")
        else:
            # 根据split选择不同的子目录
            if self.split == "train":
                subdir = "train"
            elif self.split == "val":
                subdir = "val"
            else:
                subdir = "test"  # test split
            
            data_path = os.path.join(self.data_root, subdir)
            
            if not os.path.exists(data_path):
                raise ValueError(f"Data path does not exist: {data_path}")
            
            # 获取所有PLY文件
            ply_files = glob.glob(os.path.join(data_path, "*.ply"))
        
        ply_files.sort()  # 确保顺序一致
        
        for ply_file in ply_files:
            filename = os.path.basename(ply_file)
            name = filename.split(".")[0]
            
            if self.use_color_labels:
                # 使用颜色标签模式：不需要从文件名提取类别
                class_id = None  # 将在 get_data 中从颜色提取
                original_name = name
            else:
                # 从文件名中提取类别标签
                # 文件名格式: class0_xxx.ply 或 class1_xxx.ply
                if filename.startswith("class0_"):
                    class_id = 0
                    # 移除类别前缀，保留原始文件名
                    original_name = name.replace("class0_", "", 1)
                elif filename.startswith("class1_"):
                    class_id = 1
                    original_name = name.replace("class1_", "", 1)
                else:
                    # 如果没有类别前缀，尝试从原始文件夹结构推断
                    # 检查是否在原始类别文件夹中
                    if "crop_" in ply_file:
                        class_id = 0
                    elif "plane__" in ply_file:
                        class_id = 1
                    else:
                        # 默认类别为 0，并记录警告
                        class_id = 0
                        logger.warning(f"无法确定文件 {ply_file} 的类别，默认使用类别 0")
                    original_name = name
            
            data_list.append({
                "path": ply_file,
                "name": original_name,
                "class_id": class_id,  # 如果使用颜色标签，这里为 None
            })
        
        # 如果是训练集，打乱数据列表
        if self.split == "train":
            random.shuffle(data_list)
            logger.info(f"Shuffled {len(data_list)} training samples")
        
        return data_list

    def _color_to_label(self, colors):
        """将颜色转换为类别标签"""
        if self.color_mapping_mode == "kmeans":
            # 使用 K-means 聚类
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError("需要安装 sklearn: pip install scikit-learn")
            
            # 将颜色转换为整数 (0-255)
            colors_int = (colors * 255).astype(np.float32)
            
            # 使用 K-means 聚类
            kmeans = KMeans(n_clusters=self.num_classes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(colors_int)
            return labels.astype(np.int64)
        
        elif self.color_mapping_mode == "centers":
            # 使用预定义的颜色中心
            if self.color_centers is None:
                raise ValueError("使用 'centers' 模式时必须提供 color_centers 参数")
            
            # 将颜色中心转换为 numpy 数组
            centers = np.array(self.color_centers, dtype=np.float32)
            
            # 计算每个点到各个颜色中心的距离
            # colors: (N, 3), centers: (K, 3)
            # 计算欧氏距离
            distances = np.sqrt(((colors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
            # 找到最近的类别
            labels = np.argmin(distances, axis=1)
            return labels.astype(np.int64)
        
        elif self.color_mapping_mode == "threshold":
            # 基于阈值的二值化模式
            # 提取指定的颜色通道值
            if self.color_channel == "r":
                channel_values = colors[:, 0]
            elif self.color_channel == "g":
                channel_values = colors[:, 1]
            elif self.color_channel == "b":
                channel_values = colors[:, 2]
            elif self.color_channel == "mean":
                channel_values = colors.mean(axis=1)
            elif self.color_channel == "max":
                channel_values = colors.max(axis=1)
            elif self.color_channel == "min":
                channel_values = colors.min(axis=1)
            else:
                raise ValueError(f"未知的颜色通道: {self.color_channel}，支持: 'r', 'g', 'b', 'mean', 'max', 'min'")
            
            # 基于阈值进行二值化
            # 大于等于阈值的点 -> threshold_high_class
            # 小于阈值的点 -> threshold_low_class
            labels = np.where(channel_values >= self.color_threshold, 
                             self.threshold_high_class, 
                             self.threshold_low_class)
            return labels.astype(np.int64)
        
        else:
            raise ValueError(f"未知的颜色映射模式: {self.color_mapping_mode}，支持: 'kmeans', 'centers', 'threshold'")
    
    def get_data(self, idx):
        """加载单个数据样本"""
        data_info = self.data_list[idx]
        name = data_info["name"]
        class_id = data_info["class_id"]  # 如果使用颜色标签，这里为 None
        
        # 检查缓存（使用shared_dict）
        if self.cache:
            cache_name = f"pointcept-{name}-color" if self.use_color_labels else f"pointcept-{name}-class{class_id}"
            try:
                return shared_dict(cache_name)
            except:
                pass  # 如果缓存不存在，继续加载数据
        
        # 加载PLY文件
        pcd = o3d.io.read_point_cloud(data_info["path"])
        
        # 转换为numpy数组
        coord = np.asarray(pcd.points, dtype=np.float32)
        color = np.asarray(pcd.colors, dtype=np.float32)
        
        # 从颜色中提取类别标签
        if self.use_color_labels:
            # 从颜色中提取类别标签
            segment = self._color_to_label(color)
        else:
            # 为所有点分配相同的类别标签（因为整个文件属于一个类别）
            segment = np.full((coord.shape[0],), class_id, dtype=np.int64)
        
        data_dict = {
            "coord": coord,
            "color": color,
            "feat": color,  # 使用颜色作为特征
            "segment": segment,  # 添加类别标签
            "name": name,
        }
        
        # 缓存数据（如果需要）
        if self.cache:
            cache_name = f"pointcept-{name}-color" if self.use_color_labels else f"pointcept-{name}-class{class_id}"
            shared_dict(cache_name, data_dict)
        
        return data_dict

    def prepare_train_data(self, idx):
        """准备训练数据"""
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict
    
    def prepare_test_data(self, idx):
        """准备测试数据，生成 fragment_list"""
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict.keys():
            assert "inverse" in data_dict.keys()
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            from copy import deepcopy
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        """获取数据项"""
        if self.test_mode:
            # 测试模式：使用 prepare_test_data 生成 fragment_list
            return self.prepare_test_data(idx)
        else:
            # 训练/验证模式：使用 prepare_train_data
            return self.prepare_train_data(idx)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_list)

    def get_classes(self):
        """返回类别信息"""
        if self.num_classes == 2:
            # 返回类别名称列表
            # 类别 0: crop, 类别 1: plane
            return ["crop", "plane"]
        else:
            # 如果类别数不是2，返回通用类别名称
            return [f"class_{i}" for i in range(self.num_classes)]
