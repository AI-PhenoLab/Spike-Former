"""
推理和可视化脚本 - FARO数据集

使用方法:
    python tools/inference_and_visualize_faro.py \
        --config-file configs/faro/semseg-pt-v3m1-0-cuda12.py \
        --weight exp/faro/my_faro_training/model/model_best.pth \
        --num-samples 10 \
        --output-dir vis_results

Author: Custom
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from collections import OrderedDict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn
from pointcept.utils.logger import get_root_logger
from pointcept.utils.visualization import save_point_cloud
from pointcept.engines.defaults import create_ddp_model
from pointcept.utils.config import DictAction
import pointcept.utils.comm as comm


def get_class_colors(num_classes):
    """为不同类别生成颜色"""
    # 使用不同的颜色方案
    if num_classes == 2:
        # crop (0) = 绿色, plane (1) = 红色
        colors = np.array([
            [0.0, 1.0, 0.0],  # 绿色 - crop
            [1.0, 0.0, 0.0],  # 红色 - plane
        ])
    else:
        # 为更多类别生成颜色
        colors = np.random.rand(num_classes, 3)
        colors[0] = [0.5, 0.5, 0.5]  # 背景色为灰色
    return colors


def visualize_segmentation(coord, pred_labels, gt_labels=None, class_names=None, save_path=None, show=True):
    """
    可视化分割结果
    
    Args:
        coord: 点云坐标 (N, 3) - numpy array
        pred_labels: 预测标签 (N,) - numpy array
        gt_labels: 真实标签 (N,)，可选 - numpy array
        class_names: 类别名称列表
        save_path: 保存路径
        show: 是否显示
    """
    # 确保是numpy数组
    if isinstance(coord, torch.Tensor):
        coord = coord.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    if gt_labels is not None and isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.cpu().numpy()
    
    num_classes = len(np.unique(pred_labels))
    class_colors = get_class_colors(num_classes)
    
    # 创建预测结果点云
    pred_colors = class_colors[pred_labels]
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(coord)
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
    
    # 保存预测结果
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd_pred)
        print(f"保存预测结果到: {save_path}")
    
    # 如果有真实标签，创建对比可视化
    if gt_labels is not None:
        gt_colors = class_colors[gt_labels]
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(coord)
        pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
        
        # 保存真实标签
        if save_path:
            gt_save_path = save_path.replace("_pred.ply", "_gt.ply")
            o3d.io.write_point_cloud(gt_save_path, pcd_gt)
            print(f"保存真实标签到: {gt_save_path}")
        
        # 显示对比
        if show:
            print("\n显示预测结果（按Q关闭）...")
            o3d.visualization.draw_geometries([pcd_pred], window_name="预测结果")
            print("\n显示真实标签（按Q关闭）...")
            o3d.visualization.draw_geometries([pcd_gt], window_name="真实标签")
    else:
        if show:
            print("\n显示预测结果（按Q关闭）...")
            o3d.visualization.draw_geometries([pcd_pred], window_name="预测结果")
    
    return pcd_pred


def load_model(cfg, weight_path):
    """加载模型和权重"""
    logger = get_root_logger()
    
    # 构建模型
    model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_parameters}")
    
    # 移动到GPU
    model = model.cuda()
    
    # 加载权重
    if os.path.isfile(weight_path):
        logger.info(f"加载权重: {weight_path}")
        checkpoint = torch.load(weight_path, map_location="cuda", weights_only=False)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            # 处理DDP前缀
            if key.startswith("module."):
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
            weight[key] = value
        model.load_state_dict(weight, strict=True)
        logger.info(f"成功加载权重 (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        raise RuntimeError(f"权重文件不存在: {weight_path}")
    
    model.eval()
    return model


def inference_and_visualize(cfg, weight_path, num_samples=None, output_dir="vis_results", show=True):
    """
    在测试集上进行推理并可视化
    
    Args:
        cfg: 配置对象
        weight_path: 模型权重路径
        num_samples: 要可视化的样本数量（None表示全部）
        output_dir: 输出目录
        show: 是否显示可视化结果
    """
    logger = get_root_logger()
    
    # 加载模型
    model = load_model(cfg, weight_path)
    
    # 构建测试数据集
    test_dataset = build_dataset(cfg.data.test)
    
    # 自定义 collate_fn，保留 fragment_list 不变
    def test_collate_fn(batch):
        # batch 是一个包含单个字典的列表（batch_size=1）
        if len(batch) == 1:
            return batch[0]  # 直接返回字典，不进行 collate
        else:
            # 如果 batch_size > 1，使用标准 collate_fn
            return collate_fn(batch)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Windows兼容性
        pin_memory=True,
        collate_fn=test_collate_fn,  # 使用自定义 collate_fn
    )
    
    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"开始推理和可视化...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 类别名称
    class_names = cfg.data.get("names", [f"class_{i}" for i in range(cfg.data.num_classes)])
    
    # 推理循环
    total_samples = len(test_loader)
    num_samples = num_samples if num_samples is not None else total_samples
    num_samples = min(num_samples, total_samples)
    
    for idx, data_dict in enumerate(test_loader):
        if idx >= num_samples:
            break
        
        # test_collate_fn 直接返回字典（batch_size=1时）
        # 但为了兼容性，检查是否是列表
        if isinstance(data_dict, (list, tuple)):
            data_dict = data_dict[0]
        
        # 调试：打印第一个样本的所有键
        if idx == 0:
            logger.info(f"第一个样本的 data_dict keys: {list(data_dict.keys())}")
            logger.info(f"data_dict type: {type(data_dict)}")
            logger.info(f"test_dataset.test_mode: {test_dataset.test_mode}")
            logger.info(f"test_dataset.test_cfg: {test_dataset.test_cfg}")
            if "fragment_list" in data_dict:
                logger.info(f"fragment_list type: {type(data_dict['fragment_list'])}, length: {len(data_dict['fragment_list']) if hasattr(data_dict['fragment_list'], '__len__') else 'N/A'}")
        
        # 检查 fragment_list 是否存在
        if "fragment_list" not in data_dict:
            logger.error(f"fragment_list not found in data_dict. Available keys: {list(data_dict.keys())}")
            logger.error(f"data_dict type: {type(data_dict)}")
            logger.error(f"test_dataset.test_mode: {test_dataset.test_mode}")
            logger.error(f"test_dataset.test_cfg: {test_dataset.test_cfg}")
            logger.error(f"test_dataset.aug_transform: {test_dataset.aug_transform}")
            logger.error(f"test_dataset.test_voxelize: {test_dataset.test_voxelize}")
            logger.error(f"Skipping sample {idx+1}")
            continue
        
        fragment_list = data_dict.pop("fragment_list")
        segment = data_dict.pop("segment")  # 真实标签
        data_name = data_dict.pop("name")
        
        logger.info(f"[{idx+1}/{num_samples}] 处理: {data_name}")
        
        # 收集所有片段的预测
        pred = torch.zeros((segment.size, cfg.data.num_classes)).cuda()
        
        with torch.no_grad():
            for i in range(len(fragment_list)):
                input_dict = collate_fn([fragment_list[i]])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                
                idx_part = input_dict["index"]
                pred_part = model(input_dict)["seg_logits"]  # (n, k)
                pred_part = torch.nn.functional.softmax(pred_part, -1)
                
                # 聚合预测结果（累加softmax概率）
                bs = 0
                for be in input_dict["offset"]:
                    pred[idx_part[bs:be], :] += pred_part[bs:be]
                    bs = be
                
                if cfg.empty_cache:
                    torch.cuda.empty_cache()
        
        # 获取预测标签
        pred_labels = pred.argmax(dim=1).cpu().numpy()
        
        # 处理原始坐标和标签（如果有inverse映射）
        if "origin_segment" in data_dict.keys() and "inverse" in data_dict.keys():
            # 使用原始坐标和标签
            coord = data_dict["coord"].cpu().numpy() if isinstance(data_dict["coord"], torch.Tensor) else data_dict["coord"]
            segment_cpu = data_dict["origin_segment"].cpu().numpy() if isinstance(data_dict["origin_segment"], torch.Tensor) else data_dict["origin_segment"]
            # 将预测结果映射回原始点云
            pred_labels = pred_labels[data_dict["inverse"]]
        else:
            # 从片段中聚合坐标
            if len(fragment_list) > 0:
                # 收集所有片段的坐标
                coords_list = []
                for frag in fragment_list:
                    frag_coord = frag["coord"]
                    if isinstance(frag_coord, torch.Tensor):
                        frag_coord = frag_coord.cpu().numpy()
                    coords_list.append(frag_coord)
                coord = np.concatenate(coords_list, axis=0)
            else:
                logger.warning(f"  警告: {data_name} 没有片段数据")
                continue
            
            segment_cpu = segment.cpu().numpy() if isinstance(segment, torch.Tensor) else segment
        
        # 计算准确率
        accuracy = (pred_labels == segment_cpu).mean()
        logger.info(f"  准确率: {accuracy*100:.2f}%")
        
        # 可视化
        save_path = os.path.join(output_dir, f"{data_name}_pred.ply")
        visualize_segmentation(
            coord=coord,
            pred_labels=pred_labels,
            gt_labels=segment_cpu,
            class_names=class_names,
            save_path=save_path,
            show=show and idx < 3  # 只显示前3个样本
        )
    
    logger.info(f"推理和可视化完成！结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="FARO数据集推理和可视化")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="配置文件路径",
    )
    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="模型权重路径（.pth文件）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="要可视化的样本数量（默认：全部）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vis_results",
        help="输出目录",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示可视化窗口（仅保存文件）",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="其他配置选项（格式：key=value）",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = default_config_parser(args.config_file, args.options if args.options else None)
    cfg = default_setup(cfg)
    
    # 设置测试模式
    cfg.data.test.split = "test"
    
    # 运行推理和可视化
    inference_and_visualize(
        cfg=cfg,
        weight_path=args.weight,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

