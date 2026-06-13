"""
分类评估脚本 - FARO数据集
对测试集进行分类评估，生成混淆矩阵、ROC曲线、PR曲线和分类指标

使用方法:
    python tools/simple_visualize_faro.py \
        --config-file configs/faro/semseg-pt-v3m1-0-cuda12.py \
        --weight exp/faro/my_faro_training/model/model_best.pth \
        --input-dir pointcept/datasets/0329_FARO_part_1/test \
        --output-dir evaluation_results

输出:
    - confusion_matrix.png: 混淆矩阵
    - roc_curve.png: ROC曲线
    - pr_curve.png: PR曲线
    - metrics.txt: 详细分类指标
"""

import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d  # 仅用于加载PLY文件
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import seaborn as sns

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11
rcParams['figure.titlesize'] = 14

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.config import DictAction
import glob


def get_ground_truth_label(ply_file):
    """
    从文件名或路径中提取真实标签
    
    Args:
        ply_file: PLY文件路径
    
    Returns:
        class_id: 类别ID (0=crop, 1=plane)
    """
    filename = os.path.basename(ply_file)
    
    # 从文件名中提取类别
    if filename.startswith("class0_"):
        return 0
    elif filename.startswith("class1_"):
        return 1
    elif "crop_" in ply_file:
        return 0
    elif "plane__" in ply_file:
        return 1
    else:
        # 默认返回-1（未知）
        return -1


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # 使用更柔和的配色方案：从浅蓝到深蓝的渐变
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('True Label', fontsize=12, fontfamily='Times New Roman')
    plt.xlabel('Predicted Label', fontsize=12, fontfamily='Times New Roman')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, class_names, save_path=None):
    """Plot ROC curve"""
    # Binary classification: use positive class (class 1) probability
    fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    # 使用更柔和的配色：深绿色替代橙色
    plt.plot(fpr, tpr, color='#2E7D32', lw=2.5, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#757575', lw=2, linestyle='--', 
             label='Random Guess (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontfamily='Times New Roman')
    plt.ylabel('True Positive Rate', fontsize=12, fontfamily='Times New Roman')
    plt.title('ROC Curve', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    plt.legend(loc="lower right", fontsize=11, prop={'family': 'Times New Roman'})
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    plt.close()
    
    return roc_auc


def plot_pr_curve(y_true, y_scores, class_names, save_path=None):
    """Plot Precision-Recall curve"""
    # Binary classification: use positive class (class 1) probability
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    # 使用更柔和的配色：深紫色替代深蓝色
    plt.plot(recall, precision, color='#6A1B9A', lw=2.5,
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontfamily='Times New Roman')
    plt.ylabel('Precision', fontsize=12, fontfamily='Times New Roman')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    plt.legend(loc="lower left", fontsize=11, prop={'family': 'Times New Roman'})
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    plt.close()
    
    return pr_auc


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


def inference_single_file(model, coord, color, cfg, device="cuda"):
    """
    对单个点云文件进行推理（分类模式：整个点云一个类别）
    
    Args:
        model: 训练好的模型
        coord: 点云坐标 (N, 3)
        color: 点云颜色 (N, 3)
        cfg: 配置对象
        device: 设备
    
    Returns:
        pred_class: 预测的类别（整数，0或1）
        confidence: 预测的置信度（0-1之间）
        pred_probs: 预测概率 [类别0的概率, 类别1的概率]
    """
    # 转换为tensor
    if isinstance(coord, np.ndarray):
        coord = torch.from_numpy(coord).float()
    if isinstance(color, np.ndarray):
        color = torch.from_numpy(color).float()
    
    # 归一化颜色
    if color.max() > 1.0:
        color = color / 255.0
    
    # 中心化坐标（根据训练时的预处理）
    coord_mean = coord.mean(dim=0)
    coord = coord - coord_mean
    
    # 如果点太多，进行下采样
    max_points = 50000  # 限制最大点数以避免内存问题
    if coord.shape[0] > max_points:
        indices = np.random.choice(coord.shape[0], max_points, replace=False)
        coord_sampled = coord[indices]
        color_sampled = color[indices]
    else:
        coord_sampled = coord
        color_sampled = color
    
    # 获取 grid_size（从配置的 test_cfg 中）
    # 默认使用训练时的 grid_size，如果没有则使用 0.25
    grid_size = 0.25  # 默认值
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'test'):
        test_cfg = cfg.data.test
        if hasattr(test_cfg, 'pipeline'):
            for transform in test_cfg.pipeline:
                if isinstance(transform, dict) and transform.get('type') == 'GridSample':
                    grid_size = transform.get('grid_size', 0.25)
                    break
    
    # 构建输入字典（按照训练时的格式）
    # 注意：grid_size 应该是标量值（float），Point 类会自动处理
    input_dict = {
        "coord": coord_sampled.to(device),  # (N, 3)
        "feat": torch.cat([coord_sampled, color_sampled], dim=-1).to(device),  # (N, 6)
        "offset": torch.tensor([coord_sampled.shape[0]], dtype=torch.long).to(device),
        "grid_size": grid_size,  # 必需：用于序列化（标量值）
        "batch": torch.zeros(coord_sampled.shape[0], dtype=torch.long).to(device),  # 必需：用于序列化
    }
    
    # 推理
    with torch.no_grad():
        output = model(input_dict)
        pred_logits = output["seg_logits"]  # (N, num_classes)
        
        # 计算每个点的预测概率
        pred_probs = torch.softmax(pred_logits, dim=-1)  # (N, num_classes)
        
        # 方法1：投票法 - 统计每个类别的点数
        point_predictions = pred_logits.argmax(dim=-1).cpu().numpy()  # (N,)
        class_counts = np.bincount(point_predictions, minlength=cfg.data.num_classes)
        pred_class = np.argmax(class_counts)
        
        # 方法2：平均概率法 - 计算平均概率
        avg_probs = pred_probs.mean(dim=0).cpu().numpy()  # (num_classes,)
        pred_class_avg = np.argmax(avg_probs)
        confidence = avg_probs[pred_class_avg]
        
        # 使用平均概率法的结果（更稳定）
        pred_class = pred_class_avg
        pred_probs = avg_probs  # 返回概率分布
    
    return pred_class, confidence, pred_probs


def main():
    parser = argparse.ArgumentParser(description="FARO数据集简单推理和可视化")
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
        "--input-dir",
        type=str,
        default="pointcept/datasets/0329_FARO_part_1/test",
        help="输入PLY文件目录",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="要评估的样本数量（默认：全部）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vis_results",
        help="输出目录",
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
    
    logger = get_root_logger()
    
    # 加载模型
    model = load_model(cfg, args.weight)
    
    # 获取PLY文件列表
    ply_files = glob.glob(os.path.join(args.input_dir, "*.ply"))
    ply_files.sort()
    
    if len(ply_files) == 0:
        logger.error(f"在 {args.input_dir} 中未找到PLY文件")
        return
    
    logger.info(f"找到 {len(ply_files)} 个PLY文件")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 类别名称
    class_names = cfg.data.get("names", [f"class_{i}" for i in range(cfg.data.num_classes)])
    logger.info(f"类别: {class_names}")
    
    # 收集所有结果
    true_labels = []
    pred_labels = []
    pred_scores = []  # 预测概率分数
    
    # 处理文件
    num_samples = min(args.num_samples, len(ply_files)) if args.num_samples else len(ply_files)
    logger.info(f"处理 {num_samples} 个文件...")
    
    for idx, ply_file in enumerate(ply_files[:num_samples]):
        logger.info(f"[{idx+1}/{num_samples}] 处理: {os.path.basename(ply_file)}")
        
        # 加载PLY文件
        pcd = o3d.io.read_point_cloud(ply_file)
        coord = np.asarray(pcd.points, dtype=np.float32)
        color = np.asarray(pcd.colors, dtype=np.float32)
        
        if len(coord) == 0:
            logger.warning(f"  跳过空点云: {ply_file}")
            continue
        
        logger.info(f"  点云大小: {len(coord)} 点")
        
        # 获取真实标签
        true_label = get_ground_truth_label(ply_file)
        if true_label == -1:
            logger.warning(f"  无法确定真实标签，跳过: {ply_file}")
            continue
        
        # 推理（分类模式）
        try:
            pred_class, confidence, pred_probs = inference_single_file(model, coord, color, cfg)
            class_name = class_names[pred_class] if pred_class < len(class_names) else f"class_{pred_class}"
            logger.info(f"  [{idx+1}/{num_samples}] 真实: {class_names[true_label]}, 预测: {class_name}, 置信度: {confidence*100:.2f}%")
        except Exception as e:
            logger.error(f"  推理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 收集结果
        true_labels.append(true_label)
        pred_labels.append(pred_class)
        pred_scores.append(pred_probs)  # [类别0的概率, 类别1的概率]
    
    # 转换为numpy数组
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_scores = np.array(pred_scores)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"分类评估结果")
    logger.info(f"{'='*60}")
    
    # 计算分类指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    # 每个类别的指标
    precision_per_class = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    
    logger.info(f"\n整体指标:")
    logger.info(f"  准确率 (Accuracy): {accuracy*100:.2f}%")
    logger.info(f"  精确率 (Precision): {precision*100:.2f}%")
    logger.info(f"  召回率 (Recall): {recall*100:.2f}%")
    logger.info(f"  F1分数 (F1-Score): {f1*100:.2f}%")
    
    logger.info(f"\n各类别指标:")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name} (类别 {i}):")
        logger.info(f"    精确率: {precision_per_class[i]*100:.2f}%")
        logger.info(f"    召回率: {recall_per_class[i]*100:.2f}%")
        logger.info(f"    F1分数: {f1_per_class[i]*100:.2f}%")
    
    # 打印详细分类报告
    logger.info(f"\n详细分类报告:")
    report = classification_report(true_labels, pred_labels, 
                                   target_names=class_names, 
                                   zero_division=0)
    logger.info(report)
    
    # 绘制可视化图表
    logger.info(f"\n生成可视化图表...")
    
    # 混淆矩阵
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(true_labels, pred_labels, class_names, cm_path)
    
    # ROC曲线
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    roc_auc = plot_roc_curve(true_labels, pred_scores, class_names, roc_path)
    logger.info(f"  ROC AUC: {roc_auc:.3f}")
    
    # PR曲线
    pr_path = os.path.join(args.output_dir, "pr_curve.png")
    pr_auc = plot_pr_curve(true_labels, pred_scores, class_names, pr_path)
    logger.info(f"  PR AUC: {pr_auc:.3f}")
    
    # 保存指标到文件
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("分类评估指标\n")
        f.write("="*60 + "\n\n")
        f.write(f"整体指标:\n")
        f.write(f"  准确率 (Accuracy): {accuracy*100:.2f}%\n")
        f.write(f"  精确率 (Precision): {precision*100:.2f}%\n")
        f.write(f"  召回率 (Recall): {recall*100:.2f}%\n")
        f.write(f"  F1分数 (F1-Score): {f1*100:.2f}%\n")
        f.write(f"  ROC AUC: {roc_auc:.3f}\n")
        f.write(f"  PR AUC: {pr_auc:.3f}\n\n")
        f.write(f"各类别指标:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  {class_name} (类别 {i}):\n")
            f.write(f"    精确率: {precision_per_class[i]*100:.2f}%\n")
            f.write(f"    召回率: {recall_per_class[i]*100:.2f}%\n")
            f.write(f"    F1分数: {f1_per_class[i]*100:.2f}%\n")
        f.write(f"\n详细分类报告:\n")
        f.write(report)
    
    logger.info(f"保存指标到: {metrics_path}")
    logger.info(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

