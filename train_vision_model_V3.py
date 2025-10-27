import os
import json
import torch
import yaml
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


def convert_to_yolo_format():
    """
    将数据集转换为YOLO格式，包含坡度线作为额外关键点
    """
    print("开始转换数据集为YOLO格式...")

    # 数据集路径
    data_root = "dataset/cleaning data/dateasetready"
    yolo_dataset_dir = "yolo_climbing_dataset"

    # 创建YOLO数据集目录结构
    directories = [
        f"{yolo_dataset_dir}/images/train",
        f"{yolo_dataset_dir}/images/val",
        f"{yolo_dataset_dir}/labels/train",
        f"{yolo_dataset_dir}/labels/val"
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

    # 收集所有标注文件
    annotation_files = []
    for file in os.listdir(data_root):
        if file.endswith('_annotations.json'):
            annotation_files.append(os.path.join(data_root, file))

    print(f"找到 {len(annotation_files)} 个标注文件")

    if len(annotation_files) == 0:
        print("未找到标注文件，请检查数据路径")
        return None

    # 划分训练集和验证集
    train_files, val_files = train_test_split(annotation_files, test_size=0.3, random_state=42)

    # 处理训练集
    print("处理训练集...")
    for annotation_path in tqdm(train_files):
        process_annotation_file(annotation_path, yolo_dataset_dir, "train")

    # 处理验证集
    print("处理验证集...")
    for annotation_path in tqdm(val_files):
        process_annotation_file(annotation_path, yolo_dataset_dir, "val")

    # 创建数据集配置文件
    create_dataset_yaml(yolo_dataset_dir)

    print(f"数据集转换完成！保存在: {yolo_dataset_dir}")
    return yolo_dataset_dir


def process_annotation_file(annotation_path, yolo_dataset_dir, split_type):
    """
    处理单个标注文件，转换为YOLO格式，包含坡度线作为关键点
    """
    try:
        # 读取标注文件
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # 获取图像路径并修正
        image_path = annotation['image_path']
        image_path = image_path.replace('\\', '/')
        image_name = os.path.basename(image_path)

        # 构建完整的图像路径
        json_dir = os.path.dirname(annotation_path)
        full_image_path = os.path.join(json_dir, image_name)

        if not os.path.exists(full_image_path):
            print(f"警告: 图像文件不存在: {full_image_path}")
            return

        # 读取图像获取尺寸
        image = Image.open(full_image_path)
        img_width, img_height = image.size

        # 复制图像到YOLO数据集目录
        dest_image_path = f"{yolo_dataset_dir}/images/{split_type}/{image_name}"
        shutil.copy2(full_image_path, dest_image_path)

        # 创建YOLO格式的标签文件
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_path = f"{yolo_dataset_dir}/labels/{split_type}/{label_filename}"

        # 转换为YOLO格式（包含坡度线作为关键点）
        convert_annotation_to_yolo_with_slope(annotation, label_path, img_width, img_height)

    except Exception as e:
        print(f"处理文件 {annotation_path} 时出错: {e}")


def convert_annotation_to_yolo_with_slope(annotation, label_path, img_width, img_height):
    """
    将标注转换为YOLO格式，包含坡度线作为额外关键点
    YOLO关键点格式: class x_center y_center width height kp1_x kp1_y kp2_x kp2_y ... visibility

    关键点顺序:
    0-3: 原始4个关键点
    4-5: 坡度线的两个端点
    """
    with open(label_path, 'w') as f:
        # 处理边界框
        bboxes = annotation['bboxes']
        if len(bboxes) > 0:
            bbox = bboxes[0]
            x1, y1, x2, y2 = bbox

            # 转换为YOLO格式: x_center, y_center, width, height (归一化)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # 处理原始关键点
            keypoints = annotation['keypoints']
            keypoints_data = []

            # 按标签排序关键点 (0,1,2,3)
            sorted_keypoints = sorted(keypoints, key=lambda x: x[2])

            for kp in sorted_keypoints:
                x, y, label = kp
                # 归一化坐标
                kp_x = x / img_width
                kp_y = y / img_height
                keypoints_data.extend([kp_x, kp_y, 2])  # 2表示关键点可见

            # 处理坡度线作为额外关键点
            lines = annotation['lines']
            if len(lines) > 0:
                line = lines[0]  # 只有一条线
                x1_line, y1_line = line[0]  # 坡度线起点
                x2_line, y2_line = line[1]  # 坡度线终点

                # 归一化坐标
                kp4_x = x1_line / img_width
                kp4_y = y1_line / img_height
                kp5_x = x2_line / img_width
                kp5_y = y2_line / img_height

                # 添加坡度线关键点
                keypoints_data.extend([kp4_x, kp4_y, 2])  # 坡度线起点
                keypoints_data.extend([kp5_x, kp5_y, 2])  # 坡度线终点
            else:
                # 如果没有坡度线，用0填充
                keypoints_data.extend([0, 0, 0, 0, 0, 0])

            # 如果关键点不足6个，用0填充 (4个原始关键点 + 2个坡度线关键点)
            while len(keypoints_data) < 18:  # 6个关键点 * 3个值(x,y,visibility)
                keypoints_data.extend([0, 0, 0])

            # 写入YOLO格式: class bbox keypoints
            line_data = [0, x_center, y_center, width, height] + keypoints_data
            line_str = ' '.join(f'{x:.6f}' for x in line_data)
            f.write(line_str + '\n')


def create_dataset_yaml(dataset_dir):
    """
    创建YOLO数据集配置文件，包含6个关键点
    """
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': '',

        'nc': 1,  # 类别数量
        'names': ['climber'],  # 类别名称

        'kpt_shape': [6, 3],  # 关键点格式: [6个关键点, 每个点3个值(x,y,visibility)]

        # 关键点连接（用于可视化）
        'skeleton': [
            [0, 1],  # 原始关键点连接
            [1, 2],
            [2, 3],
            [4, 5]  # 坡度线连接
        ]
    }

    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"数据集配置文件已创建: {yaml_path}")
    return yaml_path


def train_yolo_model():
    """
    使用YOLOv11训练关键点检测模型，包含坡度线检测
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("正在安装ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO

    # 首先转换数据集
    dataset_dir = convert_to_yolo_format()
    if dataset_dir is None:
        return None, None

    # 数据集配置文件路径
    data_yaml = os.path.join(dataset_dir, 'dataset.yaml')

    print("开始YOLO模型训练...")

    # 加载YOLOv11n-pose模型（关键点检测）
    model = YOLO('YOLO/yolo11n-pose.pt')  # 使用预训练的关键点检测模型

    # 训练参数配置
    train_args = {
        'data': data_yaml,
        'epochs': 150,
        'imgsz': 640,
        'batch': 16,
        'workers': 8,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'patience': 20,
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,  # 初始学习率
        'lrf': 0.000001,  # 最终学习率
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # 边界框损失权重
        'cls': 0.5,  # 分类损失权重
        'dfl': 1.5,  # 分布焦点损失权重
        'pose': 12.0,  # 姿态损失权重（关键点检测）
        'kobj': 1.0,  # 关键点对象性损失权重
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'save_period': -1,
        'seed': 42,
        'deterministic': True,
        'single_cls': True,  # 单类别训练
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # 自动混合精度
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'project': 'yolo_training',
        'name': 'climbing_pose_detection_with_slope',
        'verbose': True,
        'plots': True
    }

    # 开始训练
    print("开始训练YOLOv11关键点检测模型（包含坡度线）...")
    results = model.train(**train_args)

    # 修复：正确获取最佳模型路径
    best_model_path = "yolo_training/climbing_pose_detection_with_slope/weights/best.pt"

    # 检查模型文件是否存在
    if not os.path.exists(best_model_path):
        print(f"警告: 最佳模型文件不存在: {best_model_path}")
        # 尝试查找其他可能的模型文件
        possible_paths = [
            "yolo_training/climbing_pose_detection_with_slope/weights/last.pt",
            "runs/pose/climbing_pose_detection_with_slope/weights/best.pt",
            "runs/pose/climbing_pose_detection_with_slope/weights/last.pt"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                best_model_path = path
                print(f"使用替代模型路径: {best_model_path}")
                break

    print(f"训练完成！最佳模型保存在: {best_model_path}")

    # 在验证集上评估模型
    print("在验证集上评估模型...")
    try:
        # 重新加载最佳模型进行评估
        best_model = YOLO(best_model_path)
        val_results = best_model.val()
    except Exception as e:
        print(f"验证过程中出错: {e}")
        val_results = None

    # 绘制训练结果
    plot_training_results(results, "yolo_training/climbing_pose_detection_with_slope")

    return best_model_path, results


def plot_training_results(results, save_dir):
    """
    绘制训练结果图表
    """
    try:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 创建结果图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 从results对象中提取指标
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            epochs = list(range(1, len(metrics.get('train/box_loss', [])) + 1))

            # 边界框损失
            if 'train/box_loss' in metrics and 'val/box_loss' in metrics:
                axes[0, 0].plot(epochs, metrics['train/box_loss'], label='Train Box Loss', linewidth=2)
                axes[0, 0].plot(epochs, metrics['val/box_loss'], label='Val Box Loss', linewidth=2)
                axes[0, 0].set_title('Bounding Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # 关键点损失
            if 'train/pose_loss' in metrics and 'val/pose_loss' in metrics:
                axes[0, 1].plot(epochs, metrics['train/pose_loss'], label='Train Pose Loss', linewidth=2)
                axes[0, 1].plot(epochs, metrics['val/pose_loss'], label='Val Pose Loss', linewidth=2)
                axes[0, 1].set_title('Pose (Keypoints) Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # 精确率
            if 'metrics/precision(B)' in metrics:
                axes[1, 0].plot(epochs, metrics['metrics/precision(B)'], label='Precision', linewidth=2, color='green')
                axes[1, 0].set_title('Precision')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # 召回率
            if 'metrics/recall(B)' in metrics:
                axes[1, 1].plot(epochs, metrics['metrics/recall(B)'], label='Recall', linewidth=2, color='red')
                axes[1, 1].set_title('Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Recall')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练结果图表已保存: {plot_path}")
        plt.close()

    except Exception as e:
        print(f"绘制训练结果时出错: {e}")


if __name__ == "__main__":
    print("开始YOLO格式数据集转换和模型训练（包含坡度线）...")

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 训练YOLO模型
    best_model_path, results = train_yolo_model()

    if best_model_path:
        print(f"\n训练完成！最佳模型路径: {best_model_path}")

        # 检查模型文件
        if os.path.exists(best_model_path):
            print("模型文件存在，可以用于推理。")
        else:
            print("警告: 模型文件不存在，请检查训练过程。")
    else:
        print("\n训练失败，请检查错误信息。")