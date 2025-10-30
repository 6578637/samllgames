# RK3568手机屏幕检测系统

一个完整的AI手机屏幕检测与自动化控制系统，从数据标注到RK3568开发板部署的全流程解决方案。

## 项目概述

本项目实现了基于YOLO关键点检测的手机屏幕检测系统，能够自动识别车辆关键点并计算坡度角度，通过ADB实现自动化点击控制。系统包含完整的数据集制作、模型训练、模型转换和边缘部署流程。

## 核心功能

- **多关键点检测**: 检测6个关键点（4个车辆关键点 + 2个坡度线关键点）
- **角度计算**: 基于关键点位置计算坡度角度
- **自动化控制**: 通过ADB实现精准的屏幕点击操作
- **完整流程**: 从数据标注到边缘部署的端到端解决方案

## 项目结构

```
rkcc-car/
├── dataset/                          # 数据集目录
│   ├── annotated_images/             # 标注后的图像
│   └── cleaning data/                # 原始数据
├── image_annotator.py                # GUI图像标注工具
├── train_vision_model_V3.py          # 模型训练程序
├── convert_to_model.py               # 模型转换程序
├── rk3568_phone_detector_rknn.py     # RK3568部署主程序
├── predict_video_gui.py              # 本地测试GUI程序
├── video_processor.py                # 视频处理程序
├── run_phone_detector_rk3568.sh      # 部署脚本
├── deployment_guide.md               # 详细部署指南
├── requirements.txt                  # Python依赖
├── args.yaml                         # 配置文件
├── best.pt                           # PyTorch模型文件
└── best.rknn                         # RKNN模型文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
pip install -r requirements.txt

# 对于RK3568开发板
pip3 install -r requirements.txt
```

### 2. 数据标注

```bash
python image_annotator.py
```

使用GUI工具标注图像，支持：
- 边界框标注
- 关键点标注（前轮、后轮、车身、车顶）
- 坡度线标注

### 3. 模型训练

```bash
python train_vision_model_V3.py
```

训练YOLOv11关键点检测模型，输出best.pt文件。

### 4. 模型转换

```bash
python convert_to_model.py
```

将PyTorch模型转换为RKNN格式，适配RK3568 NPU。

### 5. 本地测试

```bash
python predict_video_gui.py
```

使用GUI界面测试模型效果。

### 6. RK3568部署

```bash
# 直接运行
./run_phone_detector_rk3568.sh run

# 安装为系统服务
sudo ./run_phone_detector_rk3568.sh install
sudo ./run_phone_detector_rk3568.sh start
```

## 技术细节

### 关键点检测

系统检测6个关键点：
- 0-3: 原始关键点（前轮、后轮、车身、车顶）
- 4-5: 坡度线的两个端点

### 角度计算

```
夹角 = -15度 + 前后轮中心和坡度的夹角
```

### 点击控制

- **油门位置**: (2417, 982) - 点击时长1秒
- **刹车位置**: (369, 991) - 点击时长0.5秒

### 状态机

- **detecting**: 检测状态
- **tapping**: 点击状态  
- **waiting**: 等待状态

## 硬件要求

- **开发板**: RK3568开发板
- **手机**: 支持ADB调试的安卓设备
- **系统**: Linux (Ubuntu/Debian)

## 依赖说明

### Python依赖
- rknn-toolkit-lite>=1.7.5
- opencv-python>=4.5.0
- numpy>=1.19.0
- ultralytics (用于训练)
- torch (用于训练)

### 系统依赖
- ADB (Android Debug Bridge)
- Python 3.6+

## 部署指南

详细部署步骤请参考 [deployment_guide.md](deployment_guide.md)

## 故障排除

常见问题及解决方案：

1. **ADB设备未连接**
   - 检查USB连接
   - 在手机上启用USB调试
   - 运行 `adb devices` 确认设备

2. **模型加载失败**
   - 确认模型文件存在
   - 检查文件权限

3. **依赖包缺失**
   - 重新安装依赖包

## 性能优化

- 设置CPU性能模式
- 关闭不必要的系统服务
- 优化网络连接


## 贡献

欢迎提交Issue和Pull Request来改进项目。


---

**注意**: 本项目仅供学习和研究使用，请遵守相关法律法规。
