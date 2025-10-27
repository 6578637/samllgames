import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch.nn as nn
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import math
import subprocess

from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights



# # 定义模型结构（必须与训练时相同）
# class MultiTaskVisionModel(nn.Module):
#     def __init__(self, num_keypoints=4):
#         super(MultiTaskVisionModel, self).__init__()
        
#         # 使用预训练的ResNet18作为backbone
#         self.backbone = resnet18(weights=None)  # 不加载预训练权重，因为我们会加载训练好的权重
        
#         # 移除最后的全连接层
#         self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
#         # 全局平均池化
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 获取backbone的输出通道数（ResNet18是512）
#         backbone_out_channels = 512
        
#         # 边界框回归头
#         self.bbox_head = nn.Sequential(
#             nn.Linear(backbone_out_channels, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 4),  # [x1, y1, x2, y2]
#             nn.Sigmoid()  # 输出在 [0,1] 范围内
#         )
        
#         # 关键点检测头
#         self.keypoints_head = nn.Sequential(
#             nn.Linear(backbone_out_channels, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, num_keypoints * 2),  # 每个关键点有x,y坐标
#             nn.Sigmoid()  # 输出在 [0,1] 范围内
#         )
        
#         # 坡度线检测头
#         self.line_head = nn.Sequential(
#             nn.Linear(backbone_out_channels, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 4),  # [x1, y1, x2, y2]
#             nn.Sigmoid()  # 输出在 [0,1] 范围内
#         )
    
#     def forward(self, x):
#         # Backbone特征提取
#         features = self.backbone(x)
        
#         # 全局平均池化
#         pooled = self.global_avg_pool(features)
#         pooled = pooled.view(pooled.size(0), -1)
        
#         # 多任务输出
#         bbox_output = self.bbox_head(pooled)
#         keypoints_output = self.keypoints_head(pooled)
#         line_output = self.line_head(pooled)
        
#         return bbox_output, keypoints_output, line_output


# 使用残差连接 + 层
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return x + self.block(x)


class MultiTaskVisionModelMobileNet(nn.Module):
    def __init__(self, num_keypoints=4, use_large=True):
        super(MultiTaskVisionModelMobileNet, self).__init__()

        # 使用预训练的MobileNet作为backbone
        if use_large:
            self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            backbone_out_channels = 960  # MobileNetV3-Large最后的特征维度
        else:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            backbone_out_channels = 576  # MobileNetV3-Small最后的特征维度

        # 移除分类头，保留特征提取部分
        self.backbone.classifier = nn.Identity()

        # 全局平均池化（MobileNetV3已经有自适应平均池化，但我们确保一下）
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 边界框回归头
        self.bbox_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

        # 关键点检测头
        self.keypoints_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, num_keypoints * 2),
            nn.Sigmoid()
        )

        # 坡度线检测头
        self.line_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

        # 初始化任务头的权重
        self._initialize_heads()

    def _initialize_heads(self):
        """初始化任务头的权重"""
        for head in [self.bbox_head, self.keypoints_head, self.line_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone特征提取
        features = self.backbone(x)

        # 如果特征还是4D的（B, C, H, W），则进行全局平均池化
        if features.dim() == 4:
            features = self.global_avg_pool(features)

        # 展平特征
        pooled = features.view(features.size(0), -1)

        # 多任务输出
        bbox_output = self.bbox_head(pooled)
        keypoints_output = self.keypoints_head(pooled)
        line_output = self.line_head(pooled)

        return bbox_output, keypoints_output, line_output
    
def load_model(model_path, device):
    """加载训练好的模型"""
    # model = MultiTaskVisionModel(num_keypoints=4)
    model = MultiTaskVisionModelMobileNet(num_keypoints=4)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print(f"模型已从 {model_path} 加载")
    print(f"训练时的最佳验证损失: {checkpoint['val_loss']:.4f}")
    print(f"训练时的最佳验证IoU: {checkpoint['val_iou']:.4f}")
    
    return model

def preprocess_frame(frame, transform):
    """预处理视频帧"""
    # 将BGR转换为RGB用于模型处理
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(frame_rgb)
    
    # 应用变换
    processed_image = transform(pil_image)
    
    # 返回原始BGR帧用于显示，RGB帧用于模型处理
    return processed_image, frame

def denormalize_coordinates(coords, width, height):
    """将归一化坐标转换回像素坐标"""
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    
    coords = coords.copy()
    coords[0] *= width  # x1
    coords[1] *= height  # y1
    coords[2] *= width   # x2
    coords[3] *= height  # y2
    
    return coords.astype(int)

def denormalize_keypoints(keypoints, width, height):
    """将归一化关键点转换回像素坐标"""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    
    keypoints = keypoints.copy()
    # 将关键点reshape为(N, 2)格式
    keypoints = keypoints.reshape(-1, 2)
    keypoints[:, 0] *= width   # x坐标
    keypoints[:, 1] *= height  # y坐标
    
    return keypoints.astype(int)

def calculate_angle(keypoints, line, width, height):
    """计算夹角：夹角 = -40度 + 前后轮中心和坡度的夹角"""
    if keypoints is None or line is None:
        return 0
    
    # 假设关键点包含前后轮中心
    keypoints_pixels = denormalize_keypoints(keypoints, width, height)
    
    if len(keypoints_pixels) < 2:
        return 0
    
    # 前后轮中心连线
    front_wheel = keypoints_pixels[0]  # 前轮
    rear_wheel = keypoints_pixels[1]   # 后轮
    
    # 坡度线
    line_pixels = denormalize_coordinates(line, width, height)
    
    # 计算前后轮连线与坡度线的夹角
    wheel_vector = (front_wheel[0] - rear_wheel[0], front_wheel[1] - rear_wheel[1])
    slope_vector = (line_pixels[2] - line_pixels[0], line_pixels[3] - line_pixels[1])
    
    # 计算向量夹角（弧度）
    dot_product = wheel_vector[0] * slope_vector[0] + wheel_vector[1] * slope_vector[1]
    wheel_magnitude = math.sqrt(wheel_vector[0]**2 + wheel_vector[1]**2)
    slope_magnitude = math.sqrt(slope_vector[0]**2 + slope_vector[1]**2)
    
    if wheel_magnitude == 0 or slope_magnitude == 0:
        return 0
    
    cos_angle = dot_product / (wheel_magnitude * slope_magnitude)
    cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    # 最终夹角计算：-45度 + 检测到的夹角
    final_angle = -45 + angle_deg
    
    return final_angle

class PhoneScreenDetector:
    """手机屏幕检测器"""
    def __init__(self):
        self.adb_path = self.find_adb()
        
    def find_adb(self):
        """查找ADB可执行文件路径"""
        possible_paths = [
            "adb",  # 如果在PATH中
            "platform-tools/adb",  # Android SDK路径
            "C:/Program Files (x86)/Android/android-sdk/platform-tools/adb.exe",
            "C:/Android/platform-tools/adb.exe",
            "~/AppData/Local/Android/Sdk/platform-tools/adb.exe"
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "version"], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"找到ADB: {path}")
                    return path
            except:
                continue
        
        print("警告: 未找到ADB，请确保ADB已安装并在PATH中")
        return "adb"
    
    def check_adb_connection(self):
        """检查ADB连接状态"""
        try:
            result = subprocess.run([self.adb_path, "devices"], capture_output=True, text=True)
            devices = [line for line in result.stdout.split('\n') if '\tdevice' in line]
            if devices:
                print(f"找到 {len(devices)} 个设备")
                return True
            else:
                print("未找到连接的设备")
                return False
        except Exception as e:
            print(f"ADB连接检查失败: {e}")
            return False
    
    def capture_phone_screen(self):
        """捕获手机屏幕"""
        try:
            # 使用ADB截图
            result = subprocess.run([self.adb_path, "exec-out", "screencap", "-p"], 
                                  capture_output=True)
            
            if result.returncode == 0:
                # 将截图数据转换为numpy数组
                nparr = np.frombuffer(result.stdout, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # 确保图像不模糊 - 使用高质量编码
                    return frame
                else:
                    print("截图解码失败")
                    return None
            else:
                print("截图失败")
                return None
                
        except Exception as e:
            print(f"截图失败: {e}")
            return None
    
    def push_to_phone(self, frame, filename="detection_result.png"):
        """将处理后的图像推送到手机"""
        try:
            # 保存处理后的图像（高质量，不模糊）
            cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # 推送到手机
            result = subprocess.run([self.adb_path, "push", filename, "/sdcard/"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"检测结果已推送到手机: /sdcard/{filename}")
                return True
            else:
                print(f"推送失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"推送失败: {e}")
            return False

class RealTimeVisualizer:
    """实时可视化类，使用OpenCV进行显示以避免Matplotlib线程问题"""
    def __init__(self, window_name="Real-time Climbing Analysis"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # 定义点击区域位置（模拟手机屏幕）
        self.screen_width = 1080  # 常见手机横向宽度
        self.screen_height = 1920  # 常见手机横向高度
        # 点击区域大小 - 扩大覆盖范围到四分之一个屏幕
        self.click_region_width = int(self.screen_width * 0.25)  # 四分之一屏幕宽度
        self.click_region_height = int(self.screen_height * 0.25)  # 四分之一屏幕高度
        # 使用用户指定的精确坐标
        self.brake_position = (369, 991)   # 左下角刹车区域
        self.gas_position = (2417, 982)     # 右下角油门区域
        
        # 点击控制相关变量
        self.current_action = None  # 当前正在执行的动作 ('gas' 或 'brake')
        self.current_tap_process = None  # 当前点击进程
        
        # 动态点击时长配置
        self.max_brake_duration = 1000  # 最大刹车持续时间（1秒）
        self.max_gas_duration = 2000    # 最大油门持续时间（2秒）
        self.angle_offset = -60         # 角度偏移量
        
    def check_adb_connection(self):
        """检查ADB连接状态"""
        try:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
            devices = [line for line in result.stdout.split('\n') if '\tdevice' in line]
            if devices:
                print(f"找到 {len(devices)} 个设备")
                return True
            else:
                print("未找到连接的设备")
                return False
        except Exception as e:
            print(f"ADB连接检查失败: {e}")
            return False
    
    def tap_screen(self, x, y, duration=500):
        """模拟屏幕点击（长按）"""
        try:
            # 使用ADB模拟长按
            command = f"adb shell input swipe {x} {y} {x} {y} {duration}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                print(f"点击失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"点击失败: {e}")
            return False
    
    def start_continuous_tap(self, x, y):
        """开始持续点击直到下次检测"""
        try:
            # 停止之前的点击进程
            if self.current_tap_process:
                self.stop_continuous_tap()
            
            # 使用长时间的长按来实现持续点击效果
            command = f"adb shell input swipe {x} {y} {x} {y} 10000"  # 10秒长按
            self.current_tap_process = subprocess.Popen(command, shell=True)
            return True
        except Exception as e:
            print(f"开始持续点击失败: {e}")
            return False
    
    def stop_continuous_tap(self):
        """停止持续点击"""
        try:
            if self.current_tap_process:
                self.current_tap_process.terminate()
                self.current_tap_process.wait()
                self.current_tap_process = None
                return True
        except Exception as e:
            print(f"停止持续点击失败: {e}")
            return False
    
    def calculate_dynamic_duration(self, final_angle):
        """根据final_angle计算动态点击时长"""
        if final_angle < 0:
            # 负角度：油门控制
            # 角度范围：从0到(angle_offset)度，对应时长从0到max_gas_duration
            max_negative_angle = self.angle_offset  # 最大负角度
            if final_angle <= max_negative_angle:
                return self.max_gas_duration
            else:
                # 线性映射：final_angle从0到max_negative_angle，时长从0到max_gas_duration
                ratio = abs(final_angle) / abs(max_negative_angle)
                return int(ratio * self.max_gas_duration)
        elif final_angle > 0:
            # 正角度：刹车控制
            # 角度范围：从0到(angle_offset+120)度，对应时长从0到max_brake_duration
            max_positive_angle = self.angle_offset + 120  # 最大正角度
            if final_angle >= max_positive_angle:
                return self.max_brake_duration
            else:
                # 线性映射：final_angle从0到max_positive_angle，时长从0到max_brake_duration
                ratio = final_angle / max_positive_angle
                return int(ratio * self.max_brake_duration)
        else:
            # final_angle为0，不点击
            return 0
    
    def update_display(self, frame, bbox, keypoints, line, fps=None):
        """更新显示内容"""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # 计算夹角
        angle = calculate_angle(keypoints, line, width, height)
        
        # 绘制边界框
        if bbox is not None and torch.any(bbox > 0):  # 如果不是全零
            bbox_pixels = denormalize_coordinates(bbox, width, height)
            x1, y1, x2, y2 = bbox_pixels
            
            # 绘制矩形框
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display_frame, 'BBox', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 绘制关键点
        if keypoints is not None and torch.any(keypoints > 0):  # 如果不是全零
            keypoints_pixels = denormalize_keypoints(keypoints, width, height)
            
            # 定义四个关键点的不同颜色 (BGR格式)
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 165, 255)]  # 红、蓝、绿、橙
            
            for i, (x, y) in enumerate(keypoints_pixels):
                if x > 0 or y > 0:  # 忽略填充的零
                    # 绘制更大的点 (半径8像素)
                    cv2.circle(display_frame, (x, y), 8, colors[i], -1)
                    cv2.putText(display_frame, f'KP{i+1}', (x+10, y+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        
        # 绘制线
        if line is not None and torch.any(line > 0):  # 如果不是全零
            line_pixels = denormalize_coordinates(line, width, height)
            x1, y1, x2, y2 = line_pixels
            
            # 绘制线
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # 在线段两端添加端点标记
            cv2.rectangle(display_frame, (x1-3, y1-3), (x1+3, y1+3), (0, 255, 0), -1)
            cv2.rectangle(display_frame, (x2-3, y2-3), (x2+3, y2+3), (0, 255, 0), -1)
            cv2.putText(display_frame, 'Slope Line', (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制黄色边界框显示点击区域
        self._draw_click_regions(display_frame, angle)
        
        # 添加角度信息
        angle_text = f"夹角: {angle:.1f}°"
        cv2.putText(display_frame, angle_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加操作建议
        action_text = "建议操作: " + ("油门" if angle < 0 else "刹车")
        cv2.putText(display_frame, action_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 添加FPS信息
        if fps is not None:
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加标题
        cv2.putText(display_frame, 'Real-time Climbing Analysis', (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示图像
        cv2.imshow(self.window_name, display_frame)
    
    def _draw_click_regions(self, frame, angle):
        """绘制点击区域 - 只显示推荐区域"""
        height, width = frame.shape[:2]
        
        # 调整点击区域位置以适应当前帧尺寸
        scale_x = width / self.screen_width
        scale_y = height / self.screen_height
        
        brake_x = int(self.brake_position[0] * scale_x)
        brake_y = int(self.brake_position[1] * scale_y)
        gas_x = int(self.gas_position[0] * scale_x)
        gas_y = int(self.gas_position[1] * scale_y)
        
        # 使用动态计算的点击区域大小
        rect_width = int(self.click_region_width * scale_x)
        rect_height = int(self.click_region_height * scale_y)
        
        # 根据角度只绘制推荐点击区域
        if angle < 0:
            # 夹角为负数：推荐油门区域（高亮显示）
            gas_x1 = gas_x - rect_width // 2
            gas_y1 = gas_y - rect_height // 2
            gas_x2 = gas_x + rect_width // 2
            gas_y2 = gas_y + rect_height // 2
            cv2.rectangle(frame, (gas_x1, gas_y1), (gas_x2, gas_y2), (0, 255, 255), 5)  # 黄色高亮
            cv2.putText(frame, "GAS", (gas_x1, gas_y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "RECOMMEND", (gas_x1, gas_y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # 夹角为正数：推荐刹车区域（高亮显示）
            brake_x1 = brake_x - rect_width // 2
            brake_y1 = brake_y - rect_height // 2
            brake_x2 = brake_x + rect_width // 2
            brake_y2 = brake_y + rect_height // 2
            cv2.rectangle(frame, (brake_x1, brake_y1), (brake_x2, brake_y2), (0, 255, 255), 5)  # 黄色高亮
            cv2.putText(frame, "BRAKE", (brake_x1, brake_y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "RECOMMEND", (brake_x1, brake_y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def close(self):
        """关闭窗口"""
        cv2.destroyWindow(self.window_name)

class VideoInferenceGUI:
    """视频推理图形界面"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("攀岩视觉控制模型 - 推理界面")
        self.root.geometry("500x400")
        
        # 模型路径
        self.model_path = "best_model(1).pth"
        
        # 推理线程控制
        self.inference_thread = None
        self.stop_inference = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="攀岩视觉控制模型推理系统", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 推理模式选择
        mode_label = ttk.Label(main_frame, text="选择推理模式:", font=("Arial", 12))
        mode_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # 模式选择按钮框架
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # 实时摄像头推理按钮
        self.camera_btn = ttk.Button(mode_frame, text="实时摄像头推理", 
                                   command=self.start_camera_inference)
        self.camera_btn.grid(row=0, column=0, padx=(0, 10))
        
        # 视频文件推理按钮
        self.video_btn = ttk.Button(mode_frame, text="视频文件推理", 
                                  command=self.start_video_inference)
        self.video_btn.grid(row=0, column=1, padx=(0, 10))
        
        # 图像推理按钮
        self.image_btn = ttk.Button(mode_frame, text="单张图像推理", 
                                  command=self.start_image_inference)
        self.image_btn.grid(row=0, column=2, padx=(0, 10))
        
        # 手机屏幕检测按钮
        self.phone_btn = ttk.Button(mode_frame, text="手机屏幕检测", 
                                  command=self.start_phone_screen_detection)
        self.phone_btn.grid(row=0, column=3)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(main_frame, text="推理参数设置", padding="10")
        param_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # 检测间隔设置
        interval_label = ttk.Label(param_frame, text="初始检测间隔 (秒):")
        interval_label.grid(row=0, column=0, sticky=tk.W)
        
        self.interval_var = tk.StringVar(value="0.2")
        interval_entry = ttk.Entry(param_frame, textvariable=self.interval_var, width=10)
        interval_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 摄像头ID设置
        camera_label = ttk.Label(param_frame, text="摄像头ID:")
        camera_label.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        self.camera_var = tk.StringVar(value="0")
        camera_entry = ttk.Entry(param_frame, textvariable=self.camera_var, width=10)
        camera_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # 状态显示
        status_frame = ttk.LabelFrame(main_frame, text="状态信息", padding="10")
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.status_text = tk.Text(status_frame, height=8, width=50)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # 控制按钮
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))
        
        self.stop_btn = ttk.Button(control_frame, text="停止推理", 
                                 command=self.stop_inference_process, state="disabled")
        self.stop_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.clear_btn = ttk.Button(control_frame, text="清空日志", 
                                  command=self.clear_log)
        self.clear_btn.grid(row=0, column=1)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
    def log_message(self, message):
        """在状态文本框中添加消息"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def clear_log(self):
        """清空日志"""
        self.status_text.delete(1.0, tk.END)
        
    def start_camera_inference(self):
        """开始摄像头推理"""
        try:
            camera_id = int(self.camera_var.get())
            detection_interval = float(self.interval_var.get())
            
            self.log_message(f"开始摄像头推理...")
            self.log_message(f"摄像头ID: {camera_id}")
            self.log_message(f"初始检测间隔: {detection_interval}秒")
            
            # 禁用按钮，启用停止按钮
            self.camera_btn.config(state="disabled")
            self.video_btn.config(state="disabled")
            self.image_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # 在新线程中运行推理
            self.stop_inference = False
            self.inference_thread = threading.Thread(
                target=self.run_camera_inference,
                args=(camera_id, detection_interval)
            )
            self.inference_thread.daemon = True
            self.inference_thread.start()
            
        except ValueError as e:
            messagebox.showerror("输入错误", "请输入有效的数字")
            
    def start_video_inference(self):
        """开始视频文件推理"""
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")]
        )
        
        if not video_path:
            return
            
        try:
            detection_interval = float(self.interval_var.get())
            
            self.log_message(f"开始视频文件推理...")
            self.log_message(f"视频文件: {video_path}")
            self.log_message(f"初始检测间隔: {detection_interval}秒")
            
            # 禁用按钮，启用停止按钮
            self.camera_btn.config(state="disabled")
            self.video_btn.config(state="disabled")
            self.image_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # 在新线程中运行推理
            self.stop_inference = False
            self.inference_thread = threading.Thread(
                target=self.run_video_inference,
                args=(video_path, detection_interval)
            )
            self.inference_thread.daemon = True
            self.inference_thread.start()
            
        except ValueError as e:
            messagebox.showerror("输入错误", "请输入有效的数字")
            
    def start_image_inference(self):
        """开始图像推理"""
        image_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        
        if not image_path:
            return
            
        self.log_message(f"开始图像推理...")
        self.log_message(f"图像文件: {image_path}")
        
        # 在新线程中运行推理
        self.inference_thread = threading.Thread(
            target=self.run_image_inference,
            args=(image_path,)
        )
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
    def start_phone_screen_detection(self):
        """开始手机屏幕检测"""
        try:
            detection_interval = float(self.interval_var.get())
            
            self.log_message(f"开始手机屏幕检测...")
            self.log_message(f"初始检测间隔: {detection_interval}秒")
            
            # 禁用按钮，启用停止按钮
            self.camera_btn.config(state="disabled")
            self.video_btn.config(state="disabled")
            self.image_btn.config(state="disabled")
            self.phone_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # 在新线程中运行手机屏幕检测
            self.stop_inference = False
            self.inference_thread = threading.Thread(
                target=self.run_phone_screen_detection,
                args=(detection_interval,)
            )
            self.inference_thread.daemon = True
            self.inference_thread.start()
            
        except ValueError as e:
            messagebox.showerror("输入错误", "请输入有效的数字")
        
    def stop_inference_process(self):
        """停止推理过程"""
        self.stop_inference = True
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        
        # 重新启用按钮
        self.camera_btn.config(state="normal")
        self.video_btn.config(state="normal")
        self.image_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        self.log_message("推理已停止")
        
    def run_camera_inference(self, camera_id, detection_interval):
        """运行摄像头推理"""
        try:
            # 设置环境变量避免OpenMP冲突
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 设备设置
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message(f"使用设备: {device}")
            
            # 加载模型
            if not os.path.exists(self.model_path):
                self.log_message(f"错误: 模型文件 {self.model_path} 不存在")
                return
                
            model = load_model(self.model_path, device)
            self.log_message("模型加载成功")
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # 初始化可视化器
            visualizer = RealTimeVisualizer()
            
            # 打开摄像头
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                self.log_message(f"错误: 无法打开摄像头 {camera_id}")
                return
                
            self.log_message(f"摄像头 {camera_id} 已打开")
            
            # 动态间隔检测变量
            current_interval = detection_interval
            last_detection_time = 0
            processing_times = []
            last_detection_result = None
            
            # FPS计算
            fps_start_time = time.time()
            frame_count = 0
            
            while not self.stop_inference:
                ret, frame = cap.read()
                if not ret:
                    self.log_message("无法读取摄像头帧")
                    break
                    
                current_time = time.time()
                frame_count += 1
                
                # 动态间隔检测逻辑
                should_detect = False
                if current_time - last_detection_time >= current_interval:
                    should_detect = True
                    detection_start_time = time.time()
                
                if should_detect:
                    # 进行检测
                    with torch.no_grad():
                        processed_frame, frame_rgb = preprocess_frame(frame, transform)
                        processed_frame = processed_frame.unsqueeze(0).to(device)
                        
                        bbox_output, keypoints_output, line_output = model(processed_frame)
                        
                        # 记录处理时间
                        processing_time = time.time() - detection_start_time
                        processing_times.append(processing_time)
                        
                        # 动态调整检测间隔
                        if len(processing_times) >= 3:
                            avg_processing_time = sum(processing_times) / len(processing_times)
                            if avg_processing_time > current_interval * 0.8:
                                current_interval = min(current_interval * 1.2, 1.0)
                                self.log_message(f"检测速度较慢，调整间隔为: {current_interval:.2f}秒")
                            elif avg_processing_time < current_interval * 0.3:
                                current_interval = max(current_interval * 0.8, 0.05)
                                self.log_message(f"检测速度较快，调整间隔为: {current_interval:.2f}秒")
                        
                        last_detection_time = current_time
                        last_detection_result = (bbox_output[0], keypoints_output[0], line_output[0])
                
                # 计算FPS
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    frame_count = 0
                else:
                    fps = None
                
                # 使用最新的检测结果更新显示
                if last_detection_result:
                    bbox, keypoints, line = last_detection_result
                    visualizer.update_display(frame_rgb, bbox, keypoints, line, fps)
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 清理资源
            cap.release()
            visualizer.close()
            cv2.destroyAllWindows()
            self.log_message("摄像头推理完成")
            
        except Exception as e:
            self.log_message(f"推理过程中发生错误: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
        
        finally:
            # 重新启用按钮
            self.camera_btn.config(state="normal")
            self.video_btn.config(state="normal")
            self.image_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
    def run_video_inference(self, video_path, detection_interval):
        """运行视频文件推理"""
        try:
            # 设置环境变量避免OpenMP冲突
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 设备设置
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message(f"使用设备: {device}")
            
            # 加载模型
            if not os.path.exists(self.model_path):
                self.log_message(f"错误: 模型文件 {self.model_path} 不存在")
                return
                
            model = load_model(self.model_path, device)
            self.log_message("模型加载成功")
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # 初始化可视化器
            visualizer = RealTimeVisualizer()
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log_message(f"错误: 无法打开视频文件 {video_path}")
                return
                
            self.log_message(f"视频文件已打开: {video_path}")
            
            # 动态间隔检测变量
            current_interval = detection_interval
            last_detection_time = 0
            processing_times = []
            last_detection_result = None
            
            # FPS计算
            fps_start_time = time.time()
            frame_count = 0
            
            while not self.stop_inference:
                ret, frame = cap.read()
                if not ret:
                    self.log_message("视频播放完成")
                    break
                    
                current_time = time.time()
                frame_count += 1
                
                # 动态间隔检测逻辑
                should_detect = False
                if current_time - last_detection_time >= current_interval:
                    should_detect = True
                    detection_start_time = time.time()
                
                if should_detect:
                    # 进行检测
                    with torch.no_grad():
                        processed_frame, frame_rgb = preprocess_frame(frame, transform)
                        processed_frame = processed_frame.unsqueeze(0).to(device)
                        
                        bbox_output, keypoints_output, line_output = model(processed_frame)
                        
                        # 记录处理时间
                        processing_time = time.time() - detection_start_time
                        processing_times.append(processing_time)
                        
                        # 动态调整检测间隔
                        if len(processing_times) >= 3:
                            avg_processing_time = sum(processing_times) / len(processing_times)
                            if avg_processing_time > current_interval * 0.8:
                                current_interval = min(current_interval * 1.2, 1.0)
                                self.log_message(f"检测速度较慢，调整间隔为: {current_interval:.2f}秒")
                            elif avg_processing_time < current_interval * 0.3:
                                current_interval = max(current_interval * 0.8, 0.05)
                                self.log_message(f"检测速度较快，调整间隔为: {current_interval:.2f}秒")
                        
                        last_detection_time = current_time
                        last_detection_result = (bbox_output[0], keypoints_output[0], line_output[0])
                
                # 计算FPS
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    frame_count = 0
                else:
                    fps = None
                
                # 使用最新的检测结果更新显示
                if last_detection_result:
                    bbox, keypoints, line = last_detection_result
                    visualizer.update_display(frame_rgb, bbox, keypoints, line, fps)
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 清理资源
            cap.release()
            visualizer.close()
            cv2.destroyAllWindows()
            self.log_message("视频推理完成")
            
        except Exception as e:
            self.log_message(f"推理过程中发生错误: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
        
        finally:
            # 重新启用按钮
            self.camera_btn.config(state="normal")
            self.video_btn.config(state="normal")
            self.image_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
    def run_phone_screen_detection(self, detection_interval):
        """运行手机屏幕检测 - 集成点击控制功能"""
        try:
            # 设置环境变量避免OpenMP冲突
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 设备设置
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message(f"使用设备: {device}")
            
            # 加载模型
            if not os.path.exists(self.model_path):
                self.log_message(f"错误: 模型文件 {self.model_path} 不存在")
                return
                
            model = load_model(self.model_path, device)
            self.log_message("模型加载成功")
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # 初始化手机屏幕检测器
            phone_detector = PhoneScreenDetector()
            
            # 检查ADB连接
            if not phone_detector.check_adb_connection():
                self.log_message("错误: 未找到连接的Android设备")
                return
                
            self.log_message("ADB连接成功")
            
            # 初始化可视化器
            visualizer = RealTimeVisualizer()
            
            # 状态控制变量
            current_state = "detecting"  # 状态: detecting, tapping, waiting
            last_detection_time = 0
            tap_start_time = 0
            tap_duration = 1.0  # 点击持续1秒
            last_detection_result = None
            current_action = None
            
            # FPS计算
            fps_start_time = time.time()
            frame_count = 0
            
            while not self.stop_inference:
                current_time = time.time()
                frame_count += 1
                
                # 状态机逻辑
                if current_state == "detecting":
                    # 捕获手机屏幕
                    frame = phone_detector.capture_phone_screen()
                    if frame is None:
                        self.log_message("无法捕获手机屏幕")
                        time.sleep(1)
                        continue
                    
                    # 进行检测
                    with torch.no_grad():
                        processed_frame, frame_rgb = preprocess_frame(frame, transform)
                        processed_frame = processed_frame.unsqueeze(0).to(device)
                        
                        bbox_output, keypoints_output, line_output = model(processed_frame)
                        
                        last_detection_time = current_time
                        last_detection_result = (bbox_output[0], keypoints_output[0], line_output[0])
                        
                        # 计算夹角并决定点击动作
                        height, width = frame_rgb.shape[:2]
                        angle = calculate_angle(keypoints_output[0], line_output[0], width, height)
                        
                        if angle < 0:
                            # 夹角为负数：点击油门区域（右下角）
                            action = 'gas'
                            position = visualizer.gas_position
                        else:
                            # 夹角为正数：点击刹车区域（左下角）
                            action = 'brake'
                            position = visualizer.brake_position
                        
                        self.log_message(f"检测完成 - 夹角: {angle:.1f}° -> 开始点击{action.upper()}")
                        print(f"[DETECTION] 夹角: {angle:.1f}° -> {action.upper()}")
                        
                        # 开始点击 - 使用动态计算的点击时长
                        visualizer.stop_continuous_tap()
                        
                        # 计算动态点击时长
                        final_angle = angle
                        duration = visualizer.calculate_dynamic_duration(final_angle)
                        
                        # 设置状态机中的持续时间（秒）
                        tap_duration = duration / 1000.0  # 转换为秒
                        
                        # 添加角度阈值控制：角度绝对值小于阈值时不执行点击
                        angle_threshold = 5  # 5度阈值
                        if abs(final_angle) < angle_threshold:
                            self.log_message(f"角度{final_angle:.1f}°小于阈值{angle_threshold}°，跳过点击")
                            print(f"[SKIP_TAP] 角度{final_angle:.1f}°小于阈值，跳过点击")
                            current_state = "detecting"  # 继续检测
                            continue
                        success = visualizer.tap_screen(position[0], position[1], duration=duration)
                        if success:
                            current_action = action
                            tap_start_time = current_time
                            current_state = "tapping"
                            self.log_message(f"成功开始点击{action.upper()}位置: {position}")
                        else:
                            self.log_message(f"点击{action.upper()}失败，请检查ADB连接")
                            current_state = "detecting"  # 继续检测
                
                elif current_state == "tapping":
                    # 检查是否点击时间已到 - 确保精确1秒
                    if current_time - tap_start_time >= tap_duration:
                        # 停止点击（ADB长按会自动结束，这里确保状态正确）
                        visualizer.stop_continuous_tap()
                        self.log_message(f"点击{current_action.upper()}完成，持续{tap_duration}秒")
                        print(f"[TAP_COMPLETE] {current_action.upper()}点击完成")
                        current_state = "detecting"  # 回到检测状态
                        current_action = None
                
                # 计算FPS
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    frame_count = 0
                else:
                    fps = None
                
                # 使用最新的检测结果更新显示
                if last_detection_result:
                    bbox, keypoints, line = last_detection_result
                    visualizer.update_display(frame_rgb, bbox, keypoints, line, fps)
                    
                    # 将检测结果推送到手机（确保图片不模糊）
                    display_frame = frame_rgb.copy()
                    height, width = display_frame.shape[:2]
                    
                    # 计算夹角
                    angle = calculate_angle(keypoints, line, width, height)
                    
                    # 绘制完整的检测结果（边界框、关键点、坡度线）
                    if bbox is not None and torch.any(bbox > 0):
                        bbox_pixels = denormalize_coordinates(bbox, width, height)
                        x1, y1, x2, y2 = bbox_pixels
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(display_frame, 'BBox', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if keypoints is not None and torch.any(keypoints > 0):
                        keypoints_pixels = denormalize_keypoints(keypoints, width, height)
                        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 165, 255)]
                        for i, (x, y) in enumerate(keypoints_pixels):
                            if x > 0 or y > 0:
                                cv2.circle(display_frame, (x, y), 3, colors[i], -1)
                                cv2.putText(display_frame, f'KP{i+1}', (x+5, y+5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
                    
                    if line is not None and torch.any(line > 0):
                        line_pixels = denormalize_coordinates(line, width, height)
                        x1, y1, x2, y2 = line_pixels
                        cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.rectangle(display_frame, (x1-3, y1-3), (x1+3, y1+3), (0, 255, 0), -1)
                        cv2.rectangle(display_frame, (x2-3, y2-3), (x2+3, y2+3), (0, 255, 0), -1)
                        cv2.putText(display_frame, 'Slope Line', (x1, y1-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 绘制黄色边界框显示点击区域
                    visualizer._draw_click_regions(display_frame, angle)
                    
                    # 添加角度信息
                    angle_text = f"夹角: {angle:.1f}°"
                    cv2.putText(display_frame, angle_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 添加操作状态
                    state_text = f"状态: {current_state}"
                    cv2.putText(display_frame, state_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if current_action:
                        action_text = f"当前操作: {'油门' if current_action == 'gas' else '刹车'}"
                        cv2.putText(display_frame, action_text, (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 添加时间戳
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(display_frame, timestamp, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 推送到手机（使用高质量PNG编码确保不模糊）
                    phone_detector.push_to_phone(display_frame, "phone_detection_result.png")
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                # 在等待状态时稍微延迟，避免过度占用CPU
                if current_state == "detecting":
                    time.sleep(0.1)
            
            # 清理资源
            visualizer.close()
            cv2.destroyAllWindows()
            self.log_message("手机屏幕检测完成")
            
        except Exception as e:
            self.log_message(f"手机屏幕检测过程中发生错误: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
        
        finally:
            # 重新启用按钮
            self.camera_btn.config(state="normal")
            self.video_btn.config(state="normal")
            self.image_btn.config(state="normal")
            self.phone_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
    def run_image_inference(self, image_path):
        """运行图像推理"""
        try:
            # 设置环境变量避免OpenMP冲突
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 设备设置
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message(f"使用设备: {device}")
            
            # 加载模型
            if not os.path.exists(self.model_path):
                self.log_message(f"错误: 模型文件 {self.model_path} 不存在")
                return
                
            model = load_model(self.model_path, device)
            self.log_message("模型加载成功")
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                self.log_message(f"错误: 无法读取图像文件 {image_path}")
                return
                
            # 预处理图像
            processed_image, image_rgb = preprocess_frame(image, transform)
            processed_image = processed_image.unsqueeze(0).to(device)
            
            # 进行推理
            with torch.no_grad():
                bbox_output, keypoints_output, line_output = model(processed_image)
                
                bbox = bbox_output[0]
                keypoints = keypoints_output[0]
                line = line_output[0]
                
                # 显示结果
                visualizer = RealTimeVisualizer()
                visualizer.update_display(image_rgb, bbox, keypoints, line)
                
                self.log_message("图像推理完成")
                self.log_message("按任意键关闭窗口...")
                
                # 等待用户关闭窗口
                plt.waitforbuttonpress()
                visualizer.close()
                
        except Exception as e:
            self.log_message(f"推理过程中发生错误: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            
    def run(self):
        """运行GUI"""
        self.root.mainloop()

# 主程序入口
if __name__ == "__main__":
    app = VideoInferenceGUI()
    app.run()
