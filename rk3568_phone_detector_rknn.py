#!/usr/bin/env python3
"""
RK3568手机屏幕检测程序 - 使用RKNNLite
基于phone_screen_test.py的逻辑，适配RK3568开发板
"""

import time
import os
import math
import subprocess
import numpy as np
import cv2
import argparse
from rknnlite.api import RKNNLite as RKNN


class PhoneScreenClickController:
    """手机屏幕点击控制器"""
    
    def __init__(self):
        self.adb_path = self.find_adb()
        
        # 定义点击区域位置（参考phone_screen_test.py）
        self.screen_width = 1080  # 常见手机横向宽度
        self.screen_height = 1920  # 常见手机横向高度
        # 使用用户指定的精确坐标
        self.brake_position = (369, 991)   # 左下角刹车区域
        self.gas_position = (2417, 982)     # 右下角油门区域
        
        # 点击控制相关变量
        self.current_action = None  # 当前正在执行的动作 ('gas' 或 'brake')
        self.current_tap_process = None  # 当前点击进程
        self.last_tap_position = None  # 上次点击位置
    
    def find_adb(self):
        """查找ADB可执行文件路径"""
        possible_paths = [
            "adb",  # 如果在PATH中
            "/usr/bin/adb",
            "/usr/local/bin/adb",
            "platform-tools/adb"
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
    
    def tap_screen_with_duration(self, x, y, duration=500):
        """模拟屏幕点击（精确时长）"""
        try:
            # 使用ADB模拟精确时长的长按
            command = [self.adb_path, "shell", "input", "swipe", str(x), str(y), str(x), str(y), str(duration)]
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                print(f"点击失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"点击失败: {e}")
            return False


class RKNNPhoneScreenDetector:
    """RKNN手机屏幕检测器 - 使用best.rknn模型"""
    
    def __init__(self, model_path="best.rknn", conf_threshold=0.5):
        """
        初始化RKNN手机屏幕检测器
        
        Args:
            model_path: RKNN模型路径
            conf_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.rknn = None
        
        # 关键点配置 - 根据YOLO_inference.py中的配置
        self.num_keypoints = 6  # 4个原始关键点 + 2个坡度线关键点
        
        # 输入尺寸
        self.input_height = 640
        self.input_width = 640
        
        # 加载模型
        self.load_model()
        
        # 初始化点击控制器
        self.click_controller = PhoneScreenClickController()
        
        # 点击控制状态变量 - 修改为持续按压模式
        self.current_state = "detecting"  # 状态: detecting, maintaining
        self.current_action = None  # 当前动作: 'gas' 或 'brake'
        self.last_action_change_time = 0  # 上次动作改变时间
        self.consecutive_negative_count = 0  # 连续负角度计数
        self.consecutive_positive_count = 0  # 连续正角度计数
        self.last_angle = 0
        self.current_tap_process = None  # 当前点击进程
        
    def load_model(self):
        """加载RKNN模型"""
        print(f"正在加载RKNN模型: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建RKNN对象
        self.rknn = RKNN()
        
        # 加载RKNN模型
        print('--> Loading RKNN model')
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"加载RKNN模型失败: {ret}")
        
        # 初始化运行时环境
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"初始化运行时环境失败: {ret}")
        
        print("RKNN模型加载成功!")
        print(f"输入尺寸: {self.input_width}x{self.input_height}")
        print(f"关键点数量: {self.num_keypoints}")
        
        # 打印模型信息用于调试
        print("[DEBUG] 模型信息:")
        print(f"  SDK版本: {self.rknn.get_sdk_version()}")
    
    def capture_phone_screen(self):
        """捕获手机屏幕"""
        try:
            # 使用ADB截图
            result = subprocess.run([self.click_controller.adb_path, "exec-out", "screencap", "-p"], 
                                  capture_output=True)
            
            if result.returncode == 0:
                # 将截图数据转换为numpy数组
                nparr = np.frombuffer(result.stdout, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
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
    
    def preprocess_frame(self, frame):
        """预处理帧数据用于RKNN推理"""
        if frame is None:
            return None
        
        # 调整尺寸到模型输入尺寸
        resized_frame = cv2.resize(frame, (self.input_width, self.input_height))
        
        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # 归一化到0-1范围
        normalized_frame = rgb_frame.astype(np.float32) / 255.0
        
        # 转换为CHW格式 (Channel, Height, Width)
        chw_frame = np.transpose(normalized_frame, (2, 0, 1))
        
        # 添加batch维度
        input_data = np.expand_dims(chw_frame, axis=0)
        
        return input_data
    
    def postprocess_results(self, outputs, original_height, original_width):
        """后处理RKNN推理结果 - 基于YOLO关键点检测模型输出格式"""
        if not outputs:
            return None, None
        
        try:
            # YOLO关键点检测模型通常输出一个包含检测结果的张量
            # 格式: [batch, num_detections, 5 + num_keypoints*3]
            # 5 = [x_center, y_center, width, height, confidence]
            # 每个关键点有3个值: [x, y, visibility]
            
            # 获取第一个输出（假设是检测结果）
            detections = outputs[0]
            
            # 检查是否有检测结果
            if detections.size == 0 or len(detections.shape) < 2:
                return None, None
            
            # 获取检测数量
            num_detections = detections.shape[1]
            if num_detections == 0:
                return None, None
            
            # 找到置信度最高的检测结果
            best_detection = None
            best_confidence = 0
            
            for i in range(num_detections):
                detection = detections[0, i]  # 第一个batch
                confidence = detection[4]  # 置信度在索引4
                
                if confidence > self.conf_threshold and confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = detection
            
            if best_detection is None:
                return None, None
            
            print(f"[DEBUG] 检测到目标，置信度: {best_confidence:.3f}")
            
            # 解析关键点信息
            # 关键点从索引5开始，每个关键点有3个值: [x, y, visibility]
            keypoints_start_idx = 5
            keypoints_data = best_detection[keypoints_start_idx:]
            
            # 提取6个关键点（4个原始关键点 + 2个坡度线关键点）
            keypoints = []
            for i in range(self.num_keypoints):
                start_idx = i * 3
                if start_idx + 2 < len(keypoints_data):
                    x = keypoints_data[start_idx] * self.input_width
                    y = keypoints_data[start_idx + 1] * self.input_height
                    visibility = keypoints_data[start_idx + 2]
                    
                    # 只使用可见的关键点（visibility > 0.5）
                    if visibility > 0.679:
                        keypoints.append([x, y])
                    else:
                        keypoints.append([0, 0])  # 不可见的关键点设为0
                else:
                    keypoints.append([0, 0])
            
            # 转换为numpy数组
            keypoints_array = np.array(keypoints, dtype=np.float32)
            
            # 提取坡度线（最后2个关键点）
            if len(keypoints) >= 6:
                slope_start = keypoints[4]  # 坡度线起点（关键点4）
                slope_end = keypoints[5]    # 坡度线终点（关键点5）
                slope_line = [slope_start[0], slope_start[1], slope_end[0], slope_end[1]]
            else:
                slope_line = [0, 0, 0, 0]
            
            # 打印调试信息
            print(f"[DEBUG] 关键点坐标:")
            for i, kp in enumerate(keypoints):
                print(f"  KP{i}: ({kp[0]:.1f}, {kp[1]:.1f})")
            
            print(f"[DEBUG] 坡度线: ({slope_line[0]:.1f}, {slope_line[1]:.1f}) -> ({slope_line[2]:.1f}, {slope_line[3]:.1f})")
            
            return keypoints_array, slope_line
            
        except Exception as e:
            print(f"后处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict_frame(self, frame):
        """对帧进行预测"""
        if frame is None:
            return None, None
        
        # 预处理帧
        input_data = self.preprocess_frame(frame)
        if input_data is None:
            return None, None
        
        # 进行推理
        try:
            outputs = self.rknn.inference(inputs=[input_data])
            
            # 后处理结果
            height, width = frame.shape[:2]
            keypoints, slope_line = self.postprocess_results(outputs, height, width)
            
            return keypoints, slope_line
            
        except Exception as e:
            print(f"推理失败: {e}")
            return None, None
    
    def calculate_angle(self, keypoints, line, width, height):
        """计算有向夹角：前轮减后轮的夹角指向坡度线上方时角度为正，指向坡度线下方时坡度为负"""
        if keypoints is None or line is None:
            return 0
        
        # 假设关键点包含前后轮中心（前两个关键点）
        if len(keypoints) < 4:  # 至少需要2个关键点（x,y坐标）
            return 0
        
        # 将关键点reshape为(N, 2)格式
        keypoints_reshaped = keypoints.reshape(-1, 2)
        
        if len(keypoints_reshaped) < 2:
            return 0
        
        # 前后轮中心连线
        front_wheel = keypoints_reshaped[0]  # 前轮
        rear_wheel = keypoints_reshaped[1]   # 后轮
        
        # 坡度线
        slope_start = (line[0], line[1])  # 坡度线起点
        slope_end = (line[2], line[3])    # 坡度线终点
        
        # 计算前后轮连线向量
        wheel_vector = (front_wheel[0] - rear_wheel[0], front_wheel[1] - rear_wheel[1])
        
        # 计算坡度线向量（从起点到终点）
        slope_vector = (slope_end[0] - slope_start[0], slope_end[1] - slope_start[1])
        
        # 计算有向角度：使用atan2计算两个向量之间的有向角度
        # 计算前后轮连线向量相对于坡度线向量的角度
        wheel_angle = math.atan2(wheel_vector[1], wheel_vector[0])
        slope_angle = math.atan2(slope_vector[1], slope_vector[0])
        
        # 计算相对角度（弧度）
        relative_angle = wheel_angle - slope_angle
        
        # 将角度归一化到[-π, π]范围
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        # 转换为角度
        angle_deg = math.degrees(relative_angle)
        
        # 根据用户需求调整角度正负定义：
        # 前轮减后轮的夹角指向坡度线上方时角度为正，指向坡度线下方时坡度为负
        # 在图像坐标系中，y轴向下为正，所以需要调整符号
        final_angle = -angle_deg
        
        return final_angle-45
    
    def run_detection(self, detection_interval=0.5):
        """运行手机屏幕检测 - 精确时序控制"""
        print("开始手机屏幕检测...")
        print(f"使用模型: {self.model_path}")
        print(f"检测间隔: {detection_interval}秒")
        print("点击位置: 刹车(369, 991), 油门(2417, 982)")
        print("点击模式: 油门1秒，刹车0.5秒，触摸维持后开始检测")
        print("时序控制: 检测和触摸不能同时进行")
        
        # 检查ADB连接
        if not self.click_controller.check_adb_connection():
            print("错误: 未找到连接的Android设备")
            return
        
        print("ADB连接成功")
        
        # 状态变量
        last_detection_time = 0
        frame_count = 0
        fps_start_time = time.time()
        
        # 时序控制变量
        self.current_state = "detecting"  # 初始状态：检测
        self.current_action = None
        self.tap_start_time = 0
        self.tap_duration = 2  # 每次触摸0.5秒
        self.last_angle = 0
        
        try:
            while True:
                current_time = time.time()
                frame_count += 1
                
                # 状态机控制
                if self.current_state == "detecting":
                    # 检测状态：进行检测并决定下一步动作
                    if current_time - last_detection_time >= detection_interval:
                        # 捕获手机屏幕
                        frame = self.capture_phone_screen()
                        if frame is None:
                            print("无法捕获手机屏幕，跳过此帧")
                            time.sleep(0.1)
                            continue
                        
                        # 进行预测
                        keypoints, line = self.predict_frame(frame)
                        
                        # 计算夹角
                        height, width = frame.shape[:2]
                        angle = self.calculate_angle(keypoints, line, width, height)
                        self.last_angle = angle
                        
                        # 决定点击动作
                        if angle < 0:
                            # 夹角为负数：点击油门区域（右下角）- 1秒
                            action = 'gas'
                            position = self.click_controller.gas_position
                            tap_duration = 1200  # 1秒
                            tap_duration_seconds = 1.2
                        else:
                            # 夹角为正数：点击刹车区域（左下角）- 0.5秒
                            action = 'brake'
                            position = self.click_controller.brake_position
                            tap_duration = 500  # 0.5秒
                            tap_duration_seconds = 0.5
                        
                        # 添加角度阈值控制：角度绝对值小于阈值时不执行点击
                        angle_threshold = 5  # 5度阈值
                        if abs(angle) < angle_threshold:
                            print(f"[SKIP_TAP] 角度{angle:.1f}°小于阈值{angle_threshold}°，跳过点击")
                            self.current_action = None
                            # 保持在检测状态
                        else:
                            # 开始触摸
                            print(f"[TAP_START] 开始{action.upper()}点击，位置: {position}, 持续时间: {tap_duration_seconds}秒")
                            success = self.click_controller.tap_screen_with_duration(position[0], position[1], duration=tap_duration)
                            
                            if success:
                                self.current_action = action
                                self.current_state = "tapping"
                                self.tap_start_time = current_time
                                self.tap_duration = tap_duration_seconds
                                print(f"[TAP_START] {action.upper()}点击开始，持续{tap_duration_seconds}秒")
                            else:
                                print(f"[TAP_FAIL] {action.upper()}点击失败，请检查ADB连接")
                                self.current_action = None
                                # 保持在检测状态
                        
                        # 打印检测信息
                        tap_status = "无点击" if self.current_action is None else f"准备点击{self.current_action.upper()}"
                        print(f"[DETECTION] 夹角: {angle:.1f}° -> 建议操作: {'油门' if angle < 0 else '刹车'} -> {tap_status}")
                        
                        last_detection_time = current_time
                
                elif self.current_state == "tapping":
                    # 触摸状态：等待触摸维持0.5秒后开始检测
                    if current_time - self.tap_start_time >= self.tap_duration:
                        print(f"[TAP_MAINTAIN] {self.current_action.upper()}点击已维持0.5秒，开始检测")
                        self.current_state = "detecting"
                        self.current_action = None
                
                # 计算FPS
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - fps_start_time)
                    print(f"[FPS] 当前FPS: {fps:.1f}")
                    fps_start_time = time.time()
                    frame_count = 0
                
                # 稍微延迟，避免过度占用CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[EXIT] 检测被用户中断")
        except Exception as e:
            print(f"[ERROR] 检测过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            print("[CLEANUP] 正在清理资源...")
            if self.rknn:
                self.rknn.release()
            print("手机屏幕检测结束")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RK3568手机屏幕检测程序')
    parser.add_argument('--model', type=str, default='best.rknn', help='RKNN模型文件路径')
    parser.add_argument('--interval', type=float, default=0.5, help='检测间隔（秒）')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RK3568手机屏幕检测程序")
    print("=" * 50)
    print(f"使用模型: {args.model}")
    print(f"检测间隔: {args.interval}秒")
    print(f"置信度阈值: {args.conf}")
    print("功能: 实时检测手机屏幕，计算坡度夹角，提供操作建议")
    print("=" * 50)
    
    # 创建检测器
    try:
        detector = RKNNPhoneScreenDetector(
            model_path=args.model,
            conf_threshold=args.conf
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    except RuntimeError as e:
        print(f"错误: {e}")
        return
    
    # 开始检测
    print("开始检测...")
    print("按 Ctrl+C 退出检测")
    print("-" * 50)
    
    detector.run_detection(detection_interval=args.interval)


if __name__ == "__main__":
    # 检查依赖
    try:
        from rknnlite.api import RKNNLite as RKNN
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"错误: 缺少必要的依赖库: {e}")
        print("请确保已安装以下依赖:")
        print("- rknn-toolkit-lite (RKNN运行时)")
        print("- opencv-python (图像处理)")
        print("- numpy (数值计算)")
        exit(1)
    
    # 运行主程序
    main()
