import cv2
import os
import json
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import math

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆图像标注工具")
        self.root.geometry("1200x800")
        
        # 标注状态
        self.current_mode = "bbox"  # bbox, keypoints, line
        self.current_image_path = None
        self.current_image = None
        self.display_image = None
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        
        # 标注数据
        self.bboxes = []
        self.keypoints = []
        self.lines = []
        self.current_bbox = None
        self.current_line = []
        
        # 关键点类型和顺序
        self.keypoint_types = ["front_wheel", "rear_wheel", "body_center", "roof_top"]
        self.keypoint_labels = ["前轮中心", "后轮中心", "车身中心", "车顶最高点"]
        self.current_keypoint_type = 0
        self.keypoint_sequence = []  # 存储按顺序标注的关键点
        
        # 创建界面
        self.create_widgets()
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-2>", self.on_middle_click)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件操作
        file_frame = ttk.LabelFrame(control_frame, text="文件操作")
        file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(file_frame, text="打开图像文件夹", command=self.open_image_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="保存标注", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        
        # 标注模式
        mode_frame = ttk.LabelFrame(control_frame, text="标注模式")
        mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.mode_var = tk.StringVar(value="bbox")
        ttk.Radiobutton(mode_frame, text="边界框", variable=self.mode_var, 
                       value="bbox", command=self.change_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="关键点", variable=self.mode_var, 
                       value="keypoints", command=self.change_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="坡度线", variable=self.mode_var, 
                       value="line", command=self.change_mode).pack(side=tk.LEFT, padx=5)
        
        # 关键点类型选择
        self.keypoint_var = tk.StringVar(value="前轮中心")
        keypoint_combo = ttk.Combobox(mode_frame, textvariable=self.keypoint_var, 
                                     values=["前轮中心", "后轮中心", "车身中心", "车顶最高点"],
                                     state="readonly", width=10)
        keypoint_combo.pack(side=tk.LEFT, padx=5)
        keypoint_combo.bind("<<ComboboxSelected>>", self.on_keypoint_type_change)
        
        # 操作按钮
        action_frame = ttk.LabelFrame(control_frame, text="操作")
        action_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(action_frame, text="清除当前", command=self.clear_current).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="清除全部", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="缩放重置", command=self.reset_zoom).pack(side=tk.LEFT, padx=5)
        
        # 图像显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建画布
        self.canvas = tk.Canvas(display_frame, bg="gray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # 图像信息
        self.info_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.info_var).pack(side=tk.RIGHT)
        
    def open_image_folder(self):
        folder_path = filedialog.askdirectory(title="选择图像文件夹")
        if folder_path:
            self.image_folder = folder_path
            # 过滤图像文件，排除可视化图片和标注文件
            self.image_files = []
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 排除可视化图片（包含_visualization）和标注文件（包含_annotations）
                    if '_visualization' not in f.lower() and '_annotations' not in f.lower():
                        self.image_files.append(f)
            
            self.current_image_index = 0
            if self.image_files:
                self.load_image(0)
                self.update_status(f"已加载 {len(self.image_files)} 张图像")
            else:
                messagebox.showwarning("警告", "文件夹中没有找到图像文件")
    
    def load_image(self, index):
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            image_path = os.path.join(self.image_folder, self.image_files[index])
            self.current_image_path = image_path
            
            # 加载图像
            self.current_image = cv2.imread(image_path)
            if self.current_image is not None:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                self.display_image = self.current_image.copy()
                self.zoom_factor = 1.0
                self.pan_offset = [0, 0]
                self.load_annotations()
                self.update_display()
                self.update_info()
            else:
                messagebox.showerror("错误", f"无法加载图像: {image_path}")
    
    def load_annotations(self):
        # 清除当前标注
        self.bboxes = []
        self.keypoints = []
        self.lines = []
        
        # 尝试加载已有的标注文件
        annotation_path = self.get_annotation_path()
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'bboxes' in data:
                    self.bboxes = data['bboxes']
                if 'keypoints' in data:
                    self.keypoints = data['keypoints']
                if 'lines' in data:
                    self.lines = data['lines']
                    
                self.update_status("已加载现有标注")
            except Exception as e:
                self.update_status(f"加载标注失败: {str(e)}")
    
    def get_annotation_path(self):
        if self.current_image_path:
            image_name = Path(self.current_image_path).stem
            return os.path.join(self.image_folder, f"{image_name}_annotations.json")
        return None
    
    def save_annotations(self):
        if not self.current_image_path:
            messagebox.showwarning("警告", "没有加载图像")
            return
        
        annotation_path = self.get_annotation_path()
        data = {
            'image_path': self.current_image_path,
            'bboxes': self.bboxes,
            'keypoints': self.keypoints,
            'lines': self.lines
        }
        
        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 保存可视化图片
            self.save_visualization()
            
            self.update_status("标注已保存，可视化图片已生成")
        except Exception as e:
            messagebox.showerror("错误", f"保存标注失败: {str(e)}")
    
    def save_visualization(self):
        """保存带有标注的可视化图片"""
        if self.current_image is None:
            return
        
        # 创建可视化图像
        vis_image = self.current_image.copy()
        
        # 绘制所有标注
        self.draw_annotations_on_image(vis_image)
        
        # 生成可视化图片路径
        image_name = Path(self.current_image_path).stem
        vis_path = os.path.join(self.image_folder, f"{image_name}_visualization.jpg")
        
        # 保存图片（转换为BGR格式）
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(vis_path, vis_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    def draw_annotations_on_image(self, img):
        """在图像上绘制标注（不缩放）"""
        h, w = img.shape[:2]
        
        # 绘制边界框
        for bbox in self.bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, "Vehicle", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制关键点
        for kp in self.keypoints:
            x, y, kp_type = kp
            x, y = int(x), int(y)
            color = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)][kp_type]
            label = ["前轮", "后轮", "车身", "车顶"][kp_type]
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.putText(img, label, (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 绘制关键点之间的连线（前轮-后轮，车身-车顶）
            if kp_type == 0:  # 前轮
                rear_wheel = next((kp2 for kp2 in self.keypoints if kp2[2] == 1), None)
                if rear_wheel:
                    cv2.line(img, (x, y), (int(rear_wheel[0]), int(rear_wheel[1])), (255, 255, 255), 2)
            elif kp_type == 2:  # 车身
                roof_top = next((kp2 for kp2 in self.keypoints if kp2[2] == 3), None)
                if roof_top:
                    cv2.line(img, (x, y), (int(roof_top[0]), int(roof_top[1])), (255, 255, 255), 2)
        
        # 绘制坡度线
        for line in self.lines:
            if len(line) >= 2:
                points = [(int(x), int(y)) for x, y in line]
                for i in range(len(points)-1):
                    cv2.line(img, points[i], points[i+1], (0, 255, 255), 3)
                for point in points:
                    cv2.circle(img, point, 5, (0, 255, 255), -1)
                
                # 计算并显示坡度角度
                if len(points) >= 2:
                    dx = points[-1][0] - points[0][0]
                    dy = points[-1][1] - points[0][1]
                    if dx != 0:
                        angle = math.degrees(math.atan2(dy, dx))
                        cv2.putText(img, f"坡度: {angle:.1f}°", 
                                  (points[0][0], points[0][1]-20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def change_mode(self):
        self.current_mode = self.mode_var.get()
        if self.current_mode == "keypoints":
            self.keypoint_sequence = []  # 重置关键点序列
            self.update_status("关键点模式：请按顺序标注前轮、后轮、车身、车顶")
        else:
            self.update_status(f"切换到 {self.current_mode} 模式")
    
    def on_keypoint_type_change(self, event):
        mapping = {
            "前轮中心": 0,
            "后轮中心": 1,
            "车身中心": 2,
            "车顶最高点": 3
        }
        self.current_keypoint_type = mapping[self.keypoint_var.get()]
    
    def clear_current(self):
        if self.current_mode == "bbox":
            self.bboxes = []
        elif self.current_mode == "keypoints":
            self.keypoints = []
            self.keypoint_sequence = []  # 同时清除关键点序列
        elif self.current_mode == "line":
            self.lines = []
        self.update_display()
        self.update_status("已清除当前模式标注")
    
    def clear_all(self):
        self.bboxes = []
        self.keypoints = []
        self.lines = []
        self.update_display()
        self.update_status("已清除所有标注")
    
    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.update_display()
    
    def prev_image(self):
        if hasattr(self, 'image_files') and self.current_image_index > 0:
            self.save_annotations()
            self.load_image(self.current_image_index - 1)
    
    def next_image(self):
        if hasattr(self, 'image_files') and self.current_image_index < len(self.image_files) - 1:
            self.save_annotations()
            self.load_image(self.current_image_index + 1)
    
    def update_display(self):
        if self.current_image is None:
            return
        
        # 创建显示图像副本
        display_img = self.display_image.copy()
        
        # 应用缩放和平移
        h, w = display_img.shape[:2]
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        if new_w != w or new_h != h:
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 绘制标注
        self.draw_annotations(display_img)
        
        # 转换为PhotoImage并显示
        pil_image = Image.fromarray(display_img)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_offset[0], self.pan_offset[1], 
                               anchor=tk.NW, image=self.tk_image)
    
    def draw_annotations(self, img):
        h, w = img.shape[:2]
        scale_x = w / self.current_image.shape[1]
        scale_y = h / self.current_image.shape[0]
        
        # 绘制边界框
        for bbox in self.bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Vehicle", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制关键点
        for kp in self.keypoints:
            x, y, kp_type = kp
            x, y = int(x * scale_x), int(y * scale_y)
            color = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)][kp_type]
            label = ["前轮", "后轮", "车身", "车顶"][kp_type]
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.putText(img, label, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制坡度线
        for line in self.lines:
            if len(line) >= 2:
                points = [(int(x * scale_x), int(y * scale_y)) for x, y in line]
                for i in range(len(points)-1):
                    cv2.line(img, points[i], points[i+1], (0, 255, 255), 2)
                for point in points:
                    cv2.circle(img, point, 3, (0, 255, 255), -1)
    
    def on_click(self, event):
        if self.current_image is None:
            return
        
        # 转换坐标到原始图像空间
        x, y = self.screen_to_image_coords(event.x, event.y)
        
        if self.current_mode == "bbox":
            self.current_bbox = [x, y, x, y]
        elif self.current_mode == "keypoints":
            # 按顺序标注关键点：前轮(0)、后轮(1)、车身(2)、车顶(3)
            if len(self.keypoint_sequence) < 4:
                kp_type = len(self.keypoint_sequence)  # 自动按顺序分配类型
                self.keypoints.append([x, y, kp_type])
                self.keypoint_sequence.append([x, y, kp_type])
                self.update_status(f"已标注 {self.keypoint_labels[kp_type]} ({len(self.keypoint_sequence)}/4)")
                self.update_display()
                
                # 如果四个点都标注完成，显示完成提示
                if len(self.keypoint_sequence) == 4:
                    self.update_status("关键点标注完成！")
            else:
                self.update_status("已标注完四个关键点，如需重新标注请清除当前标注")
        elif self.current_mode == "line":
            self.current_line = [[x, y]]
    
    def on_drag(self, event):
        if self.current_image is None:
            return
        
        x, y = self.screen_to_image_coords(event.x, event.y)
        
        if self.current_mode == "bbox" and self.current_bbox:
            self.current_bbox[2] = x
            self.current_bbox[3] = y
            self.update_display_with_current_bbox()
    
    def on_release(self, event):
        if self.current_image is None:
            return
        
        x, y = self.screen_to_image_coords(event.x, event.y)
        
        if self.current_mode == "bbox" and self.current_bbox:
            # 确保边界框坐标正确
            x1, y1, x2, y2 = self.current_bbox
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 最小尺寸检查
                self.bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            self.current_bbox = None
            self.update_display()
        
        elif self.current_mode == "line" and self.current_line:
            self.current_line.append([x, y])
            if len(self.current_line) >= 2:
                self.lines.append(self.current_line.copy())
                self.current_line = []
                self.update_display()
    
    def on_mousewheel(self, event):
        if self.current_image is None:
            return
        
        # 缩放
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_factor *= scale_factor
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))  # 限制缩放范围
        self.update_display()
    
    def on_middle_click(self, event):
        self.pan_start = [event.x, event.y]
    
    def on_middle_drag(self, event):
        if hasattr(self, 'pan_start'):
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.pan_start = [event.x, event.y]
            self.update_display()
    
    def screen_to_image_coords(self, screen_x, screen_y):
        if self.current_image is None:
            return 0, 0
        
        # 考虑平移偏移
        img_x = screen_x - self.pan_offset[0]
        img_y = screen_y - self.pan_offset[1]
        
        # 考虑缩放
        img_x = img_x / self.zoom_factor
        img_y = img_y / self.zoom_factor
        
        # 限制在图像范围内
        img_x = max(0, min(img_x, self.current_image.shape[1]))
        img_y = max(0, min(img_y, self.current_image.shape[0]))
        
        return int(img_x), int(img_y)
    
    def update_display_with_current_bbox(self):
        if self.current_image is None:
            return
        
        display_img = self.display_image.copy()
        
        # 应用缩放和平移
        h, w = display_img.shape[:2]
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        if new_w != w or new_h != h:
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 绘制现有标注
        self.draw_annotations(display_img)
        
        # 绘制当前正在绘制的边界框
        if self.current_bbox:
            scale_x = new_w / self.current_image.shape[1]
            scale_y = new_h / self.current_image.shape[0]
            x1, y1, x2, y2 = self.current_bbox
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 转换为PhotoImage并显示
        pil_image = Image.fromarray(display_img)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_offset[0], self.pan_offset[1], 
                               anchor=tk.NW, image=self.tk_image)
    
    def update_status(self, message):
        self.status_var.set(message)
    
    def update_info(self):
        if self.current_image is not None:
            info = f"图像: {self.image_files[self.current_image_index]} | "
            info += f"尺寸: {self.current_image.shape[1]}x{self.current_image.shape[0]} | "
            info += f"边界框: {len(self.bboxes)} | "
            info += f"关键点: {len(self.keypoints)} | "
            info += f"坡度线: {len(self.lines)}"
            self.info_var.set(info)

def main():
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
