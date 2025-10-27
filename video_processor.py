import cv2
import os
import argparse
from pathlib import Path

def process_video(video_path, output_dir, frames_per_second=3):
    """
    处理视频文件，每秒提取指定数量的帧
    
    Args:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录
        frames_per_second (int): 每秒提取的帧数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频文件名（不含扩展名）
    video_name = Path(video_path).stem
    
    # 创建视频特定的输出目录
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"处理视频: {video_name}")
    print(f"帧率: {fps:.2f} fps")
    print(f"总帧数: {total_frames}")
    print(f"时长: {duration:.2f} 秒")
    
    # 计算帧间隔（每多少帧提取一帧）
    frame_interval = int(fps / frames_per_second)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 每秒提取指定数量的帧
        if frame_count % frame_interval == 0:
            # 计算当前时间（秒）
            current_time = frame_count / fps
            
            # 生成文件名：视频名_时间戳_帧编号.jpg
            timestamp = f"{int(current_time):06d}"
            frame_number = (frame_count // frame_interval) % frames_per_second
            filename = f"{video_name}_{timestamp}_{frame_number:02d}.jpg"
            output_path = os.path.join(video_output_dir, filename)
            
            # 保存帧
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 30 == 0:  # 每保存30张图片打印一次进度
                print(f"已保存 {saved_count} 张图片...")
        
        frame_count += 1
    
    cap.release()
    print(f"完成！从视频 {video_name} 中提取了 {saved_count} 张图片")
    print(f"图片保存在: {video_output_dir}")

def main():
    parser = argparse.ArgumentParser(description='视频帧提取工具')
    parser.add_argument('--input_dir', type=str, default='raw-videos',
                       help='输入视频目录路径 (默认: raw-videos)')
    parser.add_argument('--output_dir', type=str, default='dataset/cleaning data',
                       help='输出图片目录路径 (默认: dataset/cleaning data)')
    parser.add_argument('--frames_per_second', type=int, default=3,
                       help='每秒提取的帧数 (默认: 3)')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录 {args.input_dir} 不存在")
        return
    
    # 获取所有MP4文件
    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.mp4')]
    
    if not video_files:
        print(f"在目录 {args.input_dir} 中未找到MP4文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for video_file in video_files:
        print(f"  - {video_file}")
    
    print(f"\n开始处理视频...")
    print(f"每秒提取帧数: {args.frames_per_second}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 50)
    
    # 处理每个视频文件
    for video_file in video_files:
        video_path = os.path.join(args.input_dir, video_file)
        process_video(video_path, args.output_dir, args.frames_per_second)
        print("-" * 50)
    
    print("所有视频处理完成！")

if __name__ == "__main__":
    main()
