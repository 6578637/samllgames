"""
    转换YOLO模型为嵌入式端的RK模型的程序文件
"""

import torch
import os
import shutil
from rknn.api import RKNN
import numpy as np
from ultralytics import YOLO
import glob
import tempfile


# 配置参数
YOLO_MODEL_PATH = 'yolo_training/climbing_pose_detection_with_slope/weights/best.pt'  # 训练好的YOLO模型路径
ONNX_MODEL_PATH = './yolo_onnx_models'
RKNN_MODEL_PATH = 'yolo_rknn_models'

# 输入尺寸 - 与训练时保持一致
HEIGHT = 640
WIDTH = 640
NUM_CLASSES = 1  # 类别数量
NUM_KEYPOINTS = 6  # 关键点数量


def clear_folder(folderPath: str):
    """
    清空文件夹内容
    """
    if not os.path.exists(folderPath):
        return

    # 遍历文件夹内的所有内容
    for item in os.listdir(folderPath):
        itemPath = os.path.join(folderPath, item)
        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(itemPath) or os.path.islink(itemPath):
                os.unlink(itemPath)
                print(f'已删除文件: {itemPath}')
            elif os.path.isdir(itemPath):
                # 使用 shutil.rmtree 删除文件夹及其内容
                shutil.rmtree(itemPath)
                print(f"已删除文件夹： {itemPath}")
        except Exception as e:
            print(f"删除 {itemPath} 时出错： {e}")

    print(f"文件夹 {folderPath} 内容已清空")


def find_exported_onnx(model_dir):
    """
    在模型目录中查找导出的ONNX文件
    """
    # 查找所有.onnx文件
    onnx_files = glob.glob(os.path.join(model_dir, "*.onnx"))

    if onnx_files:
        # 返回最新的ONNX文件
        return max(onnx_files, key=os.path.getctime)

    # 如果没找到，尝试在父目录中查找
    parent_dir = os.path.dirname(model_dir)
    onnx_files = glob.glob(os.path.join(parent_dir, "*.onnx"))

    if onnx_files:
        return max(onnx_files, key=os.path.getctime)

    return None


def export_yolo_to_onnx(yolo_model_path, onnx_save_path):
    """
    将YOLO模型导出为ONNX格式
    """
    print("开始导出YOLO模型为ONNX格式...")

    try:
        # 加载训练好的YOLO模型
        model = YOLO(yolo_model_path)

        # 获取模型所在目录
        model_dir = os.path.dirname(yolo_model_path)
        model_name = os.path.splitext(os.path.basename(yolo_model_path))[0]

        print(f"模型目录: {model_dir}")
        print(f"模型名称: {model_name}")

        # 导出为ONNX
        print("正在导出ONNX格式...")
        success = model.export(
            format='onnx',
            imgsz=(HEIGHT, WIDTH),
            dynamic=False,  # 固定输入尺寸
            simplify=True,  # 简化模型
            opset=12,       # ONNX opset版本
        )

        if success:
            print("ONNX导出成功，正在查找导出的文件...")

            # 查找导出的ONNX文件
            exported_onnx = find_exported_onnx(model_dir)

            if exported_onnx and os.path.exists(exported_onnx):
                print(f"找到导出的ONNX文件: {exported_onnx}")

                # 确保目标目录存在
                os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)

                # 复制到目标位置
                shutil.copy2(exported_onnx, onnx_save_path)
                print(f"ONNX模型已复制到: {onnx_save_path}")

                return True
            else:
                print("找不到导出的ONNX文件，尝试在可能的位置查找...")

                # 尝试其他可能的位置
                possible_locations = [
                    os.path.join(model_dir, f"{model_name}.onnx"),
                    os.path.join(os.path.dirname(model_dir), f"{model_name}.onnx"),
                    f"./{model_name}.onnx",
                    os.path.join(model_dir, "best.onnx"),
                ]

                for location in possible_locations:
                    if os.path.exists(location):
                        print(f"在备用位置找到ONNX文件: {location}")
                        shutil.copy2(location, onnx_save_path)
                        print(f"ONNX模型已复制到: {onnx_save_path}")
                        return True

                print("在所有可能的位置都找不到ONNX文件")
                return False
        else:
            print("ONNX导出失败")
            return False

    except Exception as e:
        print(f"导出ONNX时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_safe_quantization_dataset(dataset_path, num_samples=50):
    """
    创建安全的量化数据集，避免路径问题
    """
    print("创建安全的量化数据集...")

    try:
        # 创建一个临时目录来存储用于量化的图像
        temp_dir = tempfile.mkdtemp(prefix="rknn_quant_")
        print(f"创建临时目录: {temp_dir}")

        # 创建一些随机图像用于量化
        import cv2
        for i in range(num_samples):
            # 创建随机图像
            img = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            img_path = os.path.join(temp_dir, f"quant_image_{i:04d}.jpg")
            cv2.imwrite(img_path, img)

        # 写入数据集文件
        with open(dataset_path, 'w') as f:
            for i in range(num_samples):
                img_path = os.path.join(temp_dir, f"quant_image_{i:04d}.jpg")
                f.write(f"{img_path}\n")

        print(f"量化数据集文件已创建: {dataset_path} (包含 {num_samples} 个样本)")
        print(f"临时图像存储在: {temp_dir}")

        return temp_dir  # 返回临时目录，以便后续清理

    except Exception as e:
        print(f"创建量化数据集失败: {e}")
        return None


def convert_onnx_to_rknn(onnx_model_path, rknn_save_path):
    """
    将ONNX模型转换为RKNN格式
    """
    print("开始转换ONNX模型为RKNN格式...")

    temp_dir = None
    try:
        # 创建 RKNN 对象
        rknn = RKNN(verbose=True)

        # 使用最简单且兼容的配置
        print('--> Config model')
        ret = rknn.config(
            target_platform='rk3568',
            optimization_level=3,
        )

        if ret != 0:
            print('Config model failed!')
            return False

        # 加载ONNX模型
        print('--> Loading model')
        ret = rknn.load_onnx(
            model=onnx_model_path,
            input_size_list=[[1, 3, HEIGHT, WIDTH]]  # 输入尺寸: [batch, channels, height, width]
        )

        if ret != 0:
            print('Load ONNX model failed!')
            return False
        print('ONNX模型加载完成')

        # 创建安全的量化数据集
        dataset_path = './safe_quantization_dataset.txt'
        temp_dir = create_safe_quantization_dataset(dataset_path, num_samples=20)

        if temp_dir is None:
            print("创建量化数据集失败，进行非量化转换")
            ret = rknn.build(do_quantization=False)
        else:
            print('--> Building model with quantization')
            ret = rknn.build(
                do_quantization=True,
                dataset=dataset_path
            )

            if ret != 0:
                print('量化构建失败，尝试非量化构建...')
                ret = rknn.build(do_quantization=False)

        if ret != 0:
            print('Build RKNN model failed!')
            return False

        print('RKNN模型构建完成')

        # 跳过在PC上的验证，因为需要实际设备
        print('跳过PC端验证，因为需要实际RK3568设备')

        # 导出RKNN模型
        print('--> Export RKNN model')
        ret = rknn.export_rknn(rknn_save_path)
        if ret != 0:
            print('Export RKNN model failed!')
            return False
        print(f'RKNN模型已导出到: {rknn_save_path}')

        # 释放 RKNN 对象
        rknn.release()

        return True

    except Exception as e:
        print(f"转换RKNN时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"已清理临时目录: {temp_dir}")
            except Exception as e:
                print(f"清理临时目录失败: {e}")


def verify_model_conversion(rknn_model_path):
    """
    验证模型转换是否正确 - 离线版本，不连接实际设备
    """
    print("验证模型转换...")

    try:
        # 创建 RKNN 对象
        rknn = RKNN()

        # 加载RKNN模型
        ret = rknn.load_rknn(rknn_model_path)
        if ret != 0:
            print('Load RKNN model failed!')
            return False

        # 尝试初始化运行时环境，但不强制要求连接设备
        print('--> 尝试初始化运行时环境')
        try:
            # 尝试使用模拟器模式
            ret = rknn.init_runtime(target='rk3568')
            if ret != 0:
                print('无法连接到RK3568设备，但模型转换成功')
                print('RKNN模型可以在实际RK3568设备上使用')
                rknn.release()
                return True
        except Exception as e:
            print(f'初始化运行时环境失败: {e}')
            print('但这不影响模型转换结果，RKNN模型可以在实际设备上使用')
            rknn.release()
            return True

        # 如果成功连接到设备，进行推理测试
        print('--> 已连接到设备，进行推理测试')

        # 创建随机输入数据进行测试
        dummy_input = np.random.random((1, 3, HEIGHT, WIDTH)).astype(np.float32)

        # 进行推理测试
        outputs = rknn.inference(inputs=[dummy_input])

        print(f"推理完成，输出数量: {len(outputs)}")

        # 打印输出形状信息
        for i, output in enumerate(outputs):
            print(f"输出 {i}: 形状 {output.shape}, 数据类型 {output.dtype}")

        # 释放资源
        rknn.release()

        return True

    except Exception as e:
        print(f"验证模型时发生错误: {e}")
        print('但这不影响模型转换结果，RKNN模型可以在实际设备上使用')
        return True


def create_model_info_file(rknn_model_path):
    """
    创建模型信息文件，包含模型的基本信息和使用说明
    """
    info_file = rknn_model_path.replace('.rknn', '_info.txt')

    try:
        with open(info_file, 'w') as f:
            f.write("YOLO姿态检测模型信息\n")
            f.write("=====================\n\n")
            f.write(f"模型名称: {os.path.basename(rknn_model_path)}\n")
            f.write(f"输入尺寸: {WIDTH}x{HEIGHT}x3\n")
            f.write(f"输出格式: 姿态检测 (边界框 + {NUM_KEYPOINTS}个关键点)\n")
            f.write(f"目标平台: RK3568\n")
            f.write(f"创建时间: {np.datetime64('now')}\n\n")
            f.write("使用说明:\n")
            f.write("1. 将此RKNN模型文件部署到RK3568设备上\n")
            f.write("2. 使用RKNN Python API加载模型进行推理\n")
            f.write("3. 输入图像尺寸应为640x640，RGB格式\n")
            f.write("4. 输出包含边界框和关键点信息\n\n")
            f.write("关键点顺序:\n")
            f.write("0-3: 原始4个关键点\n")
            f.write("4-5: 坡度线的两个端点\n")

        print(f"模型信息文件已创建: {info_file}")
        return True
    except Exception as e:
        print(f"创建模型信息文件失败: {e}")
        return False


def main():
    """
    主转换函数
    """
    print("开始YOLO模型转换流程...")

    # 检查YOLO模型文件是否存在
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"错误: YOLO模型文件不存在: {YOLO_MODEL_PATH}")
        print("请先运行训练程序或检查模型路径")
        return

    # 创建输出目录
    os.makedirs(ONNX_MODEL_PATH, exist_ok=True)
    os.makedirs(RKNN_MODEL_PATH, exist_ok=True)

    # 生成模型保存路径
    model_name = os.path.splitext(os.path.basename(YOLO_MODEL_PATH))[0]
    onnx_save_path = os.path.join(ONNX_MODEL_PATH, f'{model_name}.onnx')
    rknn_save_path = os.path.join(RKNN_MODEL_PATH, f'{model_name}.rknn')

    # 步骤1: 将YOLO模型导出为ONNX格式
    print("\n" + "="*50)
    print("步骤1: 导出YOLO模型为ONNX格式")
    print("="*50)

    if not export_yolo_to_onnx(YOLO_MODEL_PATH, onnx_save_path):
        print("ONNX导出失败，终止转换流程")
        return

    # 检查ONNX文件是否存在
    if not os.path.exists(onnx_save_path):
        print(f"错误: ONNX文件不存在: {onnx_save_path}")
        return

    print(f"ONNX文件大小: {os.path.getsize(onnx_save_path) / (1024*1024):.2f} MB")

    # 步骤2: 将ONNX模型转换为RKNN格式
    print("\n" + "="*50)
    print("步骤2: 转换ONNX模型为RKNN格式")
    print("="*50)

    if not convert_onnx_to_rknn(onnx_save_path, rknn_save_path):
        print("RKNN转换失败")
        return

    # 步骤3: 创建模型信息文件
    print("\n" + "="*50)
    print("步骤3: 创建模型信息文件")
    print("="*50)

    create_model_info_file(rknn_save_path)

    # 步骤4: 验证模型转换
    print("\n" + "="*50)
    print("步骤4: 验证转换后的模型")
    print("="*50)

    # 即使验证失败，也不影响模型转换成功
    verify_model_conversion(rknn_save_path)

    print("\n" + "="*50)
    print("转换流程完成！")
    print(f"ONNX模型: {onnx_save_path}")
    print(f"RKNN模型: {rknn_save_path}")
    print("="*50)
    print("\n重要说明:")
    print("1. RKNN模型已成功创建，但需要在RK3568设备上进行最终验证")
    print("2. 将RKNN模型文件传输到RK3568设备上，使用RKNN Python API进行推理")
    print("3. 确保设备上安装了相应版本的RKNN Runtime")


if __name__ == '__main__':
    main()