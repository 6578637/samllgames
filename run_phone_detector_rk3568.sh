#!/bin/bash

# RK3568手机屏幕检测程序启动脚本
# 功能：在RK3568开发板上运行手机屏幕检测程序，支持开机自启动

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="phone-screen-detector"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_FILE="/var/log/${SERVICE_NAME}.log"

show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  run          直接运行程序（默认）"
    echo "  install      安装开机自启动服务"
    echo "  uninstall    卸载开机自启动服务"
    echo "  start        启动服务"
    echo "  stop         停止服务"
    echo "  status       查看服务状态"
    echo "  logs         查看服务日志"
    echo "  help         显示此帮助信息"
}

check_environment() {
    echo "=== 环境检查 ==="
    echo "当前时间: $(date)"
    echo "工作目录: $(pwd)"
    echo "脚本目录: $SCRIPT_DIR"

    # 检查Python环境
    echo "检查Python环境..."
    python3 --version
    if [ $? -ne 0 ]; then
        echo "错误: Python3未安装"
        return 1
    fi

    # 检查RKNN Toolkit Lite是否安装
    echo "检查RKNN Toolkit Lite..."
    python3 -c "from rknnlite.api import RKNNLite" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "警告: RKNN Toolkit Lite未安装，程序可能无法正常运行"
        echo "请安装: pip3 install rknn-toolkit-lite"
    fi

    # 检查OpenCV是否安装
    echo "检查OpenCV..."
    python3 -c "import cv2" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "警告: OpenCV未安装，程序可能无法正常运行"
        echo "请安装: pip3 install opencv-python"
    fi

    # 检查ADB连接
    echo "检查ADB连接..."
    adb devices
    if [ $? -ne 0 ]; then
        echo "错误: ADB未安装或未找到设备"
        echo "请确保:"
        echo "1. ADB已安装 (sudo apt install adb)"
        echo "2. 手机已通过USB连接并启用USB调试"
        echo "3. 已授权计算机调试权限"
        return 1
    fi

    # 检查模型文件
    MODEL_FILE="best.rknn"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "警告: 模型文件 $MODEL_FILE 不存在"
        echo "请确保已将转换后的RKNN模型文件放在当前目录"
        echo "可以使用 --model 参数指定其他模型文件路径"
    fi

    # 检查Python程序文件
    PYTHON_FILE="rk3568_phone_detector_rknn.py"
    if [ ! -f "$PYTHON_FILE" ]; then
        echo "错误: Python程序文件 $PYTHON_FILE 不存在"
        return 1
    fi

    return 0
}

run_program() {
    echo "=== 启动手机屏幕检测程序 ==="
    echo "按 Ctrl+C 停止程序"
    
    # 检查环境
    if ! check_environment; then
        echo "环境检查失败，无法启动程序"
        exit 1
    fi

    # 使用默认模型文件运行
    python3 rk3568_phone_detector_rknn.py

    # 如果默认模型文件不存在，提示用户指定
    if [ $? -ne 0 ] && [ ! -f "best.rknn" ]; then
        echo "请使用以下命令指定模型文件路径:"
        echo "python3 rk3568_phone_detector_rknn.py --model /path/to/your/model.rknn"
    fi

    echo "程序已退出"
}

install_service() {
    echo "=== 安装开机自启动服务 ==="
    
    # 检查root权限
    if [ "$EUID" -ne 0 ]; then
        echo "错误: 需要root权限来安装系统服务"
        echo "请使用: sudo $0 install"
        exit 1
    fi

    # 创建服务文件
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Phone Screen Detector Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
ExecStart=/bin/bash $SCRIPT_DIR/run_phone_detector_rk3568.sh run
Restart=always
RestartSec=10
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE

[Install]
WantedBy=multi-user.target
EOF

    # 重新加载systemd配置
    systemctl daemon-reload
    
    # 启用服务
    systemctl enable $SERVICE_NAME
    
    echo "服务安装成功！"
    echo "服务文件: $SERVICE_FILE"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "使用以下命令管理服务:"
    echo "  sudo $0 start    # 启动服务"
    echo "  sudo $0 stop     # 停止服务"
    echo "  sudo $0 status   # 查看服务状态"
    echo "  sudo $0 logs     # 查看服务日志"
}

uninstall_service() {
    echo "=== 卸载开机自启动服务 ==="
    
    # 检查root权限
    if [ "$EUID" -ne 0 ]; then
        echo "错误: 需要root权限来卸载系统服务"
        echo "请使用: sudo $0 uninstall"
        exit 1
    fi

    # 停止服务
    systemctl stop $SERVICE_NAME 2>/dev/null
    
    # 禁用服务
    systemctl disable $SERVICE_NAME 2>/dev/null
    
    # 删除服务文件
    if [ -f "$SERVICE_FILE" ]; then
        rm -f "$SERVICE_FILE"
        echo "已删除服务文件: $SERVICE_FILE"
    fi
    
    # 重新加载systemd配置
    systemctl daemon-reload
    
    echo "服务卸载成功！"
}

start_service() {
    if [ "$EUID" -ne 0 ]; then
        echo "错误: 需要root权限来启动服务"
        echo "请使用: sudo $0 start"
        exit 1
    fi
    
    systemctl start $SERVICE_NAME
    echo "服务启动命令已发送"
}

stop_service() {
    if [ "$EUID" -ne 0 ]; then
        echo "错误: 需要root权限来停止服务"
        echo "请使用: sudo $0 stop"
        exit 1
    fi
    
    systemctl stop $SERVICE_NAME
    echo "服务停止命令已发送"
}

status_service() {
    if [ "$EUID" -ne 0 ]; then
        echo "错误: 需要root权限来查看服务状态"
        echo "请使用: sudo $0 status"
        exit 1
    fi
    
    systemctl status $SERVICE_NAME
}

show_logs() {
    if [ "$EUID" -ne 0 ]; then
        echo "错误: 需要root权限来查看服务日志"
        echo "请使用: sudo $0 logs"
        exit 1
    fi
    
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "日志文件不存在: $LOG_FILE"
        echo "服务可能尚未运行或日志尚未生成"
    fi
}

# 主程序
case "${1:-run}" in
    "run")
        run_program
        ;;
    "install")
        install_service
        ;;
    "uninstall")
        uninstall_service
        ;;
    "start")
        start_service
        ;;
    "stop")
        stop_service
        ;;
    "status")
        status_service
        ;;
    "logs")
        show_logs
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "未知选项: $1"
        show_help
        exit 1
        ;;
esac
