# RK3568手机屏幕检测系统部署指南

## 文件说明

### 核心文件
- `rk3568_phone_detector_rknn.py` - 主程序，使用RKNNLite进行手机屏幕检测和点击控制
- `best.rknn` - 转换后的RKNN模型文件
- `requirements.txt` - Python依赖包列表
- `run_phone_detector_rk3568.sh` - 启动脚本和系统服务管理

### 参考文件
- `phone_screen_test.py` - 原始PC端测试程序（含可视化）
- `convert_to_model.py` - 模型转换程序
- `train_vision_model_V3.py` - 模型训练程序

## 部署步骤

### 1. 环境准备
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y python3 python3-pip adb

# 安装Python依赖
pip3 install -r requirements.txt
```

### 2. 模型文件准备
确保 `best.rknn` 模型文件位于当前目录

### 3. 手机连接配置
```bash
# 检查ADB连接
adb devices

# 如果设备未授权，需要在手机上允许USB调试
```

### 4. 运行程序

#### 方式一：直接运行
```bash
# 直接运行程序
./run_phone_detector_rk3568.sh run
```

#### 方式二：安装为系统服务（开机自启动）
```bash
# 安装服务
sudo ./run_phone_detector_rk3568.sh install

# 启动服务
sudo ./run_phone_detector_rk3568.sh start

# 查看服务状态
sudo ./run_phone_detector_rk3568.sh status

# 查看日志
sudo ./run_phone_detector_rk3568.sh logs
```

### 5. 服务管理命令
```bash
# 启动服务
sudo ./run_phone_detector_rk3568.sh start

# 停止服务
sudo ./run_phone_detector_rk3568.sh stop

# 重启服务
sudo systemctl restart phone-screen-detector

# 查看服务状态
sudo ./run_phone_detector_rk3568.sh status

# 查看实时日志
sudo ./run_phone_detector_rk3568.sh logs

# 卸载服务
sudo ./run_phone_detector_rk3568.sh uninstall
```

## 程序功能说明

### 检测逻辑
- 使用YOLO关键点检测模型，检测6个关键点
- 4个原始关键点 + 2个坡度线关键点
- 角度计算：夹角 = -15度 + 前后轮中心和坡度的夹角

### 点击控制
- **油门位置**: (2417, 982) - 点击时长1秒
- **刹车位置**: (369, 991) - 点击时长0.5秒
- **状态机**: detecting（检测）、tapping（点击）、waiting（等待）

### 时序控制
- 检测和触摸不能同时进行
- 触摸维持后开始检测
- 精确控制点击时长

## 故障排除

### 常见问题
1. **ADB设备未连接**
   - 检查USB连接
   - 在手机上启用USB调试
   - 运行 `adb devices` 确认设备

2. **模型加载失败**
   - 确认 `best.rknn` 文件存在
   - 检查文件权限

3. **依赖包缺失**
   - 重新运行 `pip3 install -r requirements.txt`

4. **服务启动失败**
   - 检查日志：`sudo ./run_phone_detector_rk3568.sh logs`
   - 确认有root权限

### 日志查看
```bash
# 查看系统服务日志
sudo journalctl -u phone-screen-detector -f

# 查看程序日志文件
sudo tail -f /var/log/phone-screen-detector.log
```

## 性能优化建议

1. **CPU频率设置**
   ```bash
   # 查看当前频率
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
   
   # 设置性能模式
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **内存优化**
   - 关闭不必要的服务
   - 减少后台进程

3. **网络优化**
   - 确保稳定的网络连接
   - 避免网络延迟影响检测精度

## 安全注意事项

1. **权限管理**
   - 服务以root权限运行
   - 定期检查服务状态
   - 监控系统资源使用

2. **数据安全**
   - 定期备份模型文件
   - 监控程序运行状态
   - 设置合理的重启策略

## 版本信息
- 程序版本: 1.0
- 模型版本: best.rknn
- 适配平台: RK3568开发板
- 更新时间: 2025-10-24
