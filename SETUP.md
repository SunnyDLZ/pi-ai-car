# 树莓派环境配置说明

本文档供安装在树莓派上的智能体读取并执行。目标：把 `pi-ai-car` 项目部署到一台全新树莓派 OS Bookworm 上，并启用全部功能 (含摄像头辅助避障 + 主人识别跟随)。

---

## 0. 前置条件检查

执行下面命令确认环境：

```bash
# 确认是树莓派
grep -E "Raspberry Pi|Model" /proc/cpuinfo | head -3

# 确认 OS 版本 (建议 Bookworm 64-bit)
cat /etc/os-release | grep -E "VERSION|NAME"

# 确认 git 已装
git --version || sudo apt install -y git

# 确认 Python 版本 (≥ 3.9)
python3 --version
```

如 OS 不是 Bookworm，部分 NetworkManager 命令在 wifi-ap-check.sh 中可能不可用 — Bullseye 及之前用 dhcpcd，需另外配置。

---

## 1. 拉取代码

```bash
cd ~
# 如果已有仓库目录先备份再 clone
if [ -d pi-ai-car ]; then mv pi-ai-car pi-ai-car.bak.$(date +%s); fi
git clone https://github.com/SunnyDLZ/pi-ai-car.git
cd pi-ai-car
```

---

## 2. 系统依赖安装 (apt)

```bash
sudo apt update
sudo apt install -y \
    python3-pip python3-dev python3-venv \
    build-essential cmake git \
    pigpio python3-pigpio \
    espeak espeak-data \
    libopenjp2-7 libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libatlas-base-dev gfortran \
    portaudio19-dev python3-pyaudio \
    wireless-tools network-manager
```

**说明**：
- `pigpio`: 舵机硬件 PWM 必需，使用前需 `sudo pigpiod` 启动守护进程
- `espeak`: 语音播报 (TTS)
- `cmake`/`build-essential`/`gfortran`: 编译 dlib 和 numpy/OpenCV 用
- `portaudio19-dev`/`python3-pyaudio`: USB 麦克风录音
- `libatlas-base-dev`: OpenCV/numpy 的线性代数后端

---

## 3. 启用 pigpio 守护进程 (舵机必需)

```bash
# 立即启动
sudo pigpiod

# 开机自启
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

# 验证
pgrep -a pigpiod && echo "pigpiod 已运行"
```

---

## 4. Python 依赖安装

建议用 venv 隔离：

```bash
cd ~/pi-ai-car
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 安装基础依赖 (避障、Web 控制、语音)
pip install -r requirements.txt

# 验证基础导入
python3 -c "import cv2, numpy, flask, RPi.GPIO, picamera2; print('基础依赖 OK')"
```

`requirements.txt` 应至少包含：
```
opencv-python
numpy
flask
picamera2
RPi.GPIO
pigpio
SpeechRecognition
pyaudio
```

如果 `requirements.txt` 不完整，按上面清单补装。

---

## 5. 摄像头配置 (CSI 模块)

```bash
# Bookworm 用 libcamera，无需 raspi-config 改 legacy camera
# 验证 CSI 摄像头被识别
libcamera-hello --list-cameras

# 如果未检测到，检查设备树
ls -la /sys/bus/i2c/devices/ | grep -E "imx|ov"
```

如果 `libcamera-hello` 找不到摄像头：
1. 关机检查 CSI 排线 (银色面朝向 HDMI 口一侧)
2. 重新开机进 `sudo raspi-config` → Interface Options → I2C 启用

---

## 6. 安装 dlib + face_recognition (跟随模式必需)

> **注意**: dlib 编译耗时较长 (树莓派 4B 约 30~40 分钟，5 约 15 分钟)。期间 CPU 满载、温度升高，确保散热。

```bash
# 先装编译依赖
sudo apt install -y cmake libdlib-dev libblas-dev liblapack-dev

# 用 pip 装 dlib (会从源码编译)
pip install dlib

# 验证 dlib
python3 -c "import dlib; print('dlib', dlib.__version__)"

# 装 face_recognition (封装层)
pip install face_recognition

# 验证
python3 -c "import face_recognition; print('face_recognition OK')"
```

**编译失败的常见处理**：
- 内存不足 (1GB/2GB 树莓派): 加 swap
  ```bash
  sudo swapoff -a
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon -a
  ```
  装完 dlib 后可删除 swap
- 编译器版本太老: `sudo apt install gcc-10 g++-10` 然后用 `CC=gcc-10 CXX=g++-10 pip install dlib`

---

## 7. 下载 dlib 人脸模型 (跟随模式必需)

```bash
cd ~/pi-ai-car/models

# 下载两个模型文件 (共约 30MB)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

# 解压
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2

# 验证 (两个文件都应有内容)
ls -lh *.dat
```

**预期输出**：
- `shape_predictor_68_face_landmarks.dat` 约 95MB
- `dlib_face_recognition_resnet_model_v1.dat` 约 30MB

如果 dlib.net 下载慢，可换镜像：
```bash
# 备用源 (GitHub mirror)
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
wget https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2
```

---

## 8. 验证全部模块就绪

```bash
cd ~/pi-ai-car
source .venv/bin/activate

# 预检查脚本 (不连电机/舵机硬件)
python3 -c "
import cv2, numpy, flask, RPi.GPIO, pigpio
print('基础依赖 OK')
try:
    import dlib, face_recognition
    print('人脸识别依赖 OK')
except ImportError as e:
    print('人脸识别依赖缺失 (跟随模式不可用):', e)
import os
for f in ['models/shape_predictor_68_face_landmarks.dat',
          'models/dlib_face_recognition_resnet_model_v1.dat']:
    print(f'{f}:', '存在' if os.path.exists(f) else '缺失')
"
```

---

## 9. 启动小车

```bash
cd ~/pi-ai-car
source .venv/bin/activate
sudo python3 main.py
```

**首次启动预期日志**：
```
[Servo] 云台舵机初始化完成
[Ultrasonic] 初始化完成
[CSICamera] 初始化完成 (640x480)
[AIVision] 模型文件缺失，跳过 AI 视觉初始化  ← 可忽略 (MobileNet 模型未下载)
[VisionObstacle] 视觉避障分析模块就绪
[FaceRecognizer] 初始化完成，已加载 0 个主人  ← dlib 装好后会出现这行
[Follower] 跟随线程启动
[WebServer] 控制面板: http://0.0.0.0:2222/
```

如 `[AIVision] 模型文件缺失` — 不影响避障 (vision_obstacle 是纯 OpenCV 的) 和跟随 (face_recognizer 是 dlib 的)。MobileNet 仅用于 web 端物体检测可视化，可选下载：
```bash
cd ~/pi-ai-car/models
# 从 GitHub 下载 MobileNet SSD 权重
wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.prototxt
wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel
```

---

## 10. 录入主人 (跟随模式必需)

1. 在同一局域网的电脑/手机浏览器打开 `http://<树莓派IP>:2222`
2. 滚到底部 "主人管理" 区域
3. 输入主人名字 (如 "Sunny") → 点 "注册"
4. 站到摄像头前 50~80cm 处，正脸 → 点 "采集"
5. 换角度 (左侧脸) → 点 "采集"
6. 再换角度 (右侧脸) → 点 "采集"
7. 看到 "(已就绪, 1人)" 后即可点 "👣跟随" 启动跟随

**采集建议**：
- 光线充足均匀 (避免逆光)
- 距离 50~80cm (脸在画面中占比适中)
- 三个角度差异要大 (正脸 + 左侧 30° + 右侧 30°)
- 同一人可注册多次以增强鲁棒性

---

## 11. WiFi AP 配置 (无路由器场景)

如果树莓派要在没有 WiFi 的户外使用，部署 AP 脚本：

```bash
cd ~/pi-ai-car
sudo cp scripts/wifi-ap-check.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/wifi-ap-check.sh

# 创建 systemd 服务
sudo tee /etc/systemd/system/wifi-ap-check.service > /dev/null <<'UNIT'
[Unit]
Description=WiFi Check & AP Fallback
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/wifi-ap-check.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable wifi-ap-check.service
```

启动后默认 SSID=`car`，密码=`raspberry`，IP 默认 `10.42.0.1`。手机连接后浏览器访问 `http://10.42.0.1:2222`。

---

## 12. 开机自启 (可选)

```bash
sudo tee /etc/systemd/system/pi-ai-car.service > /dev/null <<'UNIT'
[Unit]
Description=Pi AI Car Main Service
After=network-online.target pigpiod.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/pi-ai-car
ExecStart=/home/pi/pi-ai-car/.venv/bin/python3 /home/pi/pi-ai-car/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable pi-ai-car.service
sudo systemctl start pi-ai-car.service

# 查看日志
sudo journalctl -u pi-ai-car.service -f
```

---

## 故障排查

### pigpio 连接失败
```
RuntimeError: [Servo] 无法连接到 pigpio 守护进程！
```
解决: `sudo pigpiod` 启动守护进程

### CSI 摄像头初始化失败
```
[CSICamera] 初始化失败: ...
```
排查:
- `libcamera-hello --list-cameras` 看是否能列出
- 检查 CSI 排线
- Bookworm 不需要 `raspi-config` 启用 legacy camera

### 麦克风无声音
```
[VoiceInput] 麦克风错误: [Errno -9998] Invalid number of channels
```
解决:
```bash
arecord -l   # 列出录音设备
# 在 voice.py 的 sr.Microphone() 里指定 device_index=N
```

### dlib 编译卡死
增加 swap (见第 6 节) 或在更凉爽环境编译。也可尝试预编译 wheel：
```bash
pip install --only-binary :all: dlib
```

### 跟随模式按钮点不动
说明 `face_recognizer.is_ready()` 返回 False。检查：
1. dlib 装了吗: `python3 -c "import dlib"`
2. 模型文件在吗: `ls -lh ~/pi-ai-car/models/*.dat`
3. 主人库非空: 在 web 端注册并采集至少一个主人
