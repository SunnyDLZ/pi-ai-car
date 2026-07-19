"""
AI Car - 配置文件
树莓派 AI 小车主控配置
"""

# ========== GPIO 引脚定义 (BCM编号) ==========

# 电机 - L298N #1 (前轮)
# A路输出 → 右前轮 (FR)
MOTOR1_FR_IN1 = 17   # 右前轮 IN1
MOTOR1_FR_IN2 = 18   # 右前轮 IN2
MOTOR1_FR_ENA = 12   # 右前轮 ENA (PWM)
# B路输出 → 左前轮 (FL)
MOTOR1_FL_IN3 = 22   # 左前轮 IN3
MOTOR1_FL_IN4 = 23   # 左前轮 IN4
MOTOR1_FL_ENB = 13   # 左前轮 ENB (PWM)

# 电机 - L298N #2 (后轮)
# A路输出 → 右后轮 (RR)
MOTOR2_RR_IN1 = 24   # 右后轮 IN1
MOTOR2_RR_IN2 = 25   # 右后轮 IN2
MOTOR2_RR_ENA = 19   # 右后轮 ENA (PWM)
# B路输出 → 左后轮 (RL)
MOTOR2_RL_IN3 = 5    # 左后轮 IN3
MOTOR2_RL_IN4 = 6    # 左后轮 IN4
MOTOR2_RL_ENB = 26   # 左后轮 ENB (PWM)

# 超声波 HY-SRF05 (5针接口: 1=Vcc 2=Trig 3=Echo 4=OUT 5=GND)
# 注意: 实际模块上的 Vcc 是 5V，Echo 输出也是 5V，需分压后接 GPIO
ULTRASONIC_TRIG = 27   # ② Trig — 触发控制端 (GPIO.OUT)
ULTRASONIC_ECHO = 16   # ③ Echo — 回响信号 (GPIO.IN，需 5V→3.3V 分压)
# ULTRASONIC_OUT — ④ OUT 脚 (开关量输出，仅报警模式下使用，本系统未用)

# 云台舵机 (Pigpio 硬件 PWM)
SERVO_PAN_PIN = 20   # 水平 (左右)
SERVO_TILT_PIN = 21  # 俯仰 (上下)

# ========== 电机参数 ==========

# PWM 频率 (Hz)
MOTOR_PWM_FREQ = 1000

# 最大/最小占空比 (0-100)
MOTOR_SPEED_MIN = 20   # 起步最小速度
MOTOR_SPEED_MAX = 100  # 最大速度
MOTOR_SPEED_DEFAULT = 50  # 默认速度

# ========== 麦克纳姆轮运动学 ==========

# 麦克纳姆轮布局:
#          前
#    FL(前左)   FR(前右)
#    RL(后左)   RR(后右)
#          后
#
# 正速度 = 前进/顺时针 (从上往下看)
# 运动向量: [x, y, rotation]
#   x: 左右 (+右)
#   y: 前后 (+前)
#   rotation: 旋转 (+顺时针)

# ========== 超声波参数 (HY-SRF05 数据手册) ==========

# 声学参数
SONIC_SPEED = 34300  # 声速 cm/s (20°C)
MAX_DISTANCE = 450   # 最大量程 cm (手册标称 4.5m)
MIN_DISTANCE = 2     # 最小盲区 cm (2cm)
TIMEOUT_SEC = 0.027  # 超时 (对应 ~4.6m，略超最大量程)

# 电气参数
ULTRASONIC_VCC = 5       # 工作电压 5V DC
ULTRASONIC_CURRENT = 15  # 工作电流 15mA
ULTRASONIC_FREQ = 40     # 工作频率 40Hz
ULTRASONIC_ANGLE = 15    # 测量角度 15°
TRIG_PULSE_US = 10       # 触发脉冲宽度 10us
MODULE_SIZE = "45mm × 20mm × 15mm"  # 模块尺寸

# 避障阈值 (cm)
OBSTACLE_WARN = 30    # 预警距离
OBSTACLE_STOP = 15    # 停车距离

# ========== 舵机参数 ==========

SERVO_PAN_MIN = 0     # 水平范围 (度)
SERVO_PAN_MAX = 180
SERVO_PAN_CENTER = 90

SERVO_TILT_MIN = 0    # 俯仰范围 (度)
SERVO_TILT_MAX = 180
SERVO_TILT_CENTER = 90

# ========== 摄像头参数 ==========

CSI_FRAME_WIDTH = 640
CSI_FRAME_HEIGHT = 480
CSI_FRAME_RATE = 30

USB_FRAME_WIDTH = 640
USB_FRAME_HEIGHT = 480
USB_CAMERA_ID = 0  # /dev/video0

# ========== Web 服务 ==========

WEB_HOST = "0.0.0.0"
WEB_PORT = 2222

# ========== AI 视觉 ==========

# 物体检测置信度阈值
DETECTION_CONFIDENCE = 0.5

# MobileNet SSD 模型路径 (自动下载)
MOBILENET_PROTOTXT = "models/MobileNetSSD_deploy.prototxt"
MOBILENET_MODEL = "models/MobileNetSSD_deploy.caffemodel"

# VOC 类别标签 (MobileNet SSD 基于 PASCAL VOC 20 类 + background)
COCO_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]
