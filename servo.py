"""
servo.py - SG90 云台舵机控制 (Pan/Tilt)

使用 pigpio 硬件 PWM 实现平稳控制。
SG90 参数: 50Hz, 0.5ms(0°) ~ 2.5ms(180°) 脉宽
"""

import time
import pigpio
from config import SERVO_PAN_PIN, SERVO_TILT_PIN, \
    SERVO_PAN_MIN, SERVO_PAN_MAX, SERVO_PAN_CENTER, \
    SERVO_TILT_MIN, SERVO_TILT_MAX, SERVO_TILT_CENTER


class ServoGimbal:
    """双舵机云台控制"""

    def __init__(self):
        self._pi = None
        self._pan_angle = SERVO_PAN_CENTER
        self._tilt_angle = SERVO_TILT_CENTER
        self._initialized = False

    def init(self):
        """连接 pigpio 守护进程并初始化舵机"""
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError("[Servo] 无法连接到 pigpio 守护进程！请先启动: sudo pigpiod")

        # set_servo_pulsewidth 内部自行管理 PWM，无需手动设置 range/frequency

        # 归中
        self.pan(SERVO_PAN_CENTER)
        self.tilt(SERVO_TILT_CENTER)

        self._initialized = True
        print("[Servo] 云台舵机初始化完成")

    @staticmethod
    def _angle_to_pulse(angle, min_angle=0, max_angle=180):
        """将角度转换为脉宽 (0.1us 单位)

        SG90: 0° = 500us, 180° = 2500us
        范围 500~2500us = 5000~25000 (以 0.1us 为单位)
        """
        # 限制角度范围
        angle = max(min_angle, min(max_angle, angle))

        # 线性映射: 角度 → 脉宽 (us)
        pulse_us = 500 + (angle / 180.0) * 2000  # 500~2500us

        # set_servo_pulsewidth 接收微秒 (us)，无需转换
        return int(pulse_us)

    def pan(self, angle):
        """设置水平角度 (度)

        Args:
            angle: 0~180, 左~右
        """
        angle = max(SERVO_PAN_MIN, min(SERVO_PAN_MAX, angle))
        pulse = self._angle_to_pulse(angle)
        self._pi.set_servo_pulsewidth(SERVO_PAN_PIN, pulse)
        self._pan_angle = angle
        return angle

    def tilt(self, angle):
        """设置俯仰角度 (度)

        Args:
            angle: 0~180, 上~下
        """
        angle = max(SERVO_TILT_MIN, min(SERVO_TILT_MAX, angle))
        pulse = self._angle_to_pulse(angle)
        self._pi.set_servo_pulsewidth(SERVO_TILT_PIN, pulse)
        self._tilt_angle = angle
        return angle

    def get_angles(self):
        """获取当前舵机角度"""
        return self._pan_angle, self._tilt_angle

    def center(self):
        """云台归中"""
        self.pan(SERVO_PAN_CENTER)
        time.sleep(0.3)
        self.tilt(SERVO_TILT_CENTER)
        time.sleep(0.3)

    def scan(self, step=10, delay=0.05):
        """左右扫视一次

        Args:
            step: 步进角度
            delay: 每步延时 (秒)
        """
        for angle in range(SERVO_PAN_MIN, SERVO_PAN_MAX + 1, step):
            self.pan(angle)
            time.sleep(delay)
        for angle in range(SERVO_PAN_MAX, SERVO_PAN_MIN - 1, -step):
            self.pan(angle)
            time.sleep(delay)
        self.pan(SERVO_PAN_CENTER)

    def cleanup(self):
        """释放舵机 PWM"""
        if self._pi:
            self._pi.set_servo_pulsewidth(SERVO_PAN_PIN, 0)
            self._pi.set_servo_pulsewidth(SERVO_TILT_PIN, 0)
            self._pi.stop()
            self._pi = None
            print("[Servo] 舵机资源已释放")
