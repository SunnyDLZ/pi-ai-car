"""
motor.py - L298N 电机驱动 + 麦克纳姆轮运动学

支持四轮独立调速，全向运动。
驱动板: L298N x2 (每块驱动两个轮子)
"""

import RPi.GPIO as GPIO
from config import (
    MOTOR1_FL_IN1, MOTOR1_FL_IN2, MOTOR1_FL_ENA,
    MOTOR1_FR_IN3, MOTOR1_FR_IN4, MOTOR1_FR_ENB,
    MOTOR2_RL_IN1, MOTOR2_RL_IN2, MOTOR2_RL_ENA,
    MOTOR2_RR_IN3, MOTOR2_RR_IN4, MOTOR2_RR_ENB,
    MOTOR_PWM_FREQ, MOTOR_SPEED_MIN, MOTOR_SPEED_MAX,
    MOTOR_SPEED_DEFAULT
)


class MotorController:
    """四轮独立 L298N 电机控制器 + 麦克纳姆轮运动学"""

    def __init__(self):
        self._initialized = False
        self._pwm_channels = {}  # (pin, pwm_object)
        self._speed = MOTOR_SPEED_DEFAULT

    def init(self):
        """初始化 GPIO 和 PWM"""
        if self._initialized:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # 所有电机引脚
        motor_pins = [
            MOTOR1_FL_IN1, MOTOR1_FL_IN2, MOTOR1_FL_ENA,
            MOTOR1_FR_IN3, MOTOR1_FR_IN4, MOTOR1_FR_ENB,
            MOTOR2_RL_IN1, MOTOR2_RL_IN2, MOTOR2_RL_ENA,
            MOTOR2_RR_IN3, MOTOR2_RR_IN4, MOTOR2_RR_ENB,
        ]

        # 设置 GPIO 方向
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # 初始化 PWM (ENA/ENB 脚)
        pwm_pins = [
            MOTOR1_FL_ENA, MOTOR1_FR_ENB,
            MOTOR2_RL_ENA, MOTOR2_RR_ENB,
        ]
        for pin in pwm_pins:
            pwm = GPIO.PWM(pin, MOTOR_PWM_FREQ)
            pwm.start(0)
            self._pwm_channels[pin] = pwm

        self._initialized = True
        print("[Motor] 初始化完成")

    def stop(self):
        """紧急停止所有电机"""
        self._set_motor(MOTOR1_FL_IN1, MOTOR1_FL_IN2, MOTOR1_FL_ENA, 0)
        self._set_motor(MOTOR1_FR_IN3, MOTOR1_FR_IN4, MOTOR1_FR_ENB, 0)
        self._set_motor(MOTOR2_RL_IN1, MOTOR2_RL_IN2, MOTOR2_RL_ENA, 0)
        self._set_motor(MOTOR2_RR_IN3, MOTOR2_RR_IN4, MOTOR2_RR_ENB, 0)

    def _set_motor(self, in1, in2, ena, speed_pct):
        """设置单个电机的方向和速度

        Args:
            in1, in2: 方向引脚
            ena: PWM 使能引脚
            speed_pct: -100~100, 正=正转, 负=反转, 0=停止
        """
        speed_pct = max(-100, min(100, speed_pct))

        if speed_pct > 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif speed_pct < 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
            speed_pct = -speed_pct
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        # 低于最小值时直接关闭 (避免电机嗡嗡叫)
        pwm_value = speed_pct
        if 0 < pwm_value < MOTOR_SPEED_MIN:
            pwm_value = 0

        if ena in self._pwm_channels:
            self._pwm_channels[ena].ChangeDutyCycle(pwm_value)

    def set_speed(self, speed_pct):
        """设置全局速度比例"""
        self._speed = max(MOTOR_SPEED_MIN, min(MOTOR_SPEED_MAX, speed_pct))

    def get_speed(self):
        return self._speed

    def _apply_to_all(self, fl, fr, rl, rr):
        """同时对四个轮子施加速度

        麦克纳姆轮映射:
          FL(前左)   FR(前右)
          RL(后左)   RR(后右)
        """
        self._set_motor(MOTOR1_FL_IN1, MOTOR1_FL_IN2, MOTOR1_FL_ENA, fl)
        self._set_motor(MOTOR1_FR_IN3, MOTOR1_FR_IN4, MOTOR1_FR_ENB, fr)
        self._set_motor(MOTOR2_RL_IN1, MOTOR2_RL_IN2, MOTOR2_RL_ENA, rl)
        self._set_motor(MOTOR2_RR_IN3, MOTOR2_RR_IN4, MOTOR2_RR_ENB, rr)

    def _normalize(self, speeds):
        """将各轮速度限制在 [-100, 100] 范围内，保持比例"""
        max_spd = max(abs(s) for s in speeds)
        if max_spd > 100:
            speeds = [int(s * 100 / max_spd) for s in speeds]
        return speeds

    # ============== 麦克纳姆轮运动接口 ==============

    def move(self, x=0, y=0, rotation=0):
        """全向移动

        Args:
            x:   -100~100, 横向 (+右)
            y:   -100~100, 纵向 (+前)
            rotation: -100~100, 旋转 (+顺时针)
        """
        # 麦克纳姆轮运动学公式:
        # FL =  y + x + rotation
        # FR =  y - x - rotation
        # RL =  y - x + rotation
        # RR =  y + x - rotation
        fl =  y + x + rotation
        fr =  y - x - rotation
        rl =  y - x + rotation
        rr =  y + x - rotation

        fl, fr, rl, rr = self._normalize([fl, fr, rl, rr])

        # 应用速度缩放
        scale = self._speed / 100.0
        fl = int(fl * scale)
        fr = int(fr * scale)
        rl = int(rl * scale)
        rr = int(rr * scale)

        self._apply_to_all(fl, fr, rl, rr)

    # ============== 便捷方向接口 ==============

    def forward(self, speed=None):
        """前进"""
        if speed is not None:
            self.set_speed(speed)
        self.move(y=100)

    def backward(self, speed=None):
        """后退"""
        if speed is not None:
            self.set_speed(speed)
        self.move(y=-100)

    def strafe_left(self, speed=None):
        """左移"""
        if speed is not None:
            self.set_speed(speed)
        self.move(x=-100)

    def strafe_right(self, speed=None):
        """右移"""
        if speed is not None:
            self.set_speed(speed)
        self.move(x=100)

    def rotate_left(self, speed=None):
        """左旋转"""
        if speed is not None:
            self.set_speed(speed)
        self.move(rotation=-60)

    def rotate_right(self, speed=None):
        """右旋转"""
        if speed is not None:
            self.set_speed(speed)
        self.move(rotation=60)

    def forward_left(self, speed=None):
        """前进+左移 (斜向)"""
        if speed is not None:
            self.set_speed(speed)
        self.move(x=-50, y=100)

    def forward_right(self, speed=None):
        """前进+右移 (斜向)"""
        if speed is not None:
            self.set_speed(speed)
        self.move(x=50, y=100)

    def backward_left(self, speed=None):
        """后退+左移"""
        if speed is not None:
            self.set_speed(speed)
        self.move(x=-50, y=-100)

    def backward_right(self, speed=None):
        """后退+右移"""
        if speed is not None:
            self.set_speed(speed)
        self.move(x=50, y=-100)

    def cleanup(self):
        """释放 GPIO 资源"""
        self.stop()
        for pwm in self._pwm_channels.values():
            pwm.stop()
        self._initialized = False
        print("[Motor] 资源已释放")
