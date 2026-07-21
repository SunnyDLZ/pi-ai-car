"""
motor.py - L298N 电机驱动 + 麦克纳姆轮运动学

支持四轮独立调速，全向运动。
驱动板: L298N x2 (每块驱动两个轮子)
"""

import threading
import RPi.GPIO as GPIO
from config import (
    MOTOR1_FR_IN1, MOTOR1_FR_IN2, MOTOR1_FR_ENA,
    MOTOR1_FL_IN3, MOTOR1_FL_IN4, MOTOR1_FL_ENB,
    MOTOR2_RR_IN1, MOTOR2_RR_IN2, MOTOR2_RR_ENA,
    MOTOR2_RL_IN3, MOTOR2_RL_IN4, MOTOR2_RL_ENB,
    MOTOR_PWM_FREQ, MOTOR_SPEED_MIN, MOTOR_SPEED_MAX,
    MOTOR_SPEED_DEFAULT
)


class MotorController:
    """四轮独立 L298N 电机控制器 + 麦克纳姆轮运动学"""

    def __init__(self):
        self._initialized = False
        self._pwm_channels = {}  # (pin, pwm_object)
        self._speed = MOTOR_SPEED_DEFAULT
        self._lock = threading.Lock()  # 防止多线程同时控制电机

    def init(self):
        """初始化 GPIO 和 PWM"""
        if self._initialized:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # 所有电机引脚
        motor_pins = [
            MOTOR1_FR_IN1, MOTOR1_FR_IN2, MOTOR1_FR_ENA,
            MOTOR1_FL_IN3, MOTOR1_FL_IN4, MOTOR1_FL_ENB,
            MOTOR2_RR_IN1, MOTOR2_RR_IN2, MOTOR2_RR_ENA,
            MOTOR2_RL_IN3, MOTOR2_RL_IN4, MOTOR2_RL_ENB,
        ]

        # 设置 GPIO 方向
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # 初始化 PWM (ENA/ENB 脚)
        pwm_pins = [
            MOTOR1_FR_ENA, MOTOR1_FL_ENB,
            MOTOR2_RR_ENA, MOTOR2_RL_ENB,
        ]
        for pin in pwm_pins:
            pwm = GPIO.PWM(pin, MOTOR_PWM_FREQ)
            pwm.start(0)
            self._pwm_channels[pin] = pwm

        # 初始化完成后强制停止一次，确保电机上电后处于静止状态
        # (避免 GPIO 重新初始化时 L298N 使能端瞬间电平导致电机误启动)
        self.stop()

        self._initialized = True
        print("[Motor] 初始化完成")

    def stop(self):
        """紧急停止所有电机

        清除方向引脚 (IN1/IN2=LOW) 并将 PWM 占空比设为 0。
        不销毁/重建 PWM 对象，避免 RPi.GPIO 软件 PWM 重建时的时序问题
        导致某些轮子（尤其是硬件 PWM 引脚 BCM12）启动延迟或失效。
        """
        if not self._initialized:
            return
        with self._lock:
            for in1, in2, ena in [
                (MOTOR1_FR_IN1, MOTOR1_FR_IN2, MOTOR1_FR_ENA),
                (MOTOR1_FL_IN3, MOTOR1_FL_IN4, MOTOR1_FL_ENB),
                (MOTOR2_RR_IN1, MOTOR2_RR_IN2, MOTOR2_RR_ENA),
                (MOTOR2_RL_IN3, MOTOR2_RL_IN4, MOTOR2_RL_ENB),
            ]:
                GPIO.output(in1, GPIO.LOW)
                GPIO.output(in2, GPIO.LOW)
                if ena in self._pwm_channels:
                    self._pwm_channels[ena].ChangeDutyCycle(0)

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

        # 低于最小值时 clamp 到最小值 (而非清零)
        # 之前 bug: 清零会破坏麦轮运动学 — 低速时各轮占空比 < 20% 被清零，
        # 导致斜向移动/小旋转失效 (用户反馈: 速度 30% 以下按左转右转没反应)。
        # clamp 到 MIN 保证电机能转，麦轮运动学合成方向正确。
        if 0 < speed_pct < MOTOR_SPEED_MIN:
            speed_pct = MOTOR_SPEED_MIN

        # 只用 ChangeDutyCycle，不销毁/重建 PWM 对象
        # (RPi.GPIO 软件 PWM 重建有时序问题，尤其 BCM12 硬件 PWM 引脚)
        if ena in self._pwm_channels:
            self._pwm_channels[ena].ChangeDutyCycle(speed_pct)

    def set_speed(self, speed_pct):
        """设置全局速度比例

        允许 0 值 (急停场景)。非 0 值不再强制下限到 MOTOR_SPEED_MIN —
        低速时由 _set_motor 内部 clamp 到 MIN 保证电机能转，
        而不是在这里把速度本身提到 20% (那样 set_speed(10) 实际跑 20%，违反用户预期)。
        """
        self._speed = max(0, min(MOTOR_SPEED_MAX, speed_pct))

    def get_speed(self):
        return self._speed

    def _apply_to_all(self, fl, fr, rl, rr):
        """同时对四个轮子施加速度

        麦克纳姆轮映射:
          FL(前左)   FR(前右)
          RL(后左)   RR(后右)

        使用标准麦克纳姆轮运动学公式 (move() 中计算) 决定各轮目标方向。
        RL(左后轮) 电机物理接线极性与其他轮相反, 故在调用 _set_motor
        时对其速度值取反 (硬件层修正), 仅影响 RL 轮的物理转向,
        不改变运动学语义, 平移/旋转逻辑均正确。
        """
        self._set_motor(MOTOR1_FR_IN1, MOTOR1_FR_IN2, MOTOR1_FR_ENA, fr)
        self._set_motor(MOTOR1_FL_IN3, MOTOR1_FL_IN4, MOTOR1_FL_ENB, fl)
        self._set_motor(MOTOR2_RR_IN1, MOTOR2_RR_IN2, MOTOR2_RR_ENA, rr)
        # RL 物理接线极性相反 → 速度值取反 (硬件层修正, 不影响运动学)
        self._set_motor(MOTOR2_RL_IN3, MOTOR2_RL_IN4, MOTOR2_RL_ENB, -rl)

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
        if not self._initialized:
            return

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

        with self._lock:
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
        # rotation 用满幅 100，确保低速时各轮占空比 = speed% >= MOTOR_SPEED_MIN(20%)
        # 否则 rotation=-60 + speed=20% → 单轮 12% < 20% 会被 _set_motor 过滤为 0
        self.move(rotation=-100)

    def rotate_right(self, speed=None):
        """右旋转"""
        if speed is not None:
            self.set_speed(speed)
        self.move(rotation=100)

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
        self._pwm_channels.clear()
        self._initialized = False
        print("[Motor] 资源已释放")
