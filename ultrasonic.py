"""
ultrasonic.py - HY-SRF05 超声波测距模块

HY-SRF05 参数:
  - 工作电压: 5V (ECHO 输出 5V，需分压降为 3.3V 接 GPIO)
  - 量程: 2cm ~ 400cm
  - 精度: 3mm
  - 触发: 10us TTL 脉冲
  - 回响: 100~25000us 脉冲 (对应距离)
"""

import time
import RPi.GPIO as GPIO
from config import ULTRASONIC_TRIG, ULTRASONIC_ECHO, \
    MAX_DISTANCE, MIN_DISTANCE, SONIC_SPEED, TIMEOUT_SEC


class Ultrasonic:
    """HY-SRF05 超声波测距"""

    def __init__(self):
        self._initialized = False

    def init(self):
        if self._initialized:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)
        self._initialized = True
        print("[Ultrasonic] 初始化完成")

    def measure(self, samples=5):
        """测量距离 (厘米)

        多次采样取中位数，去除异常值。

        Args:
            samples: 采样次数 (默认 5)

        Returns:
            float: 距离 (cm)，超时返回 -1
        """
        distances = []
        for _ in range(samples):
            d = self._single_measure()
            if MIN_DISTANCE <= d <= MAX_DISTANCE:
                distances.append(d)
            time.sleep(0.01)

        if not distances:
            return -1

        # 排序后取中位数
        distances.sort()
        return distances[len(distances) // 2]

    def _single_measure(self):
        """单次测距"""
        # 发送 10us 触发脉冲
        GPIO.output(ULTRASONIC_TRIG, GPIO.HIGH)
        time.sleep(0.00001)  # 10us
        GPIO.output(ULTRASONIC_TRIG, GPIO.LOW)

        # 等待 ECHO 变高 (等待回响开始)
        timeout_start = time.time()
        while GPIO.input(ULTRASONIC_ECHO) == GPIO.LOW:
            if time.time() - timeout_start > 0.02:  # 20ms 超时 (约 3.4m)
                return -1

        # 记录脉冲开始时间
        pulse_start = time.time()

        # 等待 ECHO 变低 (等待回响结束)
        while GPIO.input(ULTRASONIC_ECHO) == GPIO.HIGH:
            if time.time() - pulse_start > TIMEOUT_SEC:
                return -1

        pulse_end = time.time()

        # 计算距离 (d = t * v / 2)
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * SONIC_SPEED / 2

        return distance

    def obstacle_detected(self, threshold=30):
        """检测前方是否有障碍物

        Args:
            threshold: 障碍物阈值 (cm)

        Returns:
            bool: 是否有障碍物
            float: 当前距离
        """
        dist = self.measure()
        if dist < 0:
            return True, -1  # 超时视为有障碍物
        return dist < threshold, dist

    def cleanup(self):
        print("[Ultrasonic] 资源已释放")
