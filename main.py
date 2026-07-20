"""
main.py - AI 小车主程序入口

集成了:
  - 麦克纳姆轮全向移动
  - CSI 摄像头 + 云台舵机
  - 超声波避障
  - Web 遥控界面
  - AI 视觉识别
  - 语音控制

使用方式:
  python3 main.py
  # 然后浏览器访问 http://<树莓派IP>:2222
"""

import signal
import sys
import time
import threading

from motor import MotorController
from servo import ServoGimbal
from ultrasonic import Ultrasonic
from camera import CSICamera
from voice import VoiceOutput, VoiceInput
from ai_vision import AIVision
from web_server import WebServer
from config import OBSTACLE_WARN, OBSTACLE_SLOW, OBSTACLE_STOP, \
    AUTO_MAX_SPEED, AUTO_SLOW_SPEED, WEB_PORT


class AICar:
    """AI 小车主控"""

    def __init__(self):
        self.motor = MotorController()
        self.servo = ServoGimbal()
        self.ultrasonic = Ultrasonic()
        self.camera_csi = CSICamera()
        self.voice_out = VoiceOutput()
        self.voice_in = VoiceInput()
        self.vision = AIVision()

        self.web = None
        self._running = False
        self._auto_mode = False
        self._mode = "manual"  # manual / auto / voice
        self._mode_lock = threading.Lock()
        self._saved_user_speed = None  # 进入 auto 前保存的用户速度

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def init_all(self):
        """初始化所有模块"""
        print("=" * 40)
        print("   AI 小车系统启动中...")
        print("=" * 40)

        # 电机
        try:
            self.motor.init()
        except Exception as e:
            print(f"[!] 电机初始化失败: {e}")

        # 超声波
        try:
            self.ultrasonic.init()
        except Exception as e:
            print(f"[!] 超声波初始化失败: {e}")

        # 舵机
        try:
            self.servo.init()
        except Exception as e:
            print(f"[!] 舵机初始化失败: {e}")

        # CSI 摄像头
        try:
            if self.camera_csi.init():
                self.camera_csi.start()
        except Exception as e:
            print(f"[!] CSI 摄像头初始化失败: {e}")

        # AI 视觉
        try:
            self.vision.init()
        except Exception as e:
            print(f"[!] AI 视觉初始化失败: {e}")

        # 语音
        try:
            self.voice_out.init()
        except Exception as e:
            print(f"[!] 语音输出初始化失败: {e}")

        try:
            self.voice_in.init()
        except Exception as e:
            print(f"[!] 语音输入初始化失败: {e}")

        # Web 服务器
        self.web = WebServer(
            motor=self.motor,
            servo=self.servo,
            camera_csi=self.camera_csi,
            ultrasonic=self.ultrasonic,
            vision=self.vision,
            on_mode_change=self.set_mode,
        )
        self.web.start()

        self._running = True
        print("=" * 40)
        print("   ✅ 所有模块初始化完成！")
        print(f"   🌐 打开浏览器访问本机 {WEB_PORT} 端口")
        print("=" * 40)

    def get_mode(self):
        """获取当前模式 (线程安全)"""
        with self._mode_lock:
            return self._mode

    def set_mode(self, mode):
        """切换运行模式 (线程安全，供 WebServer 回调调用)"""
        with self._mode_lock:
            prev_mode = self._mode
            self._mode = mode

            # 进入 auto: 保存用户速度，设为 AUTO_MAX_SPEED (30%)
            # auto-pilot 内 y=100 即满速 → 实际输出 100*30/100 = 30%
            if mode == "auto" and prev_mode != "auto":
                self._saved_user_speed = self.motor.get_speed()
                self.motor.set_speed(AUTO_MAX_SPEED)
            # 退出 auto: 恢复用户速度
            elif mode != "auto" and prev_mode == "auto":
                if self._saved_user_speed is not None:
                    self.motor.set_speed(self._saved_user_speed)
                    self._saved_user_speed = None

        print(f"[Main] 切换到模式: {mode}")
        if mode == "voice":
            # 语音模式无自动控制，立即停车等待语音指令
            # (否则切换前手动方向指令会一直保持运动)
            self.motor.stop()
            self.voice_out.say("语音模式已开启")
        elif mode == "auto":
            self.voice_out.say("自动模式已开启")
        else:
            self.motor.stop()

    def _auto_pilot_loop(self):
        """自动避障巡游模式

        电机 speed 设为 AUTO_MAX_SPEED (30%)，y 值=100 表示满速。
        实际占空比 = y * speed / 100:
          dist >= 50cm → y=100, 实际 30%
          30 <= dist < 50 → y=67, 实际 20%
          15 <= dist < 30 → y 从 67 线性减到 0, 实际 20%→0%
          dist < 15cm → 急停 + 后退(y=67=20%) + 转向
        """
        if not getattr(self.motor, "_initialized", False):
            print("[AutoPilot] 电机未初始化，跳过自动巡游线程")
            return
        if not getattr(self.ultrasonic, "_initialized", False):
            print("[AutoPilot] 超声波未初始化，跳过自动巡游线程")
            return

        # y=100 对应 speed% (30%), y=67 对应 ~20%
        Y_FULL = 100   # → 实际 30%
        Y_SLOW = 67    # → 实际 ~20%

        print("[AutoPilot] 自动巡游启动")
        while self._running:
            if self.get_mode() != "auto":
                time.sleep(0.5)
                continue

            dist = self.ultrasonic.measure()
            if dist < 0:
                time.sleep(0.1)
                continue

            if dist < OBSTACLE_STOP:
                # 太近 → 急停 + 后退 + 转向
                self.motor.stop()
                self.voice_out.say("前方障碍", lang="zh")
                # 避障动作间隔较长，每步前检查模式，避免切手动后仍继续后退/转向
                if self.get_mode() != "auto":
                    continue
                self.motor.move(y=-Y_SLOW)  # 后退 20%
                time.sleep(0.5)
                if self.get_mode() != "auto":
                    self.motor.stop()
                    continue
                self.motor.move(rotation=Y_SLOW)  # 转向 20%
                time.sleep(0.3)
            elif dist < OBSTACLE_SLOW:
                # 15~30cm → 线性减速 (y: 67→0, 实际 20%→0%)
                ratio = (dist - OBSTACLE_STOP) / (OBSTACLE_SLOW - OBSTACLE_STOP)
                y_val = int(Y_SLOW * ratio)
                print(f"[AutoPilot] 前方 {dist:.0f}cm → 减速 y={y_val} ({int(y_val*AUTO_MAX_SPEED/100)}%)")
                self.motor.move(y=y_val)
            elif dist < OBSTACLE_WARN:
                # 30~50cm → 固定 20% 慢速
                print(f"[AutoPilot] 前方 {dist:.0f}cm → 慢速 y={Y_SLOW} ({AUTO_SLOW_SPEED}%)")
                self.motor.move(y=Y_SLOW)
            else:
                # >= 50cm → 安全 30% 前进
                print(f"[AutoPilot] 前方 {dist:.0f}cm → 巡航 y={Y_FULL} ({AUTO_MAX_SPEED}%)")
                self.motor.move(y=Y_FULL)

            time.sleep(0.2)

    def _voice_control_loop(self):
        """语音控制循环"""
        print("[VoiceControl] 语音控制启动")
        # 不在开机时播报，进入语音模式时由 set_mode() 播报"语音模式已开启"

        while self._running:
            if self.get_mode() != "voice":
                time.sleep(0.5)
                continue

            text = self.voice_in.listen_once(timeout=5, phrase_timeout=3)
            if text is None:
                continue

            cmd = text.lower()

            if any(w in cmd for w in ["前进", "向前", "走"]):
                self.motor.forward()
                self.voice_out.say("前进")
            elif any(w in cmd for w in ["后退", "向后", "倒车"]):
                self.motor.backward()
                self.voice_out.say("后退")
            elif any(w in cmd for w in ["左转", "向左"]):
                self.motor.rotate_left()
                self.voice_out.say("左转")
            elif any(w in cmd for w in ["右转", "向右"]):
                self.motor.rotate_right()
                self.voice_out.say("右转")
            elif any(w in cmd for w in ["左移", "左侧"]):
                self.motor.strafe_left()
                self.voice_out.say("左移")
            elif any(w in cmd for w in ["右移", "右侧"]):
                self.motor.strafe_right()
                self.voice_out.say("右移")
            elif any(w in cmd for w in ["停止", "停", "刹车", "别动"]):
                self.motor.stop()
                self.voice_out.say("已停止")
            elif any(w in cmd for w in ["速度", "加速", "快一点", "快点"]):
                speed = min(100, self.motor.get_speed() + 10)
                self.motor.set_speed(speed)
                self.voice_out.say(f"速度已到{speed}")
            elif any(w in cmd for w in ["减速", "慢一点", "慢点"]):
                speed = max(20, self.motor.get_speed() - 10)
                self.motor.set_speed(speed)
                self.voice_out.say(f"速度已到{speed}")
            elif "归中" in cmd or "复位" in cmd:
                self.servo.center()
                self.voice_out.say("云台已归中")
            elif any(w in cmd for w in ["手动", "遥控"]):
                self.set_mode("manual")
                self.voice_out.say("切换为手动模式")
            elif any(w in cmd for w in ["自动", "巡航", "巡游"]):
                self.set_mode("auto")
            else:
                self.voice_out.say("没听清指令")

            time.sleep(0.3)

    def run(self):
        """启动主循环"""
        self.init_all()

        # 启动后台线程
        auto_thread = threading.Thread(target=self._auto_pilot_loop, daemon=True)
        auto_thread.start()

        voice_thread = threading.Thread(target=self._voice_control_loop, daemon=True)
        voice_thread.start()

        print("\n💡 使用提示:")
        print(f"   浏览器打开 http://<树莓派IP>:{WEB_PORT} 进入控制台")
        print("   按 Ctrl+C 安全退出")

        # 主线程保持运行
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def _signal_handler(self, signum, frame):
        """信号处理 (Ctrl+C)"""
        print("\n[Main] 收到关闭信号...")
        self._running = False

    def cleanup(self):
        """安全释放所有资源"""
        print("\n[Main] 正在关闭系统...")
        if getattr(self.motor, "_initialized", False):
            self.motor.stop()
            self.motor.cleanup()
        if getattr(self.servo, "_initialized", False):
            self.servo.cleanup()
        if getattr(self.ultrasonic, "_initialized", False):
            self.ultrasonic.cleanup()
        self.camera_csi.cleanup()
        # 清理 GPIO (由 motor/ultrasonic 共用)
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except Exception:
            pass
        # 仅在语音输出已初始化时才播报
        try:
            if self.voice_out._tts_engine:
                self.voice_out.say("小车已关机")
                time.sleep(0.5)
        except Exception:
            pass
        print("[Main] 系统已安全关闭 ✅")


if __name__ == "__main__":
    car = AICar()
    car.run()
