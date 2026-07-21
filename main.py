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
import time
import threading

from motor import MotorController
from servo import ServoGimbal
from ultrasonic import Ultrasonic
from camera import CSICamera
from voice import VoiceOutput, VoiceInput
from ai_vision import AIVision
from vision_obstacle import VisionObstacle
from face_recognizer import FaceRecognizer
from follower import Follower
from web_server import WebServer
from config import OBSTACLE_WARN, OBSTACLE_SLOW, OBSTACLE_STOP, \
    AUTO_MAX_SPEED, AUTO_SLOW_SPEED, WEB_PORT, VISION_SCAN_ANGLE, FOLLOW_SPEED


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
        self.vision_obs = VisionObstacle()
        self.face_recognizer = FaceRecognizer()
        self.follower = None  # 延迟创建，需要传入 self

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

        # 视觉避障分析 (纯 OpenCV, 无外部依赖)
        try:
            self.vision_obs.init()
        except Exception as e:
            print(f"[!] 视觉避障初始化失败: {e}")

        # 人脸识别 (dlib, 需手动安装+下载模型, 缺失则跟随功能不可用)
        try:
            self.face_recognizer.init()
        except Exception as e:
            print(f"[!] 人脸识别初始化失败: {e}")

        # 跟随控制器 (依赖 face_recognizer, 需在 face_recognizer.init 后创建)
        self.follower = Follower(self)
        self.follower.start()

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
            vision_obs=self.vision_obs,
            face_recognizer=self.face_recognizer,
            follower=self.follower,
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
        """切换运行模式 (线程安全，供 WebServer 回调调用)

        支持的模式:
          - manual: 手动遥控
          - auto:   超声波+视觉融合自动避障巡游
          - voice:  语音指令控制
          - follow: 主人跟随 (需人脸识别就绪 + 主人库非空)
        """
        if mode not in ("manual", "auto", "voice", "follow"):
            return

        # follow 模式前置检查: 人脸识别未就绪则拒绝 (返回不切，WebServer 也会因前置检查不调用到这里)
        if mode == "follow" and not self.face_recognizer.is_ready():
            print("[Main] 跟随模式不可用: 人脸识别未就绪 (未安装 dlib 或主人库为空)")
            self.voice_out.say("请先录入主人")
            return

        with self._mode_lock:
            prev_mode = self._mode
            self._mode = mode

            # 限速模式 (auto/follow) 速度保存/恢复策略:
            # - 从非限速模式进入限速模式: 保存用户原始速度
            # - 从限速模式退出到非限速模式: 恢复用户原始速度
            # - auto ↔ follow 之间切换: 不动 _saved_user_speed (避免被限速值覆盖丢失)
            # 之前 bug: auto→follow 时 prev_mode=="auto" 触发"进入 follow 保存速度"分支，
            # 把 _saved_user_speed 写成 AUTO_MAX_SPEED (30) 而非用户原始值，永久丢失
            LIMITED_MODES = ("auto", "follow")
            if mode in LIMITED_MODES and prev_mode not in LIMITED_MODES:
                self._saved_user_speed = self.motor.get_speed()
            elif mode not in LIMITED_MODES and prev_mode in LIMITED_MODES:
                if self._saved_user_speed is not None:
                    self.motor.set_speed(self._saved_user_speed)
                    self._saved_user_speed = None

            # 设置当前模式速度
            if mode == "auto":
                self.motor.set_speed(AUTO_MAX_SPEED)
            elif mode == "follow":
                self.motor.set_speed(FOLLOW_SPEED)

            # 所有模式切换都停车，避免上一模式残留的运动指令继续执行
            self.motor.stop()

        print(f"[Main] 切换到模式: {mode}")
        # 同步 WebServer._mode (语音切换模式时 web 端模式状态需保持一致)
        if self.web is not None:
            self.web.set_mode(mode)
        # 语音播报在锁外 (espeak 子进程启动慢，避免长时间持锁阻塞 auto-pilot)
        if mode == "voice":
            self.voice_out.say("语音模式已开启")
        elif mode == "auto":
            self.voice_out.say("自动模式已开启")
        elif mode == "follow":
            self.voice_out.say("跟随模式已开启")

    def _auto_pilot_loop(self):
        """自动避障巡游模式 (超声波 + 视觉融合)

        决策流程:
          1. 超声波测距 → 前方准确距离
          2. 摄像头帧 → 视觉通行性分析 (左/中/右)
          3. 融合决策:
             - dist < 15cm → 急停 + 后退 + 视觉推荐方向转向
             - 15~30cm 且中部阻塞 → 减速 + 用视觉推荐方向转向避障
             - 30~50cm → 慢速前进 (视觉中部阻塞时也减速)
             - >= 50cm 且中部畅通 → 巡航
             - >= 50cm 但中部阻塞 → 仍要避障 (视觉弥补超声波盲区)
          4. 转向前云台扫视一眼，提高避障成功率

        速度参数: speed 设为 AUTO_MAX_SPEED (30%)，y=100 → 实际 30%
        """
        if not getattr(self.motor, "_initialized", False):
            print("[AutoPilot] 电机未初始化，跳过自动巡游线程")
            return
        if not getattr(self.ultrasonic, "_initialized", False):
            print("[AutoPilot] 超声波未初始化，跳过自动巡游线程")
            return

        Y_FULL = 100   # → 实际 30%
        Y_SLOW = 67    # → 实际 ~20%

        print("[AutoPilot] 自动巡游启动 (视觉融合)")
        while self._running:
            if self.get_mode() != "auto":
                time.sleep(0.5)
                continue

            # === 1. 超声波测距 ===
            dist = self.ultrasonic.measure()
            if dist < 0:
                with self._mode_lock:
                    if self._mode == "auto":
                        self.motor.stop()
                print("[AutoPilot] 测距失败，已停车")
                time.sleep(0.1)
                continue

            # === 2. 视觉通行性分析 ===
            # 摄像头视角比超声波宽 (~60° vs 15°)，能感知左右障碍
            vision_info = self._analyze_vision()

            # === 3. 融合决策 ===
            say_obstacle = False
            retreat = False
            turn_dir = None  # "left"/"right"/None

            # 视觉判定中部是否阻塞 (超声波可能没测到)
            vision_center_blocked = vision_info["ok"] and vision_info["center_blocked"]

            with self._mode_lock:
                if self._mode != "auto":
                    continue

                if dist < OBSTACLE_STOP:
                    # 太近 → 急停 + 后退 + 转向 (用视觉推荐方向)
                    self.motor.stop()
                    say_obstacle = True
                    retreat = True
                    turn_dir = self._pick_turn_direction(vision_info)

                elif dist < OBSTACLE_SLOW or vision_center_blocked:
                    # 15~30cm 或视觉发现中部阻塞 → 避障转向
                    if vision_info["ok"]:
                        suggested = vision_info["suggested_dir"]
                        if suggested == "backward":
                            # 三面都堵 → 后退
                            self.motor.stop()
                            retreat = True
                            turn_dir = "right"  # 后退时随便选个方向
                        elif suggested == "center":
                            # 视觉说中部畅通 (但超声波说近) → 信任超声波减速
                            ratio = (dist - OBSTACLE_STOP) / (OBSTACLE_SLOW - OBSTACLE_STOP) if dist < OBSTACLE_SLOW else 1.0
                            y_val = int(Y_SLOW * ratio)
                            self.motor.move(y=y_val)
                        else:
                            # 视觉推荐左/右转
                            self.motor.stop()
                            turn_dir = suggested
                            say_obstacle = True
                    else:
                        # 视觉不可用 → 回退到旧的超声波减速逻辑
                        if dist < OBSTACLE_SLOW:
                            ratio = (dist - OBSTACLE_STOP) / (OBSTACLE_SLOW - OBSTACLE_STOP)
                            y_val = int(Y_SLOW * ratio)
                            self.motor.move(y=y_val)
                        else:
                            self.motor.move(y=Y_SLOW)

                elif dist < OBSTACLE_WARN:
                    # 30~50cm → 固定 20% 慢速
                    self.motor.move(y=Y_SLOW)

                else:
                    # >= 50cm → 巡航 (即使视觉中部阻塞，距离够远也不急转，慢速通过)
                    if vision_center_blocked and vision_info["ok"]:
                        print(f"[AutoPilot] 远距但视觉中部阻塞 → 慢速通过")
                        self.motor.move(y=Y_SLOW)
                    else:
                        self.motor.move(y=Y_FULL)

            # 锁外: 语音 + sleep (避障动作需持续时间)
            if say_obstacle:
                self.voice_out.say("前方障碍", lang="zh")

            if retreat:
                # 后退 0.5s
                with self._mode_lock:
                    if self._mode != "auto":
                        self.motor.stop()
                        continue
                    self.motor.move(y=-Y_SLOW)
                time.sleep(0.5)
                # 转向 (用视觉推荐的方向，而非固定方向)
                if turn_dir:
                    self._servo_scan_before_turn(turn_dir)
                with self._mode_lock:
                    if self._mode != "auto":
                        self.motor.stop()
                        continue
                    # turn_dir 可能是 "left"/"right"/None。
                    # 之前 bug: None 时默认 rot=-Y_SLOW (左转)，隐性约束脆弱。
                    # 显式处理 None: 不转向，仅后退后停止 (后续循环会重新决策)
                    if turn_dir == "right":
                        rot = Y_SLOW
                    elif turn_dir == "left":
                        rot = -Y_SLOW
                    else:
                        # turn_dir is None, 不转向
                        self.motor.stop()
                        time.sleep(0.4)
                        continue
                    self.motor.move(rotation=rot)
                time.sleep(0.4)
            elif turn_dir:
                # 仅转向不后退 (15~30cm 中部阻塞)
                self._servo_scan_before_turn(turn_dir)
                with self._mode_lock:
                    if self._mode != "auto":
                        self.motor.stop()
                        continue
                    if turn_dir == "right":
                        rot = Y_SLOW
                    elif turn_dir == "left":
                        rot = -Y_SLOW
                    else:
                        self.motor.stop()
                        time.sleep(0.4)
                        continue
                    self.motor.move(rotation=rot)
                time.sleep(0.4)
            else:
                time.sleep(0.2)

    def _analyze_vision(self):
        """抓取摄像头一帧并分析通行性

        Returns:
            dict: vision_obstacle.analyze() 的返回值；不可用时返回空结果
        """
        if not getattr(self.vision_obs, "_initialized", False):
            return {"ok": False}
        if not getattr(self.camera_csi, "_running", False):
            return {"ok": False}
        try:
            frame = self.camera_csi.capture()
            if frame is None:
                return {"ok": False}
            return self.vision_obs.analyze(frame)
        except Exception as e:
            print(f"[AutoPilot] 视觉分析异常: {e}")
            return {"ok": False}

    def _pick_turn_direction(self, vision_info):
        """选择转向方向 (优先视觉推荐)"""
        if vision_info.get("ok"):
            suggested = vision_info.get("suggested_dir", "")
            if suggested in ("left", "right"):
                return suggested
        return "right"  # 默认右转

    def _servo_scan_before_turn(self, turn_dir):
        """转向前云台扫视一眼，提高避障成功率

        往要转的方向先看一眼 (250ms)，避免转过去才发现还是墙。
        云台没初始化时静默跳过。
        """
        if not getattr(self.servo, "_initialized", False):
            return
        try:
            pan_offset = VISION_SCAN_ANGLE if turn_dir == "right" else -VISION_SCAN_ANGLE
            cur_pan, _ = self.servo.get_angles()
            self.servo.pan(cur_pan + pan_offset)
            time.sleep(0.25)
            self.servo.pan(cur_pan)  # 复位
        except Exception:
            pass

    def _voice_control_loop(self):
        """语音控制循环"""
        # 前置检查: 语音输入未就绪则不进入循环 (避免 listen_once 反复抛异常)
        if not getattr(self.voice_in, "_available", False):
            print("[VoiceControl] 语音输入未初始化，跳过语音控制线程")
            return
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
                if getattr(self.servo, "_initialized", False):
                    self.servo.center()
                    self.voice_out.say("云台已归中")
                else:
                    self.voice_out.say("舵机未初始化")
            elif any(w in cmd for w in ["手动", "遥控"]):
                self.set_mode("manual")
                self.voice_out.say("切换为手动模式")
            elif any(w in cmd for w in ["自动", "巡航", "巡游"]):
                self.set_mode("auto")
            elif any(w in cmd for w in ["跟随", "跟着", "跟我"]):
                self.set_mode("follow")
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
        """安全释放所有资源

        释放顺序: 先停后台线程 (follower/web) → 再停硬件 → 最后 GPIO 清理。
        之前 bug: 未停 follower 线程，cleanup 期间 follower 还可能调 motor.move/servo.pan，
        与 cleanup 的资源释放竞争。
        """
        print("\n[Main] 正在关闭系统...")
        self._running = False

        # 1. 先停 follower 线程 (避免它继续调 motor/servo/camera)
        if self.follower:
            try:
                self.follower.stop()
            except Exception as e:
                print(f"[Main] 停止 follower 异常: {e}")

        # 2. 停 web 服务器 (避免新请求触发 on_mode_change → set_mode → motor.move)
        # Flask 用 daemon 线程跑，主进程退出时自动结束；这里不显式 shutdown，
        # 但通过 _running=False 让 auto-pilot/voice 线程退出

        # 3. 停硬件
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
