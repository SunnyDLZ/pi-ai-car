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
    AUTO_MAX_SPEED, AUTO_SLOW_SPEED, WEB_PORT, VISION_SCAN_ANGLE, FOLLOW_SPEED, \
    SERVO_PAN_CENTER, SERVO_TILT_CENTER, FOLLOW_SEARCH_TILTS


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
        self._auto_thread = None   # 自动巡游线程 (存为实例属性供 cleanup join)
        self._voice_thread = None  # 语音控制线程

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
            # 用 diagnose() 获取准确未就绪原因，避免固定说"请先录入主人"
            # (实际原因可能是 dlib 未安装 / 模型缺失 / 主人库为空 / 主人未录入人脸)
            diag = self.face_recognizer.diagnose()
            print(f"[Main] 跟随模式不可用: {diag['reason']} - {diag['detail']}")
            self.voice_out.say(f"跟随模式不可用: {diag['reason']}")
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
                # auto 模式靠超声波正前方测距避障，云台必须朝正前方 (pan=90°)。
                # 超声波装在云台上，pan 偏转会朝偏方向测距、tilt 朝下会把地面
                # 当成障碍 (贴地回波 ~20-40cm) → 空旷处无故急停/撞上真障碍。
                # 强制 pan+tilt 双轴归中。
                if getattr(self.servo, "_initialized", False):
                    try:
                        self.servo.pan(SERVO_PAN_CENTER)
                        self.servo.tilt(SERVO_TILT_CENTER)
                    except Exception:
                        pass
            elif mode == "follow":
                self.motor.set_speed(FOLLOW_SPEED)
                # 主人站立时人脸在小车水平视线上方，云台先回正并微抬头找人
                # (固定平视看不到脸 — "仰视认不出"的一半根因是"看不见")
                if getattr(self.servo, "_initialized", False):
                    try:
                        self.servo.pan(SERVO_PAN_CENTER)
                        self.servo.tilt(FOLLOW_SEARCH_TILTS[0])
                    except Exception:
                        pass

            # 所有模式切换都停车，避免上一模式残留的运动指令继续执行
            self.motor.stop()

            # 同步 WebServer._mode (审查 bug: 之前在锁外调用，AICar._mode 与 WebServer._mode
            # 短暂不一致，可能导致 web 控制请求在非 manual 模式被接受或反之)
            if self.web is not None:
                self.web.set_mode(mode)

        print(f"[Main] 切换到模式: {mode}")
        # 语音播报在锁外 (espeak 子进程启动慢，避免长时间持锁阻塞 auto-pilot)
        if mode == "voice":
            self.voice_out.say("语音模式已开启")
        elif mode == "auto":
            self.voice_out.say("自动模式已开启")
        elif mode == "follow":
            self.voice_out.say("跟随模式已开启")

    def _auto_pilot_loop(self):
        """自动避障巡游 ("停车测距 + 短促移动" 架构)

        根因修复 (2026-07-24): 之前是"边开边测"，测距数据与实际位置错位 —
        measure() 5 样本中位数阻塞 50~185ms，期间小车以 30% 速度继续行驶
        (~5~15cm)；再叠加 640x480 视觉分析 (Pi 上 100~300ms) 和 200ms 循环
        间隔，一轮决策延迟高达 0.5s+。表现为:
          - 空旷时: 决策用的是几百 ms 前的旧数据 → 无故急停/转向
          - 有障碍: 超声波已贴上障碍, 决策还基于"之前还很远"的数据 → 撞上去
        另外两个加重因素:
          - 视觉 Canny 单帧误检 (地面纹理/光影判成障碍) 直接触发避障
          - 单次回波丢失 (斜面/软材质是常态) 立即急停

        新架构: "测 → 动 → 停 → 测" 循环
          1. 每次移动都是 ≤0.25s 的"短促脉冲"，脉冲结束立即停车
          2. 停车状态下测距 (3 样本 ~40ms)，数据与当前位置严格对应
          3. 视觉分析降采样到 320x240 (快 ~4 倍，不再挤占 CPU)
          4. 视觉"中部阻塞"需连续 2 帧确认才动作 (单帧误检不触发)
          5. 单次测距失败不再急停，连续 3 次失败才停车告警
        盲开窗口被限制在单次脉冲内 (~8cm)，决策永远基于新鲜数据。
        """
        if not getattr(self.motor, "_initialized", False):
            print("[AutoPilot] 电机未初始化，跳过自动巡游线程")
            return
        if not getattr(self.ultrasonic, "_initialized", False):
            print("[AutoPilot] 超声波未初始化，跳过自动巡游线程")
            return

        Y_FULL = 100   # × AUTO_MAX_SPEED(30%) → 实际 30%
        Y_SLOW = 67    # → 实际 ~20%

        BURST_FULL = 0.25     # 巡航脉冲时长 (s) — 盲开窗口 ≤ ~8cm
        BURST_SLOW = 0.20     # 慢速脉冲
        BURST_RETREAT = 0.30  # 后退脉冲
        BURST_TURN = 0.30     # 转向脉冲

        measure_fail_streak = 0     # 连续测距失败计数
        vision_blocked_streak = 0   # 视觉中部阻塞连续确认计数

        print("[AutoPilot] 自动巡游启动 (停车测距 + 短促移动)")
        while self._running:
            if self.get_mode() != "auto":
                time.sleep(0.3)
                continue

            # === 1. 停车状态测距 (上一脉冲结束已停车，此处车静止) ===
            dist = self.ultrasonic.measure(samples=3)
            # 同步距离到 follower.state (web 端 /api/distance、/api/status 统一读取)
            if self.follower:
                self.follower.set_distance(round(dist, 1) if dist > 0 else -1)

            if dist < 0:
                # 单次失败不急停 — 斜面/软材质/波束外物体回波丢失是常态
                measure_fail_streak += 1
                if measure_fail_streak >= 3:
                    with self._mode_lock:
                        if self._mode == "auto":
                            self.motor.stop()
                    print("[AutoPilot] 连续 3 次测距失败，已停车 (请检查传感器)")
                    measure_fail_streak = 0
                    time.sleep(0.3)
                else:
                    # 数据不可信时慢速短脉冲试探，下一拍重新测
                    self._auto_burst(y=Y_SLOW, duration=BURST_SLOW)
                continue
            measure_fail_streak = 0

            # === 2. 视觉通行性分析 (内部已降采样 320x240) ===
            vision_info = self._analyze_vision()
            # 视觉"中部阻塞"需连续 2 帧确认 — 地面纹理/光影会让 Canny 单帧
            # 误报障碍，这是"空旷时无故急停转向"的根因之一
            if vision_info["ok"] and vision_info["center_blocked"]:
                vision_blocked_streak += 1
            else:
                vision_blocked_streak = 0
            vision_center_blocked = vision_blocked_streak >= 2

            # === 3. 融合决策 + 短促脉冲动作 ===
            if dist < OBSTACLE_STOP:
                # 太近 (<15cm) → 急停 + 后退 + 转向 (分段执行，每段后重新测距)
                with self._mode_lock:
                    if self._mode != "auto":
                        continue
                    self.motor.stop()
                self.voice_out.say("前方障碍", lang="zh")
                turn_dir = self._pick_turn_direction(vision_info)
                self._auto_burst(y=-Y_SLOW, duration=BURST_RETREAT)
                self._servo_scan_before_turn(turn_dir)
                rot = Y_SLOW if turn_dir == "right" else -Y_SLOW
                self._auto_burst(rotation=rot, duration=BURST_TURN)
                vision_blocked_streak = 0

            elif dist < OBSTACLE_SLOW or vision_center_blocked:
                # 15~30cm 或视觉确认中部阻塞 → 原地转向避障
                suggested = vision_info.get("suggested_dir", "") if vision_info["ok"] else ""
                if suggested == "backward":
                    # 三面都堵 → 后退 + 转向
                    self._auto_burst(y=-Y_SLOW, duration=BURST_RETREAT)
                    self._auto_burst(rotation=Y_SLOW, duration=BURST_TURN)
                elif suggested in ("left", "right"):
                    self.voice_out.say("前方障碍", lang="zh")
                    self._servo_scan_before_turn(suggested)
                    rot = Y_SLOW if suggested == "right" else -Y_SLOW
                    self._auto_burst(rotation=rot, duration=BURST_TURN)
                elif dist < OBSTACLE_SLOW:
                    # 视觉没意见 → 信任超声波，按距离比例慢速贴近
                    # (ratio 0~1: 越靠近 OBSTACLE_STOP 越慢)
                    ratio = (dist - OBSTACLE_STOP) / (OBSTACLE_SLOW - OBSTACLE_STOP)
                    y_val = int(Y_SLOW * max(0.3, ratio))
                    self._auto_burst(y=y_val, duration=BURST_SLOW)
                else:
                    # 视觉确认阻塞但超声波说还远 (>30cm) → 慢速观察性前进
                    self._auto_burst(y=Y_SLOW, duration=BURST_SLOW)
                vision_blocked_streak = 0

            elif dist < OBSTACLE_WARN:
                # 30~50cm → 慢速短脉冲
                self._auto_burst(y=Y_SLOW, duration=BURST_SLOW)

            else:
                # ≥50cm → 巡航 (视觉确认中部阻塞则降速通过)
                if vision_center_blocked:
                    self._auto_burst(y=Y_SLOW, duration=BURST_SLOW)
                    vision_blocked_streak = 0
                else:
                    self._auto_burst(y=Y_FULL, duration=BURST_FULL)

    def _auto_burst(self, x=0, y=0, rotation=0, duration=0.2):
        """执行一次限时移动脉冲，结束后立即停车

        保证: ① 盲开窗口 ≤ duration；② 脉冲间车静止，下一拍测距数据新鲜；
        ③ 模式切换最迟 duration 内生效 (每次脉冲前检查模式)。
        """
        with self._mode_lock:
            if self._mode != "auto":
                self.motor.stop()
                return
            self.motor.move(x=x, y=y, rotation=rotation)
        time.sleep(duration)
        with self._mode_lock:
            self.motor.stop()

    def _analyze_vision(self):
        """抓取摄像头一帧并分析通行性 (降采样提速)

        之前在 640x480 全分辨率上跑 Canny+形态学+轮廓，Pi 上每帧 100~300ms，
        既拖慢 auto 决策循环，又长期挤占 CPU 饿死 Flask/视频流线程。
        降到 320x240 后快 ~4 倍；左中右三段占比统计对分辨率不敏感，
        精度损失可忽略 (最小轮廓面积同步按面积比换算)。

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
            import cv2
            small = cv2.resize(frame, (320, 240))
            return self.vision_obs.analyze(small)
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
            elif any(w in cmd for w in ["加速", "快一点", "快点"]):
                # 审查 bug: 之前包含"速度"，用户说"速度慢点"会先命中加速分支误加速
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

        # 启动后台线程 (存为实例属性供 cleanup join)
        self._auto_thread = threading.Thread(target=self._auto_pilot_loop, daemon=True)
        self._auto_thread.start()

        self._voice_thread = threading.Thread(target=self._voice_control_loop, daemon=True)
        self._voice_thread.start()

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

        释放顺序: 先停后台线程 (follower/auto/voice) → 再停硬件 → 最后 GPIO 清理。
        之前 bug: 未 join auto/voice 线程，cleanup 期间它们可能调 motor/ultrasonic，
        而 GPIO 已被释放导致崩溃 (ultrasonic.cleanup 不重置 _initialized)。
        """
        print("\n[Main] 正在关闭系统...")
        self._running = False

        # 1. 先停 follower 线程 (避免它继续调 motor/servo/camera)
        if self.follower:
            try:
                self.follower.stop()
            except Exception as e:
                print(f"[Main] 停止 follower 异常: {e}")

        # 2. join auto/voice 线程 (审查 bug: 之前不 join，cleanup 释放 GPIO 后
        # auto-pilot 仍可能调 ultrasonic.measure() 崩溃)
        for t in (self._auto_thread, self._voice_thread):
            if t and t.is_alive():
                t.join(timeout=1.5)

        # 3. 停 web 服务器 (避免新请求触发 on_mode_change → set_mode → motor.move)
        # Flask 用 daemon 线程跑，主进程退出时自动结束；这里不显式 shutdown

        # 4. 停硬件
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
        # 仅在语音输出已初始化时才播报 (用 say_wait 确保播完再退出)
        try:
            if self.voice_out._tts_engine:
                self.voice_out.say_wait("小车已关机")
        except Exception:
            pass
        print("[Main] 系统已安全关闭 ✅")


if __name__ == "__main__":
    car = AICar()
    car.run()
