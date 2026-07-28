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
    AUTO_MAX_SPEED, AUTO_SLOW_SPEED, AUTO_DEFAULT_SPEED, AUTO_CRUISE_DIST, \
    WEB_PORT, VISION_SCAN_ANGLE, FOLLOW_SPEED, SERVO_PAN_CENTER, SERVO_TILT_CENTER, FOLLOW_SEARCH_TILTS


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
                # 默认速度 20 (AUTO_DEFAULT_SPEED)，运行中按距离动态调整:
                # >1m 提到 30，<1m 降回 20，<50cm 减速，<15cm 停车找路
                self.motor.set_speed(AUTO_DEFAULT_SPEED)
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
        """自动避障巡游 - 超声波主测距 + 摄像头辅助找路

        速度策略 (按正前方距离动态渐变):
          - dist > 1m (100cm):   目标速度 30 (AUTO_MAX_SPEED)
          - 50cm ≤ dist ≤ 1m:    目标速度 20 (AUTO_DEFAULT_SPEED)
          - 15cm ≤ dist < 50cm:  线性减速 20→5
          - dist < 15cm:         停车找路

        找路策略 (停车后，云台+摄像头辅助):
          1. 云台右转 → 超声波+摄像头检测右侧
          2. 右侧畅通 → 车身右转，云台归中，继续前进
          3. 右侧阻塞 → 云台左转 → 检测左侧
          4. 左侧畅通 → 车身左转，云台归中，继续前进
          5. 左右都阻塞 → 原地掉头 (180°)，云台归中

        架构: "测 → 动 → 停 → 测" 循环
          - 每次移动都是 ≤0.25s 的短促脉冲，脉冲结束立即停车
          - 停车状态下测距，数据与当前位置严格对应
          - 盲开窗口限制在单次脉冲内 (~8cm)，决策永远基于新鲜数据
          - 单次测距失败不急停，连续 3 次失败才停车告警
        """
        if not getattr(self.motor, "_initialized", False):
            print("[AutoPilot] 电机未初始化，跳过自动巡游线程")
            return
        if not getattr(self.ultrasonic, "_initialized", False):
            print("[AutoPilot] 超声波未初始化，跳过自动巡游线程")
            return

        # 速度渐变步长 (每拍调整 ±2%，避免突变导致电机抖动/打滑)
        SPEED_STEP = 2
        BURST_FORWARD = 0.25        # 前进脉冲时长 (s) — 盲开窗口 ≤ ~8cm
        BURST_TURN = 0.30           # ~90° 转向脉冲 (原 0.5s 转过头)
        BURST_TURN_AROUND = 0.60    # ~180° 原地掉头脉冲 (原 1.2s 实际转了 360°)
        BURST_RETREAT = 0.30        # 后退脉冲 (太近时先退一点再找路)

        current_speed = AUTO_DEFAULT_SPEED  # 当前实际速度 (渐变到目标值)
        measure_fail_streak = 0

        print("[AutoPilot] 自动巡游启动 (超声波测距 + 摄像头辅助找路)")
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
                    # 用 get_mode() 替代持锁检查，避免与 /api/stop 争 _mode_lock
                    if self.get_mode() == "auto":
                        self.motor.stop()
                    print("[AutoPilot] 连续 3 次测距失败，已停车 (请检查传感器)")
                    measure_fail_streak = 0
                    time.sleep(0.3)
                else:
                    # 数据不可信时慢速短脉冲试探，下一拍重新测
                    self.motor.set_speed(AUTO_DEFAULT_SPEED)
                    self._auto_burst(y=80, duration=BURST_FORWARD)
                continue
            measure_fail_streak = 0

            # === 2. 距离 → 目标速度 (动态渐变) ===
            if dist >= AUTO_CRUISE_DIST:
                # >1m: 提速到 30
                target_speed = AUTO_MAX_SPEED
            elif dist >= OBSTACLE_WARN:
                # 50cm~1m: 降到 20
                target_speed = AUTO_DEFAULT_SPEED
            elif dist >= OBSTACLE_STOP:
                # 15~50cm: 线性减速 20→5
                ratio = (dist - OBSTACLE_STOP) / (OBSTACLE_WARN - OBSTACLE_STOP)
                target_speed = int(5 + (AUTO_DEFAULT_SPEED - 5) * ratio)
            else:
                # <15cm: 停车找路
                target_speed = 0

            # 渐变到目标速度 (每拍最多 ±SPEED_STEP)
            if current_speed < target_speed:
                current_speed = min(target_speed, current_speed + SPEED_STEP)
            elif current_speed > target_speed:
                current_speed = max(target_speed, current_speed - SPEED_STEP)

            # === 3. 执行动作 ===
            if dist < OBSTACLE_STOP:
                # 太近 (<15cm) → 停车找路
                if self.get_mode() != "auto":
                    continue
                self.motor.stop()
                self.voice_out.say("前方障碍", lang="zh")
                # 先退一点腾出空间 (<15cm 时车身可能已贴近障碍)
                self.motor.set_speed(AUTO_DEFAULT_SPEED)
                self._auto_burst(y=-80, duration=BURST_RETREAT)
                # 云台扫视找路 (内部会归中云台)
                turn_dir = self._find_path_and_turn()
                if turn_dir == "right":
                    self._auto_burst(rotation=80, duration=BURST_TURN)
                elif turn_dir == "left":
                    self._auto_burst(rotation=-80, duration=BURST_TURN)
                else:
                    # 左右都没路 → 原地掉头 (~180°，不是 360°)
                    self._auto_burst(rotation=80, duration=BURST_TURN_AROUND)
                # 转向后立即前进试探 — 若转对了方向则走出障碍区，
                # 若仍受阻则下一拍循环会重新测距找路 (不会原地打转)
                self._auto_burst(y=80, duration=BURST_FORWARD)
                # 找路后恢复默认速度
                current_speed = AUTO_DEFAULT_SPEED

            else:
                # 前方有空间 → 前进 (速度已按距离渐变)
                # 仅当速度变化时才调 set_speed (避免每拍重复调)
                if self.motor.get_speed() != current_speed:
                    self.motor.set_speed(current_speed)
                self._auto_burst(y=100, duration=BURST_FORWARD)

    def _find_path_and_turn(self):
        """停车后用云台+摄像头找路

        顺序: 右 → 左 → (调用方处理掉头)
        云台转向目标方向后，用超声波(主)+摄像头(辅)判断该方向是否畅通。
        无论结果如何，返回前都会把云台归中 (超声波/摄像头朝正前方)。

        Returns:
            str: "right" / "left" / "none" (左右都没路，需原地掉头)
        """
        if not getattr(self.servo, "_initialized", False):
            # 云台不可用 → 默认右转
            return "right"

        try:
            # === 1. 云台右转，检测右侧 ===
            self.servo.pan(SERVO_PAN_CENTER + VISION_SCAN_ANGLE)
            time.sleep(0.4)  # 等舵机到位 + 摄像头曝光稳定

            if self._check_path_clear():
                return "right"

            # === 2. 右侧没路 → 云台左转，检测左侧 ===
            self.servo.pan(SERVO_PAN_CENTER - VISION_SCAN_ANGLE)
            time.sleep(0.4)

            if self._check_path_clear():
                return "left"

            # === 3. 左右都没路 ===
            return "none"

        except Exception as e:
            print(f"[AutoPilot] 找路异常: {e}")
            return "right"  # 异常时默认右转
        finally:
            # 无论走哪个分支，归中云台 (超声波/摄像头朝正前方)
            try:
                self.servo.pan(SERVO_PAN_CENTER)
                time.sleep(0.15)
            except Exception:
                pass

    def _check_path_clear(self, min_clear_dist=None):
        """检查当前云台朝向是否有路

        超声波为主 + 摄像头辅助:
          - 超声波远 (≥阈值) + 摄像头畅通 → 有路
          - 超声波远 + 摄像头阻塞 → 没路 (摄像头辅助否决)
          - 超声波近 (<阈值) → 没路 (不管摄像头)
          - 超声波失败 → 只靠摄像头

        Args:
            min_clear_dist: 判定畅通的最小距离 (cm)，默认 OBSTACLE_SLOW(30cm)
                            原 50cm 太严格，狭窄空间左右都判"没路"→ 总走 360° 掉头

        Returns:
            bool: True=有路, False=阻塞
        """
        if min_clear_dist is None:
            min_clear_dist = float(OBSTACLE_SLOW)

        # 超声波测距 (主) — 云台已转向目标方向，超声波同向测距
        dist = self.ultrasonic.measure(samples=3)

        # 摄像头视觉分析 (辅) — 分析当前画面中部通行性
        vision_info = self._analyze_vision()
        vision_ok = vision_info.get("ok", False)
        vision_clear = vision_ok and not vision_info.get("center_blocked", False)

        if dist > 0:
            if dist >= min_clear_dist:
                # 超声波说远 → 信任摄像头辅助判断
                return vision_clear if vision_ok else True
            else:
                # 超声波说近 → 没路
                return False
        else:
            # 超声波失败 → 只能靠摄像头
            return vision_clear

    def _auto_burst(self, x=0, y=0, rotation=0, duration=0.2):
        """执行一次限时移动脉冲，结束后立即停车

        优化 (2026-07-25):
          - 模式检查用 get_mode() (短锁)，不长时间持 _mode_lock
          - motor.move/motor.stop 内部已有 motor._lock，无需外层再加 _mode_lock
          - 减少 _mode_lock 持有时间，避免 /api/stop 阻塞
          - 脉冲结束后无条件停车 (双保险: 模式已切换时也确保停止)
        """
        # 模式检查 (短锁，不持锁期间 sleep)
        if self.get_mode() != "auto":
            self.motor.stop()
            return
        # motor.move 内部有 motor._lock (细粒度)，不与 _mode_lock 嵌套
        self.motor.move(x=x, y=y, rotation=rotation)
        time.sleep(duration)
        # 无条件停车 — 模式已切换时也确保停止 (双保险)
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
