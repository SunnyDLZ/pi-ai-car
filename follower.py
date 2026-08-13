"""
follower.py - 人体跟随控制

跟随模式核心逻辑 (简化版 — 不再识别人脸身份):
  1. 摄像头捕获 → 人脸检测 (dlib HOG，只需检测不需识别)
  2. 选画面里最大的人脸作为跟随目标 (通常最近的)
  3. 计算偏差控制跟随
  4. 没看到人 → 小幅扫视找人，超时报丢失
  5. 超声波兜底: 跟随时前方 <20cm 障碍强制停

控制策略 (基于人脸框位置和大小):
  - box 中心 X 偏离画面中心 → 控制旋转对准 (麦轮原地转)
  - box 宽度占画面比例 < TARGET_BOX_RATIO → 目标远了 → 前进
  - box 宽度占画面比例 > TARGET_BOX_RATIO × 1.4 → 目标太近 → 后退
  - 比例合适 → 停止前后，只保持对准

云台主动追踪:
  - 人不在画面中心时，云台 pan 微调跟随
  - 云台转到底 (超出 ±35°) → 触发车身旋转
"""

import time
import threading
from config import (
    FOLLOW_SPEED,
    FOLLOW_TARGET_BOX_RATIO,
    FOLLOW_LOST_TIMEOUT,
    FOLLOW_OBSTACLE_SAFE_DIST,
    FOLLOW_DETECT_WIDTH,
    FOLLOW_PAN_LIMIT,
    FOLLOW_SEARCH_TILTS,
    FOLLOW_RETRY_INTERVAL,
    SERVO_PAN_CENTER,
    SERVO_PAN_INVERT,
)


class Follower:
    """人体跟随控制器 (跟随任意人，不识别身份)"""

    def __init__(self, car):
        """注入 AICar 实例 (需要访问 motor/camera/ultrasonic/servo/face_recognizer/voice_out)"""
        self.car = car
        self._running = False
        self._thread = None
        self._last_seen_time = 0.0       # 上次看到人的时间 (用于丢失判定)
        self._search_scan_dir = 1        # 找人时的扫视方向 +1/-1
        self._search_tilt_idx = 0        # 找人扫视仰角档位索引 (FOLLOW_SEARCH_TILTS)
        self._last_sensitive_try = 0.0   # 上次高灵敏检测重试的时间
        self._last_detect_time = 0.0     # 上次 dlib 检测时间 (节流用)
        self._entered_follow = False     # 是否已进入过 follow 模式 (用于初始化 _last_seen_time)
        self._last_obstacle_warn_time = 0.0  # 上次播报"前方有障碍"的时间 (防重复刷屏)

        # 状态字段 (供 web 端读取 + 视频流复用)
        self.state = {
            "following": False,
            "target_name": None,
            "lost": False,
            "box_ratio": 0.0,
            "offset_x": 0.0,
            "msg": "",
            "distance": -1,        # 最近一次超声波距离 (供 /api/status 读取)
            "last_faces": [],      # 最近一帧的人脸检测结果 (供视频流复用)
            "last_ids": [],        # 兼容字段 (跟随模式不再识别身份，恒为空)
        }
        self._state_lock = threading.Lock()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Follower] 跟随线程启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._state_lock:
            self.state["following"] = False
            self.state["target_name"] = None
            self.state["lost"] = False
            self.state["last_faces"] = []
            self.state["last_ids"] = []
            self._entered_follow = False
        print("[Follower] 跟随线程停止")

    def get_state(self):
        with self._state_lock:
            return dict(self.state)

    def set_distance(self, dist):
        """更新缓存距离 (供 auto-pilot 线程在 auto 模式下调用)

        auto 模式不走 follower 线程，但 web 端 /api/distance 和 /api/status
        统一从 follower.state.distance 读取 (避免重复 measure 阻塞)。
        """
        with self._state_lock:
            self.state["distance"] = dist

    def _set_state(self, **kwargs):
        with self._state_lock:
            self.state.update(kwargs)

    def _loop(self):
        """跟随主循环 — 由 main.py 启动, 仅在 mode=="follow" 时工作"""
        while self._running:
            current_mode = self.car.get_mode()
            if current_mode != "follow":
                self._set_state(following=False, target_name=None, msg="待机",
                                last_faces=[], last_ids=[])
                self._entered_follow = False
                time.sleep(0.3)
                continue

            # 刚进入 follow 模式: 从此刻开始计时
            if not self._entered_follow:
                self._last_seen_time = time.time()
                self._search_tilt_idx = 0
                self._entered_follow = True
                # 云台回正并微抬头: 人站立时人脸在小车水平视线上方
                if getattr(self.car.servo, "_initialized", False):
                    try:
                        self.car.servo.pan(SERVO_PAN_CENTER)
                        self.car.servo.tilt(FOLLOW_SEARCH_TILTS[0])
                    except Exception:
                        pass

            # 检查依赖 — 只需 face_recognizer 初始化 (检测器就绪即可，不需主人库)
            if not getattr(self.car.face_recognizer, "_initialized", False):
                self._set_state(msg="人脸检测器未就绪 (检查 dlib 安装)",
                                last_faces=[], last_ids=[])
                with self.car._mode_lock:
                    if self.car._mode == "follow":
                        self.car.motor.stop()
                time.sleep(1.0)
                continue

            # 1. 抓帧 + 检测人脸 (降采样 320px，HOG 快 ~4 倍)
            # 检测节流: 距上次检测 < 150ms 时复用上一帧结果，只更新控制
            frame = self.car.camera_csi.capture()
            if frame is None:
                self._set_state(msg="摄像头捕获失败", last_faces=[], last_ids=[])
                time.sleep(0.2)
                continue

            now = time.time()
            DETECT_THROTTLE = 0.15
            if (now - self._last_detect_time) < DETECT_THROTTLE and \
                    self.state.get("last_faces") is not None:
                # 复用上次检测结果 (不跑 dlib)
                faces = self.state.get("last_faces") or []
            else:
                # 跑 dlib 检测 (只检测，不识别身份)
                self._last_detect_time = now
                faces = self.car.face_recognizer.detect_faces(
                    frame, detect_width=FOLLOW_DETECT_WIDTH)

            # 仰视/小脸兜底: 快速检测没找到人时，每 ~1s 做一次高灵敏检测
            if not faces and \
                    (now - self._last_sensitive_try) > FOLLOW_RETRY_INTERVAL:
                self._last_sensitive_try = now
                retry_faces = self.car.face_recognizer.detect_faces(
                    frame, upsample=1, jittering=1)
                if retry_faces:
                    faces = retry_faces
                    self._last_detect_time = now

            # 缓存检测结果供视频流复用
            self._set_state(last_faces=faces, last_ids=[])

            if not faces:
                # 没看到人 → 进入"找人"模式
                self._handle_no_target()
                continue

            # 2. 选画面里最大的人脸作为跟随目标 (通常最近的)
            target_face = max(faces, key=lambda x: x["box"][2] * x["box"][3])
            self._last_seen_time = time.time()

            # 3. 超声波兜底: 前方近距障碍强制停
            dist = self.car.ultrasonic.measure(samples=3)
            self._set_state(distance=round(dist, 1) if dist > 0 else -1)

            if dist < 0:
                with self.car._mode_lock:
                    if self.car._mode == "follow":
                        self.car.motor.stop()
                self._set_state(
                    following=True,
                    target_name="人",
                    msg="超声波测距失败，已停车",
                    lost=False,
                )
                time.sleep(0.5)
                continue

            if dist < FOLLOW_OBSTACLE_SAFE_DIST:
                with self.car._mode_lock:
                    if self.car._mode == "follow":
                        self.car.motor.stop()
                self._set_state(
                    following=True,
                    target_name="人",
                    msg=f"前方 {dist:.0f}cm 有障碍，已停",
                    lost=False,
                )
                now = time.time()
                if now - self._last_obstacle_warn_time > 3.0:
                    self.car.voice_out.say("前方有障碍")
                    self._last_obstacle_warn_time = now
                time.sleep(0.5)
                continue

            # 4. 计算控制偏差
            h, w = frame.shape[:2]
            box_x, box_y, box_w, box_h = target_face["box"]
            cx = box_x + box_w / 2
            offset_x = (cx - w / 2) / (w / 2)  # -1 (最左) ~ 1 (最右)
            box_ratio = box_w / w                  # 0 ~ 1

            # 5. 控制决策
            self._control(offset_x, box_ratio, frame_h=h)

            # 6. 云台主动追踪
            self._servo_track(offset_x, box_y + box_h / 2, h)

            self._set_state(
                following=True,
                target_name="人",
                lost=False,
                box_ratio=round(box_ratio, 3),
                offset_x=round(offset_x, 3),
                msg="跟随中",
            )

            time.sleep(0.05)

    def _control(self, offset_x, box_ratio, frame_h):
        """根据偏差控制电机 ("云台先行 + 比例控制")

        offset_x: -1~1 (画面中心偏移)
        box_ratio: 0~1 (人脸框占画面宽度比)
        """
        target = FOLLOW_TARGET_BOX_RATIO
        too_far = box_ratio < target * 0.85
        too_close = box_ratio > target * 1.4

        # --- 车身旋转由云台偏角驱动 (云台先行) ---
        pan_err = 0
        if getattr(self.car.servo, "_initialized", False):
            try:
                cur_pan, _ = self.car.servo.get_angles()
                # 审查 bug: get_angles() 返回逻辑角度，而 pan() 在 SERVO_PAN_INVERT=True
                # 时发送 180-angle 的物理脉宽。因此"逻辑角度增大"在物理上对应摄像头
                # 向左转。若直接 cur_pan - CENTER，pan_err 的符号与物理方向相反，
                # 导致车身旋转方向错误 (目标在右却左转)。
                # 修正: 按 INVERT 反转符号，使 pan_err>0 恒表示"云台物理偏右→目标在右"。
                if SERVO_PAN_INVERT:
                    pan_err = SERVO_PAN_CENTER - cur_pan
                else:
                    pan_err = cur_pan - SERVO_PAN_CENTER
            except Exception:
                pan_err = 0

        rot_val = 0
        if abs(pan_err) > FOLLOW_PAN_LIMIT:
            rot_val = int(max(-50, min(50, pan_err * 1.5)))
        elif abs(offset_x) > 0.55:
            # 云台失效(未初始化)时兜底
            rot_val = int(offset_x * 45)

        # --- 前后移动按比例调速 ---
        y_val = 0
        if too_far:
            ratio_gap = target - box_ratio
            y_val = int(30 + min(30, ratio_gap * 150))
        elif too_close:
            ratio_gap = box_ratio - target
            y_val = -int(20 + min(20, ratio_gap * 100))

        # 大角度旋转时暂停前后移动
        if abs(rot_val) > 30:
            y_val = 0

        with self.car._mode_lock:
            if self.car._mode != "follow":
                return
            self.car.motor.set_speed(FOLLOW_SPEED)
            if y_val != 0 or rot_val != 0:
                self.car.motor.move(y=y_val, rotation=rot_val)
            else:
                self.car.motor.stop()

    def _servo_track(self, offset_x, face_cy, frame_h):
        """云台微调追踪人"""
        if not getattr(self.car.servo, "_initialized", False):
            return
        try:
            cur_pan, cur_tilt = self.car.servo.get_angles()
            # 审查 bug: 与 _control 同理，SERVO_PAN_INVERT=True 时逻辑角度增大对应
            # 物理向左。目标在右 (offset_x>0) 应让摄像头物理右转 → 逻辑 pan 需减小。
            # 故按 INVERT 反转 pan_delta 符号，使云台追踪方向与物理一致。
            pan_delta = int(offset_x * 3)
            if SERVO_PAN_INVERT:
                pan_delta = -pan_delta
            new_pan = max(0, min(180, cur_pan + pan_delta))

            offset_y = (face_cy - frame_h / 2) / (frame_h / 2)  # -1~1
            tilt_delta = int(offset_y * 2)
            new_tilt = max(0, min(180, cur_tilt + tilt_delta))

            if abs(new_pan - cur_pan) >= 2:
                self.car.servo.pan(new_pan)
            if abs(new_tilt - cur_tilt) >= 3:
                self.car.servo.tilt(new_tilt)
        except Exception:
            pass

    def _handle_no_target(self):
        """看不到人时的处理"""
        now = time.time()
        lost_duration = now - self._last_seen_time

        if lost_duration < FOLLOW_LOST_TIMEOUT:
            self._set_state(
                following=False,
                target_name=None,
                lost=False,
                msg=f"寻找人中 ({lost_duration:.1f}s)",
                last_faces=[], last_ids=[],
            )
            with self.car._mode_lock:
                if self.car._mode == "follow":
                    self.car.motor.stop()
            # 云台左右扫 + 仰角档位切换找人
            if getattr(self.car.servo, "_initialized", False):
                try:
                    cur_pan, _ = self.car.servo.get_angles()
                    new_pan = cur_pan + self._search_scan_dir * 20
                    hit_edge = False
                    if new_pan > 150:
                        self._search_scan_dir = -1
                        new_pan = 150
                        hit_edge = True
                    elif new_pan < 30:
                        self._search_scan_dir = 1
                        new_pan = 30
                        hit_edge = True
                    self.car.servo.pan(new_pan)
                    if hit_edge:
                        self._search_tilt_idx = \
                            (self._search_tilt_idx + 1) % len(FOLLOW_SEARCH_TILTS)
                        self.car.servo.tilt(FOLLOW_SEARCH_TILTS[self._search_tilt_idx])
                except Exception:
                    pass
            time.sleep(0.3)
        else:
            # 超时 → 真丢失，停车 + 播报
            self._set_state(
                following=False,
                target_name=None,
                lost=True,
                msg="人走丢了",
                last_faces=[], last_ids=[],
            )
            with self.car._mode_lock:
                if self.car._mode == "follow":
                    self.car.motor.stop()
            self.car.voice_out.say("人走丢了")
            time.sleep(1.0)
