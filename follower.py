"""
follower.py - 主人跟随控制

跟随模式核心逻辑:
  1. 摄像头捕获 → 人脸检测 → 识别身份
  2. 若是已注册主人 → 计算偏差控制跟随
  3. 若不是主人 → 停车不跟 (避免跟陌生人)
  4. 没看到主人 → 小幅扫视找人，超时报丢失
  5. 超声波兜底: 跟随时前方 <20cm 障碍强制停

控制策略 (基于人脸框位置和大小):
  - box 中心 X 偏离画面中心 → 控制旋转对准 (麦轮原地转)
  - box 宽度占画面比例 < TARGET_BOX_RATIO → 主人远了 → 前进
  - box 宽度占画面比例 > TARGET_BOX_RATIO × 1.4 → 主人太近 → 后退
  - 比例合适 → 停止前后，只保持对准

云台主动追踪:
  - 主人不在画面中心时，云台 pan 微调跟随
  - 云台转到底 (超出 ±30°) → 触发车身旋转
"""

import time
import threading
from config import (
    FOLLOW_SPEED,
    FOLLOW_TARGET_BOX_RATIO,
    FOLLOW_LOST_TIMEOUT,
    FOLLOW_OBSTACLE_SAFE_DIST,
)


class Follower:
    """主人跟随控制器"""

    def __init__(self, car):
        """注入 AICar 实例 (需要访问 motor/camera/ultrasonic/servo/face_recognizer/voice_out)"""
        self.car = car
        self._running = False
        self._thread = None
        self._last_seen_time = 0.0       # 上次看到主人的时间 (用于丢失判定)
        self._last_target_name = None    # 上次跟随的主人名 (用于丢失播报)
        self._search_scan_dir = 1        # 找人时的扫视方向 +1/-1
        self._entered_follow = False     # 是否已进入过 follow 模式 (用于初始化 _last_seen_time)

        # 状态字段 (供 web 端读取 + 视频流复用)
        self.state = {
            "following": False,
            "target_name": None,
            "lost": False,
            "box_ratio": 0.0,
            "offset_x": 0.0,
            "msg": "",
            "distance": -1,        # 最近一次超声波距离 (供 /api/status 读取，避免重复测距)
            "last_faces": [],      # 最近一帧的人脸检测结果 (供视频流复用，避免重复 dlib)
            "last_ids": [],        # 最近一帧的识别结果
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

            # 刚进入 follow 模式: 从此刻开始计时，
            # 否则 _last_seen_time=0 + lost_duration=0 永远不超时 (审查 bug 3.1)
            if not self._entered_follow:
                self._last_seen_time = time.time()
                self._last_target_name = None
                self._entered_follow = True

            # 检查依赖
            if not self.car.face_recognizer.is_ready():
                self._set_state(msg="主人识别未就绪 (未安装 dlib 或主人库为空)",
                                last_faces=[], last_ids=[])
                with self.car._mode_lock:
                    if self.car._mode == "follow":
                        self.car.motor.stop()
                time.sleep(1.0)
                continue

            # 1. 抓帧 + 检测人脸
            frame = self.car.camera_csi.capture()
            if frame is None:
                # 摄像头捕获失败时更新 state，避免 web 端看到旧状态 (审查 bug 3.7)
                self._set_state(msg="摄像头捕获失败", last_faces=[], last_ids=[])
                time.sleep(0.2)
                continue

            faces = self.car.face_recognizer.detect_faces(frame)

            # 2. 识别身份 → 筛出主人
            my_owners = []
            identifications = []
            for f in faces:
                name = self.car.face_recognizer.identify(f)
                identifications.append(name)
                if name:
                    my_owners.append((f, name))

            # 缓存检测结果供视频流复用 (避免视频流线程再跑一次 dlib)
            self._set_state(last_faces=faces, last_ids=identifications)

            if not my_owners:
                # 看到人但都不是主人 / 完全没人 → 进入"找人"模式
                self._handle_no_target()
                continue

            # 3. 选画面里最大的人脸作为跟随目标 (通常最近的)
            target_face, target_name = max(my_owners, key=lambda x: x[0]["box"][2] * x[0]["box"][3])
            self._last_seen_time = time.time()
            self._last_target_name = target_name

            # 4. 超声波兜底: 前方近距障碍强制停
            dist = self.car.ultrasonic.measure()
            # 把距离写到 state 供 /api/status 读取 (避免它再调一次 measure 阻塞)
            self._set_state(distance=round(dist, 1) if dist > 0 else -1)

            # dist < 0: 测距失败 → 盲目前进有撞车风险，强制停车 (审查 bug 3.3)
            if dist < 0:
                with self.car._mode_lock:
                    if self.car._mode == "follow":
                        self.car.motor.stop()
                self._set_state(
                    following=True,
                    target_name=target_name,
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
                    target_name=target_name,
                    msg=f"前方 {dist:.0f}cm 有障碍，已停",
                    lost=False,
                )
                self.car.voice_out.say("前方有障碍")
                time.sleep(0.5)
                continue

            # 5. 计算控制偏差
            h, w = frame.shape[:2]
            box_x, box_y, box_w, box_h = target_face["box"]
            cx = box_x + box_w / 2
            offset_x = (cx - w / 2) / (w / 2)  # -1 (最左) ~ 1 (最右)
            box_ratio = box_w / w                  # 0 ~ 1

            # 6. 控制决策
            self._control(target_name, offset_x, box_ratio, frame_h=h)

            # 7. 云台主动追踪 (微调对准主人)
            self._servo_track(offset_x, box_y + box_h / 2, h)

            self._set_state(
                following=True,
                target_name=target_name,
                lost=False,
                box_ratio=round(box_ratio, 3),
                offset_x=round(offset_x, 3),
                msg=f"跟随 {target_name}",
            )

            time.sleep(0.1)  # ~10 FPS

    def _control(self, target_name, offset_x, box_ratio, frame_h):
        """根据偏差控制电机

        offset_x: -1~1 (画面中心偏移)
        box_ratio: 0~1 (人脸框占画面宽度比)

        区间设计 (审查 bug 3.6): 之前 just_right=[0.85, 1.4] 与 too_far<0.7 / too_close>1.6
        之间有 [0.7, 0.85) 和 (1.4, 1.6] 两段死区，落入死区时车会无谓停顿。
        改为连续区间: too_far<0.85, just_right=[0.85, 1.4], too_close>1.4
        """
        target = FOLLOW_TARGET_BOX_RATIO
        too_far = box_ratio < target * 0.85
        too_close = box_ratio > target * 1.4
        just_right = target * 0.85 <= box_ratio <= target * 1.4

        # 左右控制: offset_x
        # 偏差 <0.15 视为对准, 不旋转
        needs_rotate = abs(offset_x) > 0.15
        rot_val = int(offset_x * 50) if needs_rotate else 0  # ±50

        with self.car._mode_lock:
            if self.car._mode != "follow":
                return
            # set_speed 在锁内执行 (审查 bug 1.2): 之前在锁外执行，可能与 set_mode 恢复用户速度竞态
            self.car.motor.set_speed(FOLLOW_SPEED)

            if too_far:
                # 远了 → 前进 + 微调对准
                self.car.motor.move(y=60, rotation=rot_val)
            elif too_close:
                # 太近 → 后退 + 微调对准
                self.car.motor.move(y=-60, rotation=rot_val)
            elif just_right and not needs_rotate:
                # 距离和方向都对 → 停下等主人
                self.car.motor.stop()
            elif needs_rotate:
                # 距离合适但偏离 → 原地旋转对准
                self.car.motor.move(rotation=rot_val)
            else:
                self.car.motor.stop()

    def _servo_track(self, offset_x, face_cy, frame_h):
        """云台微调追踪主人

        offset_x: 人脸中心 X 偏离画面中心的比例 (-1~1)
        face_cy: 人脸中心 Y 像素
        frame_h: 画面高度
        """
        if not getattr(self.car.servo, "_initialized", False):
            return
        try:
            cur_pan, cur_tilt = self.car.servo.get_angles()
            # pan 跟随: offset_x > 0 (人在右) → pan 角度增大 (向右)
            # 每帧微调 3°, 避免震荡
            pan_delta = int(offset_x * 3)
            new_pan = max(0, min(180, cur_pan + pan_delta))

            # tilt 跟随: 人脸中心 Y 偏离画面中心 → 调整俯仰
            # 人脸偏上 (face_cy < frame_h/2) → 抬头 (tilt 减小)
            offset_y = (face_cy - frame_h / 2) / (frame_h / 2)  # -1~1
            tilt_delta = int(offset_y * 2)
            new_tilt = max(0, min(180, cur_tilt + tilt_delta))

            # 只在有显著偏移时调整，减少舵机抖动
            if abs(new_pan - cur_pan) >= 2:
                self.car.servo.pan(new_pan)
            if abs(new_tilt - cur_tilt) >= 3:
                self.car.servo.tilt(new_tilt)
        except Exception:
            pass

    def _handle_no_target(self):
        """看不到主人时的处理"""
        now = time.time()
        # _last_seen_time 在进入 follow 时已初始化为 time.time() (审查 bug 3.1)
        lost_duration = now - self._last_seen_time

        if lost_duration < FOLLOW_LOST_TIMEOUT:
            # 短暂丢失 → 原地小幅扫视找人
            self._set_state(
                following=False,
                target_name=self._last_target_name,
                lost=False,
                msg=f"寻找主人中 ({lost_duration:.1f}s)",
                last_faces=[], last_ids=[],
            )
            with self.car._mode_lock:
                if self.car._mode == "follow":
                    self.car.motor.stop()
            # 云台左右扫
            if getattr(self.car.servo, "_initialized", False):
                try:
                    cur_pan, _ = self.car.servo.get_angles()
                    new_pan = cur_pan + self._search_scan_dir * 20
                    if new_pan > 150:
                        self._search_scan_dir = -1
                        new_pan = 150
                    elif new_pan < 30:
                        self._search_scan_dir = 1
                        new_pan = 30
                    self.car.servo.pan(new_pan)
                except Exception:
                    pass
            time.sleep(0.3)
        else:
            # 超时 → 真丢失，停车 + 播报
            self._set_state(
                following=False,
                target_name=None,
                lost=True,
                msg="主人走丢了",
                last_faces=[], last_ids=[],
            )
            with self.car._mode_lock:
                if self.car._mode == "follow":
                    self.car.motor.stop()
            # 只在刚进入"丢失"状态时播报一次, 避免重复刷屏
            if self._last_target_name is not None:
                self.car.voice_out.say("主人走丢了")
                self._last_target_name = None
            time.sleep(1.0)
