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
    FOLLOW_DETECT_WIDTH,
    FOLLOW_PAN_LIMIT,
    FOLLOW_SEARCH_TILTS,
    FOLLOW_RETRY_INTERVAL,
    SERVO_PAN_CENTER,
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

    def set_distance(self, dist):
        """更新缓存距离 (供 auto-pilot 线程在 auto 模式下调用)

        auto 模式不走 follower 线程，但 web 端 /api/distance 和 /api/status
        统一从 follower.state.distance 读取 (避免重复 measure 阻塞)。
        所以 auto-pilot 测距后需调用此方法同步距离，否则 web 端显示 -1。
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

            # 刚进入 follow 模式: 从此刻开始计时，
            # 否则 _last_seen_time=0 + lost_duration=0 永远不超时 (审查 bug 3.1)
            if not self._entered_follow:
                self._last_seen_time = time.time()
                self._last_target_name = None
                self._entered_follow = True
                self._search_tilt_idx = 0
                # 云台回正并微抬头: 主人站立时人脸在小车水平视线上方，
                # 平视根本看不到脸 ("仰视认不出"的一半根因是"看不见")
                if getattr(self.car.servo, "_initialized", False):
                    try:
                        self.car.servo.pan(SERVO_PAN_CENTER)
                        self.car.servo.tilt(FOLLOW_SEARCH_TILTS[0])
                    except Exception:
                        pass

            # 检查依赖
            if not self.car.face_recognizer.is_ready():
                self._set_state(msg="主人识别未就绪 (未安装 dlib 或主人库为空)",
                                last_faces=[], last_ids=[])
                with self.car._mode_lock:
                    if self.car._mode == "follow":
                        self.car.motor.stop()
                time.sleep(1.0)
                continue

            # 1. 抓帧 + 检测人脸 (降采样 320px，HOG 快 ~4 倍 → 控制周期缩短，
            #    减少"一冲就过头的"过冲 — 修复跟随不平滑的关键提速)
            # 性能优化 (2026-07-25): capture() 现在是零阻塞读缓存，
            # 但 dlib HOG 检测仍是 ~50ms CPU 重活，节流到每 ~200ms 一次
            frame = self.car.camera_csi.capture()
            if frame is None:
                self._set_state(msg="摄像头捕获失败", last_faces=[], last_ids=[])
                time.sleep(0.2)
                continue

            # 检测节流: 距上次检测 < 150ms 时复用上一帧结果，只更新控制 (云台+电机)
            # 这样检测周期 ~200ms，控制周期 ~50ms，CPU 占用降一半，过冲也减少
            now = time.time()
            DETECT_THROTTLE = 0.15
            if (now - self._last_detect_time) < DETECT_THROTTLE and \
                    self.state.get("last_faces") is not None:
                # 复用上次检测结果 (不跑 dlib)
                faces = self.state.get("last_faces") or []
                identifications = self.state.get("last_ids") or []
                my_owners = [(f, n) for f, n in zip(faces, identifications) if n]
            else:
                # 跑 dlib 检测
                self._last_detect_time = now
                faces = self.car.face_recognizer.detect_faces(
                    frame, detect_width=FOLLOW_DETECT_WIDTH)
                my_owners, identifications = self._identify_all(faces)

            # 仰视/小脸兜底: 快速检测没找到主人时，每 ~1s 做一次高灵敏检测
            # (upsample=1 检测小脸/仰角 + jittering=1 更鲁棒的 embedding)。
            # 主人站直时人脸在画面中变小且呈仰视角度 (下巴/鼻孔视角)，
            # HOG 正面检测器容易漏检 — 这是"贴脸能认出、站起来认不出"的根因。
            # 开销大 (~0.5-1s)，不能每帧做，限频重试。
            if not my_owners and \
                    (now - self._last_sensitive_try) > FOLLOW_RETRY_INTERVAL:
                self._last_sensitive_try = now
                retry_faces = self.car.face_recognizer.detect_faces(
                    frame, upsample=1, jittering=1)
                if retry_faces:
                    retry_owners, retry_ids = self._identify_all(retry_faces)
                    if retry_owners:
                        faces, my_owners, identifications = \
                            retry_faces, retry_owners, retry_ids
                        self._last_detect_time = now  # 高灵敏检测也更新节流时间

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
            # samples=3 (默认 5 阻塞 50~185ms 会拖长控制周期加剧过冲)
            dist = self.car.ultrasonic.measure(samples=3)
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
                # 防重复播报: 主人持续站在 20cm 内时每 0.5s 循环一次，
                # 不加限制会反复播报"前方有障碍"。限制 3s 播报一次
                now = time.time()
                if now - self._last_obstacle_warn_time > 3.0:
                    self.car.voice_out.say("前方有障碍")
                    self._last_obstacle_warn_time = now
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

            # 7. 云台主动追踪 (微调对准主人，车身移动时云台持续反向补偿保持目标居中)
            self._servo_track(offset_x, box_y + box_h / 2, h)

            self._set_state(
                following=True,
                target_name=target_name,
                lost=False,
                box_ratio=round(box_ratio, 3),
                offset_x=round(offset_x, 3),
                msg=f"跟随 {target_name}",
            )

            time.sleep(0.05)  # 检测提速后控制周期 ~200ms (~5 FPS)

    def _identify_all(self, faces):
        """对所有人脸识别身份，返回 (主人列表, 身份列表)

        Returns:
            (my_owners, identifications):
              my_owners: [(face, name), ...] 识别为主人的人脸
              identifications: [name|None, ...] 与 faces 等长
        """
        my_owners = []
        identifications = []
        for f in faces:
            name = self.car.face_recognizer.identify(f)
            identifications.append(name)
            if name:
                my_owners.append((f, name))
        return my_owners, identifications

    def _control(self, target_name, offset_x, box_ratio, frame_h):
        """根据偏差控制电机 ("云台先行 + 比例控制")

        之前 bug (根因): 车身和云台同时直接追人脸偏差 —
        云台 pan 微调和车身旋转是双控制器抢同一个误差信号，互相震荡；
        且一检测到位移就满幅 (y=60/rot=±50) 动作，控制周期又长 (~300ms)，
        一次冲过头人脸就出画面 → "冲出去就丢目标"。

        新策略:
          1. 云台先行: 人脸偏移先由 _servo_track 用云台吸收 (快速、无惯性)。
             只有云台偏到 FOLLOW_PAN_LIMIT(35°) 之外，说明靠云台跟不上，
             车身才按云台偏角比例旋转跟上 (车身转过去后云台自然回中)。
          2. 比例调速: 前进/后退速度随"距离偏差"线性变化，越接近目标越慢，
             不再一冲到底。
          3. 大角度旋转时暂停前后移动 (先对准再走近)，避免螺旋冲出去。

        offset_x: -1~1 (画面中心偏移)
        box_ratio: 0~1 (人脸框占画面宽度比)
        """
        target = FOLLOW_TARGET_BOX_RATIO
        too_far = box_ratio < target * 0.85
        too_close = box_ratio > target * 1.4
        just_right = target * 0.85 <= box_ratio <= target * 1.4

        # --- 车身旋转由云台偏角驱动 (云台先行) ---
        pan_err = 0
        if getattr(self.car.servo, "_initialized", False):
            try:
                cur_pan, _ = self.car.servo.get_angles()
                pan_err = cur_pan - SERVO_PAN_CENTER  # >0: 云台向右偏 → 主人在右
            except Exception:
                pan_err = 0

        rot_val = 0
        if abs(pan_err) > FOLLOW_PAN_LIMIT:
            # 云台跟不上了 → 车身按比例旋转 (偏角越大转越快，限幅 ±50)
            rot_val = int(max(-50, min(50, pan_err * 1.5)))
        elif abs(offset_x) > 0.55:
            # 云台失效(未初始化)时兜底: 人脸已严重偏出画面，直接按偏差比例旋转
            rot_val = int(offset_x * 45)

        # --- 前后移动按比例调速 ---
        y_val = 0
        if too_far:
            # 距离偏差越大走得越快 (30~60)，接近目标自动减速
            ratio_gap = target - box_ratio
            y_val = int(30 + min(30, ratio_gap * 150))
        elif too_close:
            ratio_gap = box_ratio - target
            y_val = -int(20 + min(20, ratio_gap * 100))

        # 大角度旋转时暂停前后移动: 先对准再走近，避免边转边冲画螺旋
        if abs(rot_val) > 30:
            y_val = 0

        with self.car._mode_lock:
            if self.car._mode != "follow":
                return
            # set_speed 在锁内执行 (审查 bug 1.2): 之前在锁外执行，可能与 set_mode 恢复用户速度竞态
            self.car.motor.set_speed(FOLLOW_SPEED)

            if y_val != 0 or rot_val != 0:
                self.car.motor.move(y=y_val, rotation=rot_val)
            else:
                # 距离和方向都合适 → 停下等主人
                self.car.motor.stop()
            # motor.move/stop 内部已有 motor._lock，此处 _mode_lock 短暂持有不嵌套

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
            # 短暂丢失 → 原地扫视找人 (水平扫 + 扫到尽头换仰角档位)
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
            # 云台左右扫；扫到尽头切换仰角档位 (平视→微抬头→高抬头)。
            # 主人站立时人脸在小车水平视线上方 ~30-45°，固定平视扫视
            # 永远扫不到脸 — 丢失重搜必须覆盖仰角维度。
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
                        # 换下一个仰角档位继续扫
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
