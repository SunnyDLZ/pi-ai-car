"""
camera.py - CSI 摄像头管理

CSI 摄像头: picamera2 库 (树莓派原生)

性能优化 (2026-07-25):
  - 后台采集线程持续 capture_array，所有消费者 (视频流/auto-pilot/follower)
    共享同一帧缓存，避免多线程并发调 capture_array 导致 picamera2 内部状态错乱
  - 帧缓存用双缓冲 + Lock，读取零拷贝 (返回引用)
  - 采集失败计数 + 自动重启，自愈黑屏
  - capture() 仅读缓存 (~0ms)，不再阻塞调用方线程
"""

import threading
import os
import time
import numpy as np
from config import CSI_FRAME_WIDTH, CSI_FRAME_HEIGHT, CSI_FRAME_RATE, CSI_FLIP_180


class CSICamera:
    """CSI 摄像头 (picamera2) — 后台采集 + 帧缓存"""

    def __init__(self):
        self._camera = None
        self._running = False
        self._initialized = False

        # 帧缓存 (双缓冲: 后台写, 前台读)
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # 后台采集线程
        self._capture_thread = None
        self._capture_interval = 1.0 / max(1, CSI_FRAME_RATE)  # 30fps → 33ms

        # 采集失败计数 (用于自动重启)
        self._fail_streak = 0

    def init(self):
        """初始化 CSI 摄像头

        仅当检测到树莓派 CSI 接口上有相机模块时才初始化。
        """
        if self._initialized:
            return True

        csi_found = self._detect_csi_camera()
        if not csi_found:
            print("[CSICamera] 未检测到 CSI 相机模块，跳过初始化")
            return False

        try:
            from picamera2 import Picamera2
            self._camera = Picamera2()

            config = self._camera.create_video_configuration(
                main={"size": (CSI_FRAME_WIDTH, CSI_FRAME_HEIGHT),
                      "format": "RGB888"},
                controls={"FrameRate": CSI_FRAME_RATE}
            )
            self._camera.configure(config)
            self._initialized = True
            print(f"[CSICamera] 初始化完成 ({CSI_FRAME_WIDTH}x{CSI_FRAME_HEIGHT})")
            return True
        except Exception as e:
            print(f"[CSICamera] 初始化失败: {e}")
            if self._camera is not None:
                try:
                    self._camera.close()
                except Exception:
                    pass
            self._camera = None
            return False

    @staticmethod
    def _detect_csi_camera():
        """检测树莓派 CSI 摄像头"""
        i2c_dir = "/sys/bus/i2c/devices"
        keywords = ("imx", "ov", "camera", "mt9", "tcs", "adv", "tvp", "ov5647", "ov9281")

        if os.path.isdir(i2c_dir):
            for entry in os.listdir(i2c_dir):
                name_path = os.path.join(i2c_dir, entry, "name")
                if not os.path.exists(name_path):
                    continue
                try:
                    with open(name_path, "r") as f:
                        dev_name = f.read().lower().strip()
                    if any(kw in dev_name for kw in keywords):
                        return True
                except Exception:
                    continue

        if os.path.exists("/dev/video0"):
            return True
        return False

    def start(self):
        """启动摄像头 + 后台采集线程"""
        if not self._camera:
            print("[CSICamera] start() 失败: 相机未初始化")
            return False
        if self._running:
            return True
        try:
            self._camera.start()
            self._running = True
            # 启动后台采集线程 (持续 capture_array 写入 _latest_frame)
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._capture_thread.start()
            print("[CSICamera] 已启动 (后台采集线程运行中)")
            return True
        except Exception as e:
            print(f"[CSICamera] start() 失败: {e}")
            return False

    def _capture_loop(self):
        """后台采集线程 — 持续 capture_array，写入帧缓存

        关键: picamera2.capture_array() 不是线程安全的，由本线程独占调用，
        所有消费者通过 capture() 读 _latest_frame，避免并发状态错乱。
        """
        while self._running and self._camera:
            try:
                frame = self._camera.capture_array()
                if frame is not None:
                    # 180° 翻转 (物理倒装) — numpy 切片是零拷贝 view，
                    # ascontiguousarray 确保内存连续供 cv2/dlib 使用
                    if CSI_FLIP_180:
                        frame = np.ascontiguousarray(frame[::-1, ::-1])
                    # 写帧缓存 (短锁，仅交换引用)
                    with self._frame_lock:
                        self._latest_frame = frame
                    self._fail_streak = 0
                else:
                    self._fail_streak += 1
            except Exception as e:
                print(f"[CSICamera] 采集异常: {e}")
                self._fail_streak += 1
                # 连续失败 5 次 → 重启摄像头自愈
                if self._fail_streak >= 5:
                    print(f"[CSICamera] 连续 {self._fail_streak} 次失败，尝试重启")
                    self._restart()
                # 退避避免异常循环吃满 CPU
                time.sleep(0.05)
                continue

            # 控制采集速率 (~30fps，CPU 友好)
            time.sleep(self._capture_interval)

    def _restart(self):
        """重启摄像头 (后台线程内调用)"""
        try:
            try:
                self._camera.stop()
            except Exception:
                pass
            time.sleep(0.2)
            self._camera.start()
            self._fail_streak = 0
            print("[CSICamera] 摄像头重启完成")
        except Exception as e:
            print(f"[CSICamera] 摄像头重启失败: {e}")
            self._fail_streak = 0

    def capture(self):
        """读取最新一帧 (零阻塞)

        Returns:
            numpy.ndarray: RGB 图像, 失败/未就绪返回 None

        性能: 仅读 _latest_frame 引用 (~0ms)，不调 capture_array，
        多线程并发调用安全且无锁竞争。
        """
        if not self._running:
            return None
        with self._frame_lock:
            # 返回引用 (下游只读; 若需修改应自行 copy)
            return self._latest_frame

    def stop(self):
        """停止摄像头"""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        self._capture_thread = None
        if self._camera:
            try:
                self._camera.stop()
                print("[CSICamera] 已停止")
            except Exception as e:
                print(f"[CSICamera] stop() 异常 (可忽略): {e}")

    def cleanup(self):
        self._running = False
        self._initialized = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        self._capture_thread = None
        if self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        with self._frame_lock:
            self._latest_frame = None
        print("[CSICamera] 资源已释放")
