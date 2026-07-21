"""
camera.py - CSI 摄像头管理

CSI 摄像头: picamera2 库 (树莓派原生)
"""

import threading
import os
import time
import numpy as np
from config import CSI_FRAME_WIDTH, CSI_FRAME_HEIGHT, CSI_FRAME_RATE


class CSICamera:
    """CSI 摄像头 (picamera2)"""

    def __init__(self):
        self._camera = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()      # 保护 capture_array 调用 (多线程并发会崩)
        self._thread = None
        self._initialized = False

    def init(self):
        """初始化 CSI 摄像头

        仅当检测到树莓派 CSI 接口上有相机模块时才初始化。
        """
        if self._initialized:
            return True

        # 检测 CSI 相机模块是否存在
        # /sys/bus/i2c/devices/<bus>-<addr>/name 文件里是传感器型号 (如 "imx219")
        # 之前的代码错把目录名当作型号匹配，几乎永远找不到
        csi_found = self._detect_csi_camera()
        if not csi_found:
            print("[CSICamera] 未检测到 CSI 相机模块，跳过初始化 (不占用摄像头设备)")
            return False

        try:
            from picamera2 import Picamera2
            self._camera = Picamera2()

            # 配置视频流
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
            self._camera = None
            return False

    @staticmethod
    def _detect_csi_camera():
        """检测树莓派 CSI 摄像头

        优先用 i2c 设备 name 文件 (Bookworm)，找不到时回退到检查 /dev/video*
        """
        i2c_dir = "/sys/bus/i2c/devices"
        # 关键字覆盖常见树莓派 CSI 传感器 (Sony IMX 系列、OmniVision OV 系列、
        # 以及一些第三方模块的 name 可能是 "camera" 或含 "csi")
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

        # 回退: 检查 /dev/video* 是否存在 (libcamera 会创建 video 设备节点)
        if os.path.exists("/dev/video0"):
            return True

        return False

    def start(self):
        """启动摄像头，返回是否成功启动"""
        if not self._camera:
            print("[CSICamera] start() 失败: 相机未初始化")
            return False
        if self._running:
            return True
        try:
            self._camera.start()
            self._running = True
            print("[CSICamera] 已启动")
            return True
        except Exception as e:
            print(f"[CSICamera] start() 失败: {e}")
            return False

    def stop(self):
        """停止摄像头"""
        self._running = False
        if self._camera:
            try:
                self._camera.stop()
                print("[CSICamera] 已停止")
            except Exception as e:
                print(f"[CSICamera] stop() 异常 (可忽略): {e}")

    def capture(self):
        """捕获一帧 (线程安全)

        Returns:
            numpy.ndarray: RGB 图像, 失败返回 None
        """
        if not self._running or not self._camera:
            return None
        # picamera2.capture_array() 不是线程安全的，多线程并发 (如视频流 + 采集)
        # 会导致内部状态错乱或返回损坏 buffer，用锁串行化
        with self._lock:
            try:
                return self._camera.capture_array()
            except Exception as e:
                print(f"[CSICamera] 捕获失败: {e}")
                return None

    def cleanup(self):
        self._running = False
        self._initialized = False
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
        print("[CSICamera] 资源已释放")
