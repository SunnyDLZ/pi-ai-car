"""
camera.py - CSI 摄像头管理

CSI 摄像头: picamera2 库 (树莓派原生)
"""

import threading
import time
import numpy as np
from config import CSI_FRAME_WIDTH, CSI_FRAME_HEIGHT, CSI_FRAME_RATE


class CSICamera:
    """CSI 摄像头 (picamera2)"""

    def __init__(self):
        self._camera = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

    def init(self):
        """初始化 CSI 摄像头

        仅当检测到树莓派 CSI 接口上有相机模块时才初始化。
        """
        # 检测 CSI 相机模块是否存在 (设备树 i2c 节点)
        import os
        csi_found = False
        for root, dirs, files in os.walk("/sys/bus/i2c/devices"):
            for name in dirs:
                if any(s in name.lower() for s in ("imx", "ov", "camera")):
                    csi_found = True
                    break
            if csi_found:
                break
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
            print(f"[CSICamera] 初始化完成 ({CSI_FRAME_WIDTH}x{CSI_FRAME_HEIGHT})")
            return True
        except Exception as e:
            print(f"[CSICamera] 初始化失败: {e}")
            return False

    def start(self):
        """启动摄像头"""
        if self._camera and not self._running:
            self._camera.start()
            self._running = True
            print("[CSICamera] 已启动")

    def stop(self):
        """停止摄像头"""
        self._running = False
        if self._camera:
            self._camera.stop()
            print("[CSICamera] 已停止")

    def capture(self):
        """捕获一帧

        Returns:
            numpy.ndarray: RGB 图像, 失败返回 None
        """
        if not self._running or not self._camera:
            return None
        try:
            return self._camera.capture_array()
        except Exception as e:
            print(f"[CSICamera] 捕获失败: {e}")
            return None

    def cleanup(self):
        self.stop()
        if self._camera:
            self._camera.close()
        print("[CSICamera] 资源已释放")
