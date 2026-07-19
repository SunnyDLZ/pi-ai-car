"""
camera.py - CSI + USB 双摄像头管理

CSI 摄像头: picamera2 库 (树莓派原生)
USB 摄像头: OpenCV (通用)
"""

import threading
import time
import numpy as np
from config import CSI_FRAME_WIDTH, CSI_FRAME_HEIGHT, CSI_FRAME_RATE, \
    USB_FRAME_WIDTH, USB_FRAME_HEIGHT, USB_CAMERA_ID


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
        无 CSI 硬件时直接跳过，避免 Picamera2/libcamera 枚举时
        误占用 USB 摄像头设备节点导致 USB 摄像头无法打开。
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


class USBCamera:
    """USB 摄像头 (OpenCV)"""

    def __init__(self):
        self._cap = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

    def init(self, camera_id=USB_CAMERA_ID):
        """初始化 USB 摄像头"""
        import cv2
        # 明确指定 V4L2 后端，避免 OpenCV 的 obsensor (Orbbec) 后端
        # 误把普通 UVC 摄像头识别为深度相机导致 "Camera index out of range"
        try:
            self._cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        except Exception:
            self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            print(f"[USBCamera] 无法打开摄像头 #{camera_id}")
            return False

        # 设置分辨率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_FRAME_HEIGHT)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[USBCamera] 初始化完成 ({actual_w}x{actual_h})")

        # 预热 (丢弃前几帧)
        for _ in range(10):
            self._cap.read()

        return True

    def start(self):
        """启动连续取帧线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[USBCamera] 取帧线程已启动")

    def _capture_loop(self):
        """后台取帧循环"""
        import cv2
        while self._running and self._cap:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                time.sleep(0.01)

    def capture(self):
        """获取最新帧

        Returns:
            numpy.ndarray: RGB 图像, 失败返回 None
        """
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def stop(self):
        """停止取帧"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        print("[USBCamera] 已停止")

    def cleanup(self):
        self.stop()
        if self._cap:
            self._cap.release()
        print("[USBCamera] 资源已释放")
