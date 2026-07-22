#!/usr/bin/env python3
"""AI 小车 - CSI 摄像头拍照模块

DEPRECATED: 由 camera.py (picamera2 方案) 取代，仅保留作独立调试工具。
主程序不引用本文件。如需拍照测试可直接运行: python3 camera/photo.py
"""

import subprocess


def take_photo(filename="photo.jpg", resolution="1920x1080"):
    """用 CSI 摄像头拍照"""
    cmd = [
        "libcamera-still",
        "-o", filename,
        "--width", resolution.split("x")[0],
        "--height", resolution.split("x")[1],
        "-q", "95",
        "-t", "1000",  # 预热 1 秒
        "--nopreview",
    ]
    subprocess.run(cmd, check=True)
    print(f"📸 已拍照: {filename} ({resolution})")


def take_photo_lowlight(filename="photo_night.jpg"):
    """暗光环境拍照（拉高增益+长曝光）"""
    cmd = [
        "libcamera-still",
        "-o", filename,
        "--width", "1280",
        "--height", "720",
        "-q", "95",
        "-t", "2000",      # 预热 2 秒
        "--nopreview",
        "--shutter", "50000",  # 长曝光 50ms
        "--gain", "7",         # 高增益
    ]
    subprocess.run(cmd, check=True)
    print(f"🌙 暗光拍照: {filename}")


if __name__ == "__main__":
    take_photo()
