#!/usr/bin/env python3
"""AI 小车 - 摄像头拍照模块"""

import subprocess


def take_photo(filename="photo.jpg", resolution="1920x1080"):
    """用 USB 摄像头拍照"""
    cmd = [
        "fswebcam",
        "-d", "/dev/video0",
        "-r", resolution,
        "--jpeg", "95",
        "-D", "1",
        "--no-banner",
        filename,
    ]
    subprocess.run(cmd, check=True)
    print(f"📸 已拍照: {filename} ({resolution})")


def take_photo_lowlight(filename="photo_night.jpg"):
    """暗光环境拍照（拉高增益+曝光）"""
    # 先调摄像头参数
    subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "auto_exposure=1"], check=False)
    subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "exposure_time_absolute=50000"], check=False)
    subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "gain=7"], check=False)

    cmd = [
        "fswebcam",
        "-d", "/dev/video0",
        "-r", "1280x720",
        "--jpeg", "95",
        "-D", "1",
        "--no-banner",
        filename,
    ]
    subprocess.run(cmd, check=True)
    print(f"🌙 暗光拍照: {filename}")

    # 恢复自动曝光
    subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "auto_exposure=3"], check=False)


if __name__ == "__main__":
    take_photo()
