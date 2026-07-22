#!/usr/bin/env python3
"""AI 小车 - CSI 摄像头录像模块

DEPRECATED: 由 camera.py (picamera2 方案) 取代，仅保留作独立调试工具。
主程序不引用本文件。如需录像测试可直接运行: python3 camera/video.py
"""

import subprocess


def record_video(filename="video.mp4", duration=10):
    """录制一段视频 (CSI 摄像头, 无音频)"""
    cmd = [
        "libcamera-vid",
        "-o", filename,
        "--width", "1920",
        "--height", "1080",
        "-t", str(duration * 1000),  # 毫秒
        "--codec", "libx264",
        "--preset", "ultrafast",
        "--nopreview",
    ]
    subprocess.run(cmd, check=True)
    print(f"🎥 已录像: {filename} ({duration}秒)")


def record_video_lowres(filename="video_low.mp4", duration=10):
    """录制低分辨率视频（省空间）"""
    cmd = [
        "libcamera-vid",
        "-o", filename,
        "--width", "640",
        "--height", "480",
        "-t", str(duration * 1000),
        "--codec", "libx264",
        "--preset", "ultrafast",
        "--nopreview",
    ]
    subprocess.run(cmd, check=True)
    print(f"🎥 已录像(低分辨率): {filename} ({duration}秒)")


if __name__ == "__main__":
    try:
        record_video(duration=5)
    except KeyboardInterrupt:
        print("\n已中断录像")
    except subprocess.CalledProcessError as e:
        print(f"录像失败: {e}")
