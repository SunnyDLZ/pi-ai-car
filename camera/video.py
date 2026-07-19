#!/usr/bin/env python3
"""AI 小车 - CSI 摄像头录像模块"""

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
    record_video(duration=5)
