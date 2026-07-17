#!/usr/bin/env python3
"""AI 小车 - 摄像头录像模块"""

import subprocess


def record_video(filename="video.mp4", duration=10):
    """录制一段视频（带音频）"""
    cmd = [
        "ffmpeg",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-video_size", "1920x1080",
        "-i", "/dev/video0",
        "-f", "alsa",
        "-ac", "1",
        "-i", "plughw:3,0",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-y",
        filename,
    ]
    subprocess.run(cmd, check=True)
    print(f"🎥 已录像: {filename} ({duration}秒)")


def record_video_lowres(filename="video_low.mp4", duration=10):
    """录制低分辨率视频（省空间）"""
    cmd = [
        "ffmpeg",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-video_size", "640x480",
        "-i", "/dev/video0",
        "-f", "alsa",
        "-ac", "1",
        "-i", "plughw:3,0",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac",
        "-y",
        filename,
    ]
    subprocess.run(cmd, check=True)
    print(f"🎥 已录像(低分辨率): {filename} ({duration}秒)")


if __name__ == "__main__":
    record_video(duration=5)
