"""
web_server.py - Web 遥控服务器

Flask + 移动端友好的控制界面
功能: 方向盘、云台、实时画面、AI 视觉、模式切换
"""

import json
import base64
import threading
import time
import io
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

from config import WEB_HOST, WEB_PORT
from motor import MotorController
from servo import ServoGimbal


class WebServer:
    """Web 遥控服务器"""

    def __init__(self, motor: MotorController, servo: ServoGimbal,
                 camera_csi=None, camera_usb=None,
                 ultrasonic=None, vision=None):
        self._motor = motor
        self._servo = servo
        self._camera_csi = camera_csi
        self._camera_usb = camera_usb
        self._ultrasonic = ultrasonic
        self._vision = vision

        self._app = Flask(__name__)

        # 运行模式: manual / auto / voice
        self._mode = "manual"

        # 当前摄像头源: "csi" 或 "usb"
        self._cam_source = "csi"

        # 最新帧 (for MJPEG stream)
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        app = self._app

        @app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @app.route("/api/control", methods=["POST"])
        def control():
            data = request.get_json()
            x = float(data.get("x", 0))
            y = float(data.get("y", 0))
            r = float(data.get("rotation", 0))
            s = float(data.get("speed", 50))

            self._motor.set_speed(s)
            self._motor.move(x, y, r)
            return jsonify({"status": "ok"})

        @app.route("/api/stop", methods=["POST"])
        def api_stop():
            self._motor.stop()
            return jsonify({"status": "ok"})

        @app.route("/api/servo", methods=["POST"])
        def api_servo():
            data = request.get_json()
            pan = data.get("pan")
            tilt = data.get("tilt")
            if pan is not None:
                self._servo.pan(int(pan))
            if tilt is not None:
                self._servo.tilt(int(tilt))
            return jsonify({"status": "ok"})

        @app.route("/api/distance")
        def api_distance():
            if self._ultrasonic:
                d = self._ultrasonic.measure()
                return jsonify({"distance": round(d, 1) if d > 0 else -1})
            return jsonify({"distance": -1})

        @app.route("/api/mode", methods=["POST"])
        def api_mode():
            data = request.get_json()
            mode = data.get("mode", "manual")
            if mode in ("manual", "auto", "voice"):
                self._mode = mode
                return jsonify({"status": "ok", "mode": mode})
            return jsonify({"status": "error", "msg": "无效模式"}), 400

        @app.route("/api/mode")
        def get_mode():
            return jsonify({"mode": self._mode})

        @app.route("/api/camera", methods=["POST"])
        def api_camera():
            data = request.get_json()
            source = data.get("source", "csi")
            if source in ("csi", "usb"):
                self._cam_source = source
                return jsonify({"status": "ok", "source": source})
            return jsonify({"status": "error"}), 400

        @app.route("/api/detect")
        def api_detect():
            """对当前帧做物体检测"""
            if not self._vision:
                return jsonify({"detections": []})

            frame = self._latest_frame
            if frame is None:
                return jsonify({"detections": []})

            detections = self._vision.detect(frame)
            return jsonify({"detections": detections})

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    def _get_current_frame(self):
        """获取当前摄像头帧"""
        if self._cam_source == "csi" and self._camera_csi:
            return self._camera_csi.capture()
        elif self._cam_source == "usb" and self._camera_usb:
            return self._camera_usb.capture()
        return None

    def _generate_frames(self):
        """MJPEG 视频流生成器"""
        import cv2
        while True:
            frame = self._get_current_frame()
            if frame is not None:
                # 保存最新帧供检测使用
                with self._frame_lock:
                    self._latest_frame = frame.copy()

                # 可选: 如果 AI 视觉开启且检测到目标, 绘制框
                if self._vision and self._mode == "auto":
                    detections = self._vision.detect(frame)
                    if detections:
                        frame = self._vision.draw_detections(frame, detections)

                # 编码为 JPEG
                _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')
            else:
                # 无画面时显示等待帧
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')

            time.sleep(0.05)

    def start(self, threaded=True):
        """启动 Web 服务器 (非阻塞)"""
        threading.Thread(
            target=lambda: self._app.run(
                host=WEB_HOST, port=WEB_PORT,
                debug=False, use_reloader=False
            ),
            daemon=True
        ).start()
        print(f"[WebServer] 控制面板: http://{WEB_HOST}:{WEB_PORT}/")
        print(f"[WebServer] 在局域网其他设备上访问: http://<树莓派IP>:{WEB_PORT}/")


# ============================================================
# HTML 控制界面 (移动端自适应)
# ============================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no">
<title>AI 小车控制台</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e; color: #eee;
    overflow-x: hidden; height: 100vh;
  }
  .container {
    display: grid;
    grid-template-columns: 1fr 280px 1fr;
    grid-template-rows: 60px 1fr 120px;
    gap: 10px; padding: 10px;
    height: 100vh; max-width: 100vw;
  }
  .header {
    grid-column: 1/-1;
    display: flex; align-items: center; justify-content: space-between;
    background: #16213e; border-radius: 12px; padding: 0 20px;
  }
  .header h1 { font-size: 18px; }
  .header .status { font-size: 13px; color: #0f0; }
  .camera-panel {
    grid-column: 1/-1;
    background: #000; border-radius: 12px; overflow: hidden;
    position: relative; min-height: 0;
    display: flex; align-items: center; justify-content: center;
  }
  .camera-panel img {
    width: 100%; height: 100%; object-fit: contain;
  }
  .camera-overlay {
    position: absolute; bottom: 10px; left: 10px; right: 10px;
    display: flex; justify-content: space-between;
  }
  .control-area {
    grid-column: 1/-1;
    display: flex; gap: 10px; align-items: center; justify-content: center;
  }
  .joystick-area {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  .joystick-grid {
    display: grid;
    grid-template-columns: 60px 60px 60px;
    grid-template-rows: 60px 60px 60px;
    gap: 3px;
  }
  .joystick-grid button {
    width: 60px; height: 60px;
    border: none; border-radius: 12px;
    font-size: 22px; cursor: pointer;
    background: #16213e; color: #0f3460;
    transition: all 0.1s; touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
    display: flex; align-items: center; justify-content: center;
  }
  .joystick-grid button:active {
    background: #e94560; color: #fff; transform: scale(0.92);
  }
  .joystick-grid .center-btn {
    background: #0f3460; color: #e94560; font-size: 28px;
  }
  .side-controls {
    display: flex; flex-direction: column; gap: 8px; align-items: center;
  }
  .side-controls label { font-size: 12px; color: #888; text-align: center; }
  .side-controls input[type=range] {
    width: 100px; height: 4px; -webkit-appearance: none;
    background: #0f3460; border-radius: 2px; outline: none;
  }
  .side-controls input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 20px; height: 20px;
    border-radius: 50%; background: #e94560; cursor: pointer;
  }
  .servo-controls {
    display: flex; gap: 8px; flex-direction: column; align-items: center;
  }
  .servo-row { display: flex; gap: 4px; align-items: center; }
  .servo-row button {
    width: 36px; height: 36px; border: none; border-radius: 8px;
    background: #16213e; color: #e94560; font-size: 16px; cursor: pointer;
  }
  .servo-row button:active { background: #e94560; color: #fff; }
  .btn-mode { padding: 6px 14px; border-radius: 20px; border: 1px solid #0f3460;
    background: transparent; color: #eee; font-size: 13px; cursor: pointer; }
  .btn-mode.active { background: #e94560; border-color: #e94560; }
  .distance-badge {
    background: #16213e; padding: 4px 12px; border-radius: 20px;
    font-size: 13px; color: #0f0;
  }
  @media (min-width: 768px) {
    .container {
      grid-template-columns: 1fr 400px 250px;
      grid-template-rows: 60px 1fr;
    }
    .camera-panel { grid-column: 2; grid-row: 2; }
    .control-area { grid-column: 3; grid-row: 2; flex-direction: column; }
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🚗 AI 小车</h1>
    <div>
      <span class="distance-badge" id="distDisplay">📡 -- cm</span>
      <span id="modeDisplay" class="btn-mode active">手动</span>
    </div>
  </div>

  <div class="camera-panel">
    <img id="cameraFeed" src="/video_feed" alt="摄像头画面">
    <div class="camera-overlay">
      <button class="btn-mode" onclick="switchCamera('csi')">CSI</button>
      <button class="btn-mode" onclick="switchCamera('usb')">USB</button>
    </div>
  </div>

  <div class="control-area">
    <!-- 方向控制 -->
    <div class="joystick-area">
      <div class="joystick-grid">
        <div></div>
        <button data-dir="forward" ontouchstart="moveCar(0,-100,0)" ontouchend="stopCar()">▲</button>
        <div></div>
        <button data-dir="left" ontouchstart="moveCar(-100,0,0)" ontouchend="stopCar()">◀</button>
        <button class="center-btn" onclick="stopCar()">●</button>
        <button data-dir="right" ontouchstart="moveCar(100,0,0)" ontouchend="stopCar()">▶</button>
        <div></div>
        <button data-dir="backward" ontouchstart="moveCar(0,100,0)" ontouchend="stopCar()">▼</button>
        <div></div>
      </div>
    </div>

    <!-- 旋转 + 速度 -->
    <div class="side-controls">
      <div style="display:flex; gap:4px;">
        <button class="btn-mode" ontouchstart="moveCar(0,0,-60)" ontouchend="stopCar()">⟳</button>
        <button class="btn-mode" ontouchstart="moveCar(0,0,60)" ontouchend="stopCar()">⟳</button>
      </div>
      <div>
        <label>速度</label>
        <input type="range" id="speedSlider" min="20" max="100" value="50"
               oninput="updateSpeed(this.value)">
        <span id="speedDisplay" style="font-size:13px;">50</span>
      </div>
    </div>

    <!-- 云台控制 -->
    <div class="servo-controls">
      <div class="servo-row">
        <button ontouchstart="servoPan(-10)" ontouchend="servoStop()">←</button>
        <span style="font-size:12px;color:#888;">云台</span>
        <button ontouchstart="servoPan(10)" ontouchend="servoStop()">→</button>
      </div>
      <div class="servo-row">
        <button ontouchstart="servoTilt(-10)" ontouchend="servoStop()">↑</button>
        <span style="font-size:12px;color:#888;">俯仰</span>
        <button ontouchstart="servoTilt(10)" ontouchend="servoStop()">↓</button>
      </div>
      <button class="btn-mode" onclick="servoCenter()">归中</button>
    </div>

    <!-- 模式切换 -->
    <div style="display:flex; gap:6px; flex-wrap:wrap; justify-content:center;">
      <button class="btn-mode active" onclick="setMode('manual')">🕹 手动</button>
      <button class="btn-mode" onclick="setMode('auto')">🤖 自动</button>
      <button class="btn-mode" onclick="setMode('voice')">🎤 语音</button>
    </div>
  </div>
</div>

<script>
let speedValue = 50;
let currentPan = 90, currentTilt = 90;

async function moveCar(x, y, r) {
  await fetch('/api/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x, y, rotation: r, speed: speedValue})
  });
}

function stopCar() {
  fetch('/api/stop', {method: 'POST'});
}

function updateSpeed(val) {
  speedValue = val;
  document.getElementById('speedDisplay').textContent = val;
}

async function servoPan(delta) {
  currentPan = Math.max(0, Math.min(180, currentPan + delta));
  await fetch('/api/servo', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pan: currentPan})
  });
}

async function servoTilt(delta) {
  currentTilt = Math.max(0, Math.min(180, currentTilt + delta));
  await fetch('/api/servo', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tilt: currentTilt})
  });
}

function servoStop() {}

async function servoCenter() {
  currentPan = 90; currentTilt = 90;
  await fetch('/api/servo', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pan: 90, tilt: 90})
  });
}

async function setMode(mode) {
  await fetch('/api/mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode})
  });
  document.querySelectorAll('.btn-mode[onclick*="setMode"]').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  const names = {manual: '🕹 手动', auto: '🤖 自动', voice: '🎤 语音'};
  document.getElementById('modeDisplay').textContent = names[mode] || mode;
}

async function switchCamera(source) {
  await fetch('/api/camera', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source})
  });
}

// 实时更新超声波距离
async function updateDistance() {
  try {
    const r = await fetch('/api/distance');
    const d = await r.json();
    document.getElementById('distDisplay').textContent =
      d.distance > 0 ? `📡 ${d.distance} cm` : '📡 -- cm';
  } catch(e) {}
}
setInterval(updateDistance, 1000);

// 防止触摸滚动
document.addEventListener('touchmove', e => e.preventDefault(), {passive: false});
</script>
</body>
</html>
'''
