"""
web_server.py - Web 遥控服务器

Flask + 手机横屏双手控制界面
功能: 8方向全向移动、原地旋转、滑块调速、云台控制、实时画面、AI 视觉、模式切换
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
                 ultrasonic=None, vision=None,
                 on_mode_change=None):
        self._motor = motor
        self._servo = servo
        self._camera_csi = camera_csi
        self._camera_usb = camera_usb
        self._ultrasonic = ultrasonic
        self._vision = vision
        self._on_mode_change = on_mode_change

        self._app = Flask(__name__)

        self._mode = "manual"
        self._cam_source = "csi"

        self._latest_frame = None
        self._latest_detections = []
        self._frame_lock = threading.Lock()

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
                if self._on_mode_change:
                    self._on_mode_change(mode)
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
            if not self._vision:
                return jsonify({"detections": []})
            with self._frame_lock:
                return jsonify({"detections": self._latest_detections})

        @app.route("/api/status")
        def api_status():
            """综合状态接口，减少前端轮询次数"""
            dist = -1
            if self._ultrasonic:
                d = self._ultrasonic.measure()
                dist = round(d, 1) if d > 0 else -1
            return jsonify({
                "mode": self._mode,
                "distance": dist,
                "speed": self._motor.get_speed(),
                "camera": self._cam_source
            })

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    def _get_current_frame(self):
        if self._cam_source == "csi" and self._camera_csi:
            return self._camera_csi.capture()
        elif self._cam_source == "usb" and self._camera_usb:
            return self._camera_usb.capture()
        return None

    def _generate_frames(self):
        import cv2
        while True:
            frame = self._get_current_frame()
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame.copy()

                if self._vision and self._mode == "auto":
                    detections = self._vision.detect(frame)
                    with self._frame_lock:
                        self._latest_detections = detections
                    if detections:
                        frame = self._vision.draw_detections(frame, detections)

                _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')
            else:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')

            time.sleep(0.05)

    def start(self, threaded=True):
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
# HTML 控制界面 (手机横屏双手控制)
# ============================================================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>AI 小车控制台</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
:root {
  --bg: #0d0d1a; --panel: #16213e; --accent: #e94560;
  --accent2: #0f3460; --text: #eee; --green: #00e676;
  --btn: #1a1a3e; --btn-active: #e94560; --radius: 14px;
  /* 自适应尺寸单位 — 基于视口短边 vmin */
  --dpad-btn: min(11vmin, 64px);
  --gimbal-btn: min(8vmin, 44px);
  --rotate-w: min(16vmin, 80px);
  --rotate-h: min(8vmin, 48px);
}
html, body {
  width:100%; height:100%; overflow:hidden;
  font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
  background:var(--bg); color:var(--text);
  touch-action:none; user-select:none; -webkit-user-select:none;
}

/* ===== 竖屏旋转提示 ===== */
#rotate-hint {
  display:none; position:fixed; inset:0; z-index:9999;
  background:var(--bg); flex-direction:column;
  align-items:center; justify-content:center; text-align:center;
}
#rotate-hint .icon { font-size:64px; animation:rotate 2s ease-in-out infinite; }
#rotate-hint p { margin-top:16px; font-size:18px; color:var(--text); }
@keyframes rotate { 0%,100%{transform:rotate(0)} 50%{transform:rotate(90deg)} }
@media (orientation:portrait) {
  #rotate-hint { display:flex; }
  #main-app { display:none; }
}

/* ===== 主布局: 横屏三栏 (flex 弹性) ===== */
#main-app {
  display:flex;
  flex-direction:column;
  width:100%; height:100%;
  gap:4px; padding:4px;
}

/* ===== 顶栏 ===== */
.topbar {
  flex:0 0 auto; height:36px;
  display:flex; align-items:center; justify-content:space-between;
  background:var(--panel); border-radius:10px; padding:0 12px;
  font-size:13px;
}
.topbar-left { display:flex; align-items:center; gap:10px; }
.topbar-right { display:flex; align-items:center; gap:8px; }
.dist-badge {
  background:var(--accent2); padding:3px 10px; border-radius:16px;
  font-size:12px; color:var(--green); white-space:nowrap;
}
.dist-badge.warn { color:#ffb74d; }
.dist-badge.danger { color:var(--accent); animation:pulse 0.5s infinite; }
@keyframes pulse { 50% { opacity:0.5; } }

/* ===== 主体三栏 ===== */
.main-body {
  flex:1 1 auto; display:flex; gap:4px; min-height:0;
}

/* ===== 左栏: 方向控制 (左手) ===== */
.left-panel {
  flex:0 0 auto; width:min(26vmin, 220px);
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; gap:6px;
  background:var(--panel); border-radius:var(--radius); padding:6px;
}

/* 8方向 D-Pad */
.dpad {
  display:grid;
  grid-template-columns: repeat(3, var(--dpad-btn));
  grid-template-rows: repeat(3, var(--dpad-btn));
  gap:4px;
}
.dpad button {
  width:var(--dpad-btn); height:var(--dpad-btn); border:none; border-radius:12px;
  background:var(--btn); color:var(--text); font-size:calc(var(--dpad-btn) * 0.4);
  cursor:pointer; transition:all 0.08s; touch-action:manipulation;
  display:flex; align-items:center; justify-content:center;
  border:1px solid rgba(255,255,255,0.05);
}
.dpad button:active, .dpad button.active {
  background:var(--btn-active); color:#fff; transform:scale(0.9);
  box-shadow:0 0 12px rgba(233,69,96,0.5);
}
.dpad .stop-btn {
  background:var(--accent2); color:var(--accent); font-size:calc(var(--dpad-btn) * 0.35);
  border:1px solid var(--accent);
}
.dpad .stop-btn:active, .dpad .stop-btn.active {
  background:var(--accent); color:#fff;
}

.dpad-label {
  font-size:10px; color:#666; text-align:center; margin-top:2px;
}

/* ===== 中栏: 摄像头画面 ===== */
.center-panel {
  flex:1 1 auto; min-width:0;
  background:#000; border-radius:var(--radius); overflow:hidden;
  position:relative;
  display:flex; align-items:center; justify-content:center;
}
.center-panel img {
  width:100%; height:100%; object-fit:contain;
}
.cam-overlay-top {
  position:absolute; top:6px; left:6px; right:6px;
  display:flex; justify-content:space-between; pointer-events:none;
}
.cam-overlay-bottom {
  position:absolute; bottom:6px; left:50%; transform:translateX(-50%);
  display:flex; gap:6px;
}
.cam-btn {
  background:rgba(22,33,62,0.85); border:1px solid var(--accent2);
  color:var(--text); padding:4px 12px; border-radius:16px;
  font-size:12px; cursor:pointer; backdrop-filter:blur(4px);
  pointer-events:auto; transition:all 0.15s;
}
.cam-btn.active { background:var(--accent); border-color:var(--accent); }

/* ===== 右栏: 功能控制 (右手) ===== */
.right-panel {
  flex:0 0 auto; width:min(26vmin, 220px);
  display:flex; flex-direction:column; gap:4px;
  background:var(--panel); border-radius:var(--radius); padding:6px;
  overflow-y:auto; -webkit-overflow-scrolling:touch;
}

/* 旋转控制 */
.section-label {
  font-size:10px; color:#555; text-transform:uppercase;
  letter-spacing:1px; text-align:center; margin-bottom:2px;
}
.rotate-row {
  display:flex; gap:6px; justify-content:center;
}
.rotate-btn {
  width:var(--rotate-w); height:var(--rotate-h); border:none; border-radius:10px;
  background:var(--btn); color:var(--text); font-size:13px;
  cursor:pointer; transition:all 0.08s; touch-action:manipulation;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:2px; border:1px solid rgba(255,255,255,0.05);
}
.rotate-btn .icon { font-size:20px; }
.rotate-btn:active, .rotate-btn.active {
  background:var(--btn-active); transform:scale(0.92);
  box-shadow:0 0 10px rgba(233,69,96,0.4);
}

/* 速度滑块 */
.speed-section {
  display:flex; flex-direction:column; align-items:center; gap:4px;
}
.speed-row {
  display:flex; align-items:center; gap:6px; width:100%;
}
.speed-slider {
  flex:1; -webkit-appearance:none; appearance:none;
  height:36px; border-radius:8px; outline:none;
  background:linear-gradient(to right, var(--accent2), var(--accent));
  touch-action:pan-x; /* 允许水平拖动 */
  pointer-events:auto;
}
.speed-slider::-webkit-slider-thumb {
  -webkit-appearance:none; width:36px; height:36px;
  border-radius:50%; background:#fff; cursor:grab;
  box-shadow:0 2px 8px rgba(0,0,0,0.5); border:3px solid var(--accent);
}
.speed-slider:active::-webkit-slider-thumb { cursor:grabbing; transform:scale(1.15); }
.speed-slider::-moz-range-thumb {
  width:36px; height:36px; border-radius:50%; background:#fff;
  cursor:grab; border:3px solid var(--accent);
}
.speed-value {
  min-width:40px; text-align:center; font-size:16px; font-weight:bold;
  color:var(--accent);
}
.speed-label { font-size:10px; color:#555; }

/* 云台控制 */
.gimbal-grid {
  display:grid;
  grid-template-columns:repeat(3, var(--gimbal-btn));
  grid-template-rows:repeat(3, var(--gimbal-btn));
  gap:3px; justify-content:center; margin:0 auto;
}
.gimbal-grid button {
  width:var(--gimbal-btn); height:var(--gimbal-btn); border:none; border-radius:8px;
  background:var(--btn); color:var(--text); font-size:calc(var(--gimbal-btn) * 0.4);
  cursor:pointer; transition:all 0.08s; touch-action:manipulation;
  display:flex; align-items:center; justify-content:center;
  border:1px solid rgba(255,255,255,0.05);
}
.gimbal-grid button:active, .gimbal-grid button.active {
  background:var(--accent2); color:var(--accent); transform:scale(0.9);
}
.gimbal-grid .gimbal-center {
  background:transparent; border:1px dashed #333; font-size:10px; color:#444;
}
.gimbal-grid .gimbal-center:active { transform:none; background:transparent; }

/* 模式切换 */
.mode-row {
  display:flex; gap:4px; justify-content:center;
}
.mode-btn {
  flex:1; padding:8px 4px; border:1px solid var(--accent2);
  border-radius:8px; background:transparent; color:var(--text);
  font-size:12px; cursor:pointer; transition:all 0.15s;
  touch-action:manipulation;
}
.mode-btn.active {
  background:var(--accent); border-color:var(--accent); color:#fff;
  box-shadow:0 0 8px rgba(233,69,96,0.3);
}

/* 分隔线 */
.divider {
  width:100%; height:1px; background:rgba(255,255,255,0.06);
  margin:2px 0; flex:0 0 auto;
}

/* 超小屏幕适配 */
@media (max-height: 340px) {
  :root {
    --dpad-btn: min(10vmin, 52px);
    --gimbal-btn: min(7vmin, 36px);
  }
  .dpad-label, .section-label, .speed-label { display:none; }
  .divider { margin:1px 0; }
  .right-panel { padding:4px; gap:3px; }
  .left-panel { padding:4px; }
}
</style>
</head>
<body>

<!-- 竖屏旋转提示 -->
<div id="rotate-hint">
  <div class="icon">📱↻</div>
  <p>请横屏使用</p>
</div>

<!-- 主应用 -->
<div id="main-app">

  <!-- 顶栏 -->
  <div class="topbar">
    <div class="topbar-left">
      <span style="font-size:16px;">🚗</span>
      <span style="font-weight:600;">AI 小车</span>
      <span id="modeLabel" style="font-size:12px;color:var(--accent);">手动模式</span>
    </div>
    <div class="topbar-right">
      <span class="dist-badge" id="distDisplay">📡 --</span>
    </div>
  </div>

  <!-- 主体三栏 -->
  <div class="main-body">

  <!-- 左栏: 方向控制 (左手) -->
  <div class="left-panel">
    <div class="dpad-label">麦克纳姆轮 · 8方向</div>
    <div class="dpad">
      <button data-act="fl" ontouchstart="hold(-70,70,0,this)" ontouchend="release(this)" onmousedown="hold(-70,70,0,this)" onmouseup="release(this)" onmouseleave="release(this)">↖</button>
      <button data-act="f" ontouchstart="hold(0,100,0,this)" ontouchend="release(this)" onmousedown="hold(0,100,0,this)" onmouseup="release(this)" onmouseleave="release(this)">↑</button>
      <button data-act="fr" ontouchstart="hold(70,70,0,this)" ontouchend="release(this)" onmousedown="hold(70,70,0,this)" onmouseup="release(this)" onmouseleave="release(this)">↗</button>
      <button data-act="sl" ontouchstart="hold(-100,0,0,this)" ontouchend="release(this)" onmousedown="hold(-100,0,0,this)" onmouseup="release(this)" onmouseleave="release(this)">←</button>
      <button class="stop-btn" onclick="stopCar(this)">⏹</button>
      <button data-act="sr" ontouchstart="hold(100,0,0,this)" ontouchend="release(this)" onmousedown="hold(100,0,0,this)" onmouseup="release(this)" onmouseleave="release(this)">→</button>
      <button data-act="bl" ontouchstart="hold(-70,-70,0,this)" ontouchend="release(this)" onmousedown="hold(-70,-70,0,this)" onmouseup="release(this)" onmouseleave="release(this)">↙</button>
      <button data-act="b" ontouchstart="hold(0,-100,0,this)" ontouchend="release(this)" onmousedown="hold(0,-100,0,this)" onmouseup="release(this)" onmouseleave="release(this)">↓</button>
      <button data-act="br" ontouchstart="hold(70,-70,0,this)" ontouchend="release(this)" onmousedown="hold(70,-70,0,this)" onmouseup="release(this)" onmouseleave="release(this)">↘</button>
    </div>
  </div>

  <!-- 中栏: 摄像头 -->
  <div class="center-panel">
    <img id="cameraFeed" src="/video_feed" alt="摄像头画面">
    <div class="cam-overlay-top">
      <span id="aiStatus" style="font-size:11px;color:var(--green);background:rgba(0,0,0,0.5);padding:2px 8px;border-radius:10px;">● LIVE</span>
      <span></span>
    </div>
    <div class="cam-overlay-bottom">
      <button class="cam-btn active" id="csiBtn" onclick="switchCamera('csi')">CSI</button>
      <button class="cam-btn" id="usbBtn" onclick="switchCamera('usb')">USB</button>
    </div>
  </div>

  <!-- 右栏: 功能控制 (右手) -->
  <div class="right-panel">

    <!-- 旋转 -->
    <div class="section-label">原地旋转</div>
    <div class="rotate-row">
      <button class="rotate-btn" ontouchstart="hold(0,0,-70,this)" ontouchend="release(this)" onmousedown="hold(0,0,-70,this)" onmouseup="release(this)" onmouseleave="release(this)">
        <span class="icon">⟲</span>左转
      </button>
      <button class="rotate-btn" ontouchstart="hold(0,0,70,this)" ontouchend="release(this)" onmousedown="hold(0,0,70,this)" onmouseup="release(this)" onmouseleave="release(this)">
        <span class="icon">⟳</span>右转
      </button>
    </div>

    <div class="divider"></div>

    <!-- 速度 -->
    <div class="speed-section">
      <div class="speed-label">速度调节</div>
      <div class="speed-row">
        <span style="font-size:12px;color:#555;">慢</span>
        <input type="range" class="speed-slider" id="speedSlider" min="20" max="100" value="50"
               oninput="updateSpeed(this.value)" onchange="commitSpeed()">
        <span style="font-size:12px;color:#555;">快</span>
      </div>
      <div class="speed-value" id="speedDisplay">50%</div>
    </div>

    <div class="divider"></div>

    <!-- 云台 -->
    <div class="section-label">云台控制</div>
    <div class="gimbal-grid">
      <div></div>
      <button ontouchstart="gimbalTilt(-10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalTilt(-10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">↑</button>
      <div></div>
      <button ontouchstart="gimbalPan(-10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalPan(-10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">←</button>
      <button class="gimbal-center" onclick="gimbalCenter()">归中</button>
      <button ontouchstart="gimbalPan(10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalPan(10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">→</button>
      <div></div>
      <button ontouchstart="gimbalTilt(10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalTilt(10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">↓</button>
      <div></div>
    </div>

    <div class="divider"></div>

    <!-- 模式 -->
    <div class="section-label">运行模式</div>
    <div class="mode-row">
      <button class="mode-btn active" id="modeManual" onclick="setMode('manual')">🕹手动</button>
      <button class="mode-btn" id="modeAuto" onclick="setMode('auto')">🤖自动</button>
      <button class="mode-btn" id="modeVoice" onclick="setMode('voice')">🎤语音</button>
    </div>

  </div>
  </div><!-- /main-body -->
</div>

<script>
let speedValue = 50;
let currentPan = 90, currentTilt = 90;
let activeDir = null;
let speedSendTimer = null;

// ===== 按住移动 =====
async function hold(x, y, r, btn) {
  btn.classList.add('active');
  if (navigator.vibrate) navigator.vibrate(15);
  activeDir = btn;
  try {
    await fetch('/api/control', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({x, y, rotation:r, speed:speedValue})
    });
  } catch(e) {}
}

function release(btn) {
  btn.classList.remove('active');
  if (activeDir === btn) {
    activeDir = null;
    stopCar(null);
  }
}

function stopCar(btn) {
  if (btn) { btn.classList.add('active'); if(navigator.vibrate) navigator.vibrate(20); setTimeout(()=>btn.classList.remove('active'),200); }
  fetch('/api/stop', {method:'POST'}).catch(()=>{});
}

// ===== 速度滑块 (支持拖动) =====
function updateSpeed(val) {
  speedValue = parseInt(val);
  document.getElementById('speedDisplay').textContent = val + '%';
  // 拖动中只更新本地值，不发送请求（节流）
  if (speedSendTimer) clearTimeout(speedSendTimer);
  speedSendTimer = setTimeout(function() {
    sendSpeed(speedValue);
  }, 150);
}

function commitSpeed() {
  // 松手时立即发送最终值
  if (speedSendTimer) { clearTimeout(speedSendTimer); speedSendTimer = null; }
  sendSpeed(speedValue);
}

function sendSpeed(s) {
  fetch('/api/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({x:0, y:0, rotation:0, speed:s})
  }).catch(()=>{});
}

// ===== 云台 =====
async function gimbalPan(delta, btn) {
  btn.classList.add('active');
  if (navigator.vibrate) navigator.vibrate(10);
  currentPan = Math.max(0, Math.min(180, currentPan + delta));
  await fetch('/api/servo', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({pan: currentPan})
  }).catch(()=>{});
}

async function gimbalTilt(delta, btn) {
  btn.classList.add('active');
  if (navigator.vibrate) navigator.vibrate(10);
  currentTilt = Math.max(0, Math.min(180, currentTilt + delta));
  await fetch('/api/servo', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({tilt: currentTilt})
  }).catch(()=>{});
}

function gimbalRelease(btn) { btn.classList.remove('active'); }

async function gimbalCenter() {
  currentPan = 90; currentTilt = 90;
  await fetch('/api/servo', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({pan:90, tilt:90})
  }).catch(()=>{});
}

// ===== 模式 =====
async function setMode(mode) {
  await fetch('/api/mode', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mode})
  }).catch(()=>{});
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
  const names = {manual:'手动模式', auto:'自动模式', voice:'语音模式'};
  document.getElementById('modeLabel').textContent = names[mode];
  if (navigator.vibrate) navigator.vibrate(15);
}

// ===== 摄像头切换 =====
async function switchCamera(source) {
  await fetch('/api/camera', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({source})
  }).catch(()=>{});
  document.getElementById('csiBtn').classList.toggle('active', source==='csi');
  document.getElementById('usbBtn').classList.toggle('active', source==='usb');
}

// ===== 距离轮询 =====
async function updateDistance() {
  try {
    const r = await fetch('/api/distance');
    const d = await r.json();
    const el = document.getElementById('distDisplay');
    if (d.distance > 0) {
      el.textContent = '📡 ' + d.distance.toFixed(0) + 'cm';
      el.className = 'dist-badge' + (d.distance < 15 ? ' danger' : d.distance < 30 ? ' warn' : '');
    } else {
      el.textContent = '📡 --';
      el.className = 'dist-badge';
    }
  } catch(e) {}
}
setInterval(updateDistance, 800);

// ===== 防止触摸滚动/缩放 (但不阻止滑块拖动) =====
document.addEventListener('touchmove', function(e) {
  // 允许 speed-slider 上的触摸移动
  if (e.target && (e.target.id === 'speedSlider' || e.target.classList.contains('speed-slider'))) {
    return; // 不阻止
  }
  e.preventDefault();
}, {passive:false});
document.addEventListener('gesturestart', e => e.preventDefault());
document.addEventListener('dblclick', e => e.preventDefault());

// ===== 键盘控制 (电脑端) =====
const keyMap = {
  'ArrowUp':[0,100,0], 'w':[0,100,0], 'W':[0,100,0],
  'ArrowDown':[0,-100,0], 's':[0,-100,0], 'S':[0,-100,0],
  'ArrowLeft':[-100,0,0], 'a':[-100,0,0], 'A':[-100,0,0],
  'ArrowRight':[100,0,0], 'd':[100,0,0], 'D':[100,0,0],
  'q':[-70,70,0], 'Q':[-70,70,0],
  'e':[70,70,0], 'E':[70,70,0],
  'z':[-70,-70,0], 'Z':[-70,-70,0],
  'x':[70,-70,0], 'X':[70,-70,0],
};
const pressedKeys = new Set();
document.addEventListener('keydown', e => {
  if (pressedKeys.has(e.key)) return;
  if (e.key in keyMap) {
    e.preventDefault();
    pressedKeys.add(e.key);
    const [x,y,r] = keyMap[e.key];
    fetch('/api/control', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({x,y,rotation:r,speed:speedValue})}).catch(()=>{});
  }
  if (e.key === ' ') { e.preventDefault(); stopCar(null); }
});
document.addEventListener('keyup', e => {
  if (e.key in keyMap) { e.preventDefault(); pressedKeys.delete(e.key); stopCar(null); }
});
</script>
</body>
</html>'''
