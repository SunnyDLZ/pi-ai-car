"""
web_server.py - Web 遥控服务器

Flask + 手机横屏双手控制界面
功能: 8方向全向移动、原地旋转、滑块调速、云台控制、实时画面、AI 视觉、模式切换
"""

import json
import base64
import threading
import time
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

from config import WEB_HOST, WEB_PORT
from motor import MotorController
from servo import ServoGimbal


class WebServer:
    """Web 遥控服务器"""

    def __init__(self, motor: MotorController, servo: ServoGimbal,
                 camera_csi=None,
                 ultrasonic=None, vision=None,
                 on_mode_change=None):
        self._motor = motor
        self._servo = servo
        self._camera_csi = camera_csi
        self._ultrasonic = ultrasonic
        self._vision = vision
        self._on_mode_change = on_mode_change

        self._app = Flask(__name__)

        self._mode = "manual"

        self._latest_frame = None
        self._latest_detections = []
        self._frame_lock = threading.Lock()

        self._register_routes()

    def set_mode(self, mode):
        """外部同步模式状态 (不触发回调，避免递归)

        供 AICar.set_mode 在语音/自动切换模式时同步 WebServer._mode，
        使 /api/control 的模式检查和前端按钮状态保持正确。
        """
        if mode in ("manual", "auto", "voice"):
            self._mode = mode

    def _register_routes(self):
        app = self._app

        @app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @app.route("/api/control", methods=["POST"])
        def control():
            # 非 manual 模式下忽略手动控制请求：
            # 1) 避免覆盖自动模式的 30% 安全限速 (set_speed 会把速度改成滑块值)
            # 2) 避免与 auto-pilot / 语音控制的电机指令冲突
            if self._mode != "manual":
                return jsonify({"status": "ignored", "mode": self._mode})
            data = request.get_json(silent=True) or {}
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
            # 舵机未初始化时返回错误，避免 AttributeError 崩溃
            if not getattr(self._servo, "_initialized", False):
                return jsonify({"status": "error", "msg": "舵机未初始化"}), 503
            data = request.get_json(silent=True) or {}
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
            data = request.get_json(silent=True) or {}
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
            })

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    def _get_current_frame(self):
        if self._camera_csi:
            return self._camera_csi.capture()
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
  /* 自适应尺寸单位 — 基于视口短边 vmin (D-Pad进一步放大) */
  --dpad-btn: min(22vmin, 130px);
  --gimbal-btn: min(16vmin, 88px);
  --rotate-w: min(22vmin, 120px);
  --rotate-h: min(11vmin, 64px);
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

/* ===== 主布局: 横屏三栏 (无顶栏，全高利用) ===== */
#main-app {
  display:flex;
  flex-direction:column;
  width:100%; height:100%;
  gap:4px; padding:4px;
}

/* ===== 主体三栏 ===== */
.main-body {
  flex:1 1 auto; display:flex; gap:4px; min-height:0;
}

/* ===== 左栏: 方向控制 (左手) ===== */
.left-panel {
  flex:3 1 0; min-width:0;
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
  flex:2 1 0; min-width:0;
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
  flex:3 1 0; min-width:0;
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
  height:18px; border-radius:6px; outline:none;
  background:linear-gradient(to right, var(--accent2), var(--accent));
  touch-action:pan-x; /* 允许水平拖动 */
  pointer-events:auto;
}
.speed-slider::-webkit-slider-thumb {
  -webkit-appearance:none; width:28px; height:28px;
  border-radius:50%; background:#fff; cursor:grab;
  box-shadow:0 2px 8px rgba(0,0,0,0.5); border:3px solid var(--accent);
}
.speed-slider:active::-webkit-slider-thumb { cursor:grabbing; transform:scale(1.15); }
.speed-slider::-moz-range-thumb {
  width:28px; height:28px; border-radius:50%; background:#fff;
  cursor:grab; border:3px solid var(--accent);
}
.speed-value {
  min-width:42px; text-align:center; font-size:14px; font-weight:bold;
  color:var(--accent); flex:0 0 auto;
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
    --dpad-btn: min(18vmin, 96px);
    --gimbal-btn: min(13vmin, 68px);
  }
  .dpad-label, .section-label, .speed-label { display:none; }
  .divider { margin:1px 0; }
  .right-panel { padding:4px; gap:3px; }
  .left-panel { padding:4px; }
}

/* 测距徽章颜色 */
.dist-badge.warn { color: #ff9800 !important; }
.dist-badge.danger { color: #f44336 !important; animation: pulse 0.5s infinite alternate; }
@keyframes pulse { from { opacity: 0.6; } to { opacity: 1; } }
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

  <!-- 主体三栏 (无顶栏，全高) -->
  <div class="main-body">

  <!-- 左栏: 方向控制 (左手) -->
  <div class="left-panel">
    <div class="dpad-label">麦克纳姆轮 · 8方向</div>
    <div class="dpad">
      <button data-act="fl" ontouchstart="event.preventDefault();toggleDir(-70,70,0,this)" onclick="toggleDir(-70,70,0,this)">↖</button>
      <button data-act="f" ontouchstart="event.preventDefault();toggleDir(0,100,0,this)" onclick="toggleDir(0,100,0,this)">↑</button>
      <button data-act="fr" ontouchstart="event.preventDefault();toggleDir(70,70,0,this)" onclick="toggleDir(70,70,0,this)">↗</button>
      <button data-act="sl" ontouchstart="event.preventDefault();toggleDir(-100,0,0,this)" onclick="toggleDir(-100,0,0,this)">←</button>
      <button class="stop-btn" ontouchstart="event.preventDefault();stopCar(this)" onclick="stopCar(this)">⏹</button>
      <button data-act="sr" ontouchstart="event.preventDefault();toggleDir(100,0,0,this)" onclick="toggleDir(100,0,0,this)">→</button>
      <button data-act="bl" ontouchstart="event.preventDefault();toggleDir(-70,-70,0,this)" onclick="toggleDir(-70,-70,0,this)">↙</button>
      <button data-act="b" ontouchstart="event.preventDefault();toggleDir(0,-100,0,this)" onclick="toggleDir(0,-100,0,this)">↓</button>
      <button data-act="br" ontouchstart="event.preventDefault();toggleDir(70,-70,0,this)" onclick="toggleDir(70,-70,0,this)">↘</button>
    </div>
  </div>

  <!-- 中栏: 摄像头 -->
  <div class="center-panel">
    <img id="cameraFeed" src="/video_feed" alt="摄像头画面">
    <div class="cam-overlay-top">
      <span id="aiStatus" style="font-size:11px;color:var(--green);background:rgba(0,0,0,0.5);padding:2px 8px;border-radius:10px;">● LIVE</span>
      <span></span>
    </div>
  </div>

  <!-- 右栏: 功能控制 (右手) -->
  <div class="right-panel">

    <!-- 旋转 -->
    <div class="section-label">原地旋转</div>
    <div class="rotate-row">
      <button class="rotate-btn" ontouchstart="event.preventDefault();toggleRotate(0,0,-70,this)" onclick="toggleRotate(0,0,-70,this)">
        <span class="icon">⟲</span>左转
      </button>
      <button class="rotate-btn" ontouchstart="event.preventDefault();toggleRotate(0,0,70,this)" onclick="toggleRotate(0,0,70,this)">
        <span class="icon">⟳</span>右转
      </button>
    </div>

    <div class="divider"></div>

    <!-- 速度 -->
    <div class="speed-section">
      <div class="speed-label">速度调节</div>
      <div class="speed-row">
        <span class="speed-value" id="speedDisplay">50%</span>
        <span style="font-size:11px;color:#555;">慢</span>
        <input type="range" class="speed-slider" id="speedSlider" min="20" max="100" value="50"
               oninput="updateSpeed(this.value)" onchange="commitSpeed()">
        <span style="font-size:11px;color:#555;">快</span>
      </div>
    </div>

    <div class="divider"></div>

    <!-- 测距 + 云台 (水平排列: 测距在左, 云台在右) -->
    <div class="section-label">云台控制</div>
    <div style="display:flex;align-items:center;justify-content:center;gap:8px;">
      <span class="dist-badge" id="distDisplay" style="background:var(--accent2);padding:6px 10px;border-radius:12px;font-size:13px;color:var(--green);white-space:nowrap;writing-mode:vertical-rl;text-orientation:upright;">📡 --</span>
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
    </div><!-- /测距+云台 flex -->

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
let activeDirBtn = null;
let activeRotateBtn = null;
let speedSendTimer = null;
let currentMove = {x:0, y:0, rotation:0};  // 当前运动方向，sendSpeed 用它保持运动

// ===== 点击切换方向 (点一下持续运动，再点一下停止) =====
function toggleDir(x, y, r, btn) {
  if (navigator.vibrate) navigator.vibrate(15);

  // 点同一个按钮 = 停止
  if (activeDirBtn === btn) {
    btn.classList.remove('active');
    activeDirBtn = null;
    stopCar(null);
    return;
  }

  // 清除之前的方向按钮
  if (activeDirBtn) activeDirBtn.classList.remove('active');
  // 清除旋转按钮
  if (activeRotateBtn) { activeRotateBtn.classList.remove('active'); activeRotateBtn = null; }

  // 激活新方向
  btn.classList.add('active');
  activeDirBtn = btn;
  currentMove = {x, y, rotation:r};  // 记住当前方向

  fetch('/api/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({x, y, rotation:r, speed:speedValue})
  }).catch(()=>{});
}

// ===== 点击切换旋转 =====
function toggleRotate(x, y, r, btn) {
  if (navigator.vibrate) navigator.vibrate(15);

  // 点同一个按钮 = 停止
  if (activeRotateBtn === btn) {
    btn.classList.remove('active');
    activeRotateBtn = null;
    stopCar(null);
    return;
  }

  // 清除之前的旋转按钮
  if (activeRotateBtn) activeRotateBtn.classList.remove('active');
  // 清除方向按钮
  if (activeDirBtn) { activeDirBtn.classList.remove('active'); activeDirBtn = null; }

  // 激活新旋转
  btn.classList.add('active');
  activeRotateBtn = btn;
  currentMove = {x, y, rotation:r};  // 记住当前方向

  fetch('/api/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({x, y, rotation:r, speed:speedValue})
  }).catch(()=>{});
}

function stopCar(btn) {
  if (btn) { btn.classList.add('active'); if(navigator.vibrate) navigator.vibrate(20); setTimeout(()=>btn.classList.remove('active'),200); }
  if (activeDirBtn) { activeDirBtn.classList.remove('active'); activeDirBtn = null; }
  if (activeRotateBtn) { activeRotateBtn.classList.remove('active'); activeRotateBtn = null; }
  currentMove = {x:0, y:0, rotation:0};  // 清除方向
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
  // 用当前运动方向 + 新速度发送，不会意外停车
  fetch('/api/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({x:currentMove.x, y:currentMove.y, rotation:currentMove.rotation, speed:s})
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
function setMode(mode) {
  // 先更新 UI，再发请求（避免 await 阻塞 UI 响应）
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
  if (navigator.vibrate) navigator.vibrate(15);
  fetch('/api/mode', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mode})
  }).catch(()=>{});
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

// ===== 模式同步轮询 =====
// 语音/自动模式切换会改后端模式状态，前端需轮询保持按钮高亮一致
async function syncMode() {
  try {
    const r = await fetch('/api/mode');
    const d = await r.json();
    const mode = d.mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1));
    if (btn) btn.classList.add('active');
  } catch(e) {}
}
setInterval(syncMode, 1000);

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
// 合成所有按下的移动键为单一指令，支持多键组合 (如 W+D 斜向)
function sendCurrentKeys() {
  let x = 0, y = 0, r = 0;
  for (const key of pressedKeys) {
    if (key in keyMap) {
      const [kx, ky, kr] = keyMap[key];
      x += kx; y += ky; r += kr;
    }
  }
  // 叠加后限制在 [-100, 100]，保持方向比例
  const maxAbs = Math.max(Math.abs(x), Math.abs(y), Math.abs(r));
  if (maxAbs > 100) {
    x = Math.round(x * 100 / maxAbs);
    y = Math.round(y * 100 / maxAbs);
    r = Math.round(r * 100 / maxAbs);
  }
  if (x === 0 && y === 0 && r === 0) {
    fetch('/api/stop', {method:'POST'}).catch(()=>{});
  } else {
    fetch('/api/control', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({x, y, rotation:r, speed:speedValue})}).catch(()=>{});
  }
}
document.addEventListener('keydown', e => {
  if (e.key === ' ') { e.preventDefault(); stopCar(null); return; }
  if (pressedKeys.has(e.key)) return;
  if (e.key in keyMap) {
    e.preventDefault();
    pressedKeys.add(e.key);
    sendCurrentKeys();
  }
});
document.addEventListener('keyup', e => {
  if (e.key in keyMap) {
    e.preventDefault();
    pressedKeys.delete(e.key);
    // 松开一键后发送剩余按键的合成指令，而非直接停车
    sendCurrentKeys();
  }
});
</script>
</body>
</html>'''
