"""
web_server.py - Web 遥控服务器

Flask + 手机横屏双手控制界面
功能: 8方向全向移动、原地旋转、滑块调速、云台控制、实时画面、AI 视觉、模式切换
"""

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
                 ultrasonic=None, vision=None, vision_obs=None,
                 face_recognizer=None, follower=None,
                 on_mode_change=None):
        self._motor = motor
        self._servo = servo
        self._camera_csi = camera_csi
        self._ultrasonic = ultrasonic
        self._vision = vision
        self._vision_obs = vision_obs
        self._face_recognizer = face_recognizer
        self._follower = follower
        self._on_mode_change = on_mode_change

        self._app = Flask(__name__)

        self._mode = "manual"

        self._latest_detections = []
        self._frame_lock = threading.Lock()

        self._register_routes()

    def set_mode(self, mode):
        """外部同步模式状态 (不触发回调，避免递归)

        供 AICar.set_mode 在语音/自动切换模式时同步 WebServer._mode，
        使 /api/control 的模式检查和前端按钮状态保持正确。
        """
        if mode in ("manual", "auto", "voice", "follow"):
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
            # 审查 bug: 非数字字符串导致 float() 抛 ValueError → 500
            try:
                x = float(data.get("x", 0))
                y = float(data.get("y", 0))
                r = float(data.get("rotation", 0))
                s = float(data.get("speed", 50))
            except (TypeError, ValueError):
                return jsonify({"status": "error", "msg": "参数必须为数字"}), 400
            # 限幅，防止异常输入
            s = max(0, min(100, s))

            self._motor.set_speed(s)
            self._motor.move(x, y, r)
            return jsonify({"status": "ok"})

        @app.route("/api/stop", methods=["POST"])
        def api_stop():
            # 急停是最高优先级的安全操作，任何模式都必须响应。
            # 若当前在 auto/voice 模式下只调 motor.stop()，auto-pilot 线程
            # 会在 0.2s 内再次 move() 覆盖停车指令，车继续动。
            # 解决: 先切回 manual (会触发 AICar.set_mode 内的 motor.stop)，
            # 再 stop() 一次确保电机立即停转。
            if self._mode != "manual":
                self._mode = "manual"
                if self._on_mode_change:
                    self._on_mode_change("manual")
            self._motor.stop()
            return jsonify({"status": "ok", "mode": "manual"})

        @app.route("/api/servo", methods=["POST"])
        def api_servo():
            # 舵机未初始化时返回错误，避免 AttributeError 崩溃
            if not getattr(self._servo, "_initialized", False):
                return jsonify({"status": "error", "msg": "舵机未初始化"}), 503
            data = request.get_json(silent=True) or {}

            # 审查 bug: int("90.5") 或 int("abc") 抛 ValueError → 500
            def _to_int(v):
                try:
                    return int(float(v))
                except (TypeError, ValueError):
                    return None

            pan = _to_int(data.get("pan"))
            tilt = _to_int(data.get("tilt"))
            if pan is not None:
                self._servo.pan(pan)
            if tilt is not None:
                self._servo.tilt(tilt)
            return jsonify({"status": "ok"})

        @app.route("/api/distance")
        def api_distance():
            # auto/follow 模式下 follower/auto-pilot 线程在持续测距，
            # 此处再调 measure() 会阻塞 50-100ms 并竞争超声波锁。
            # 改为读 follower 缓存的距离 (与 /api/status 一致)
            if self._follower and self._mode in ("auto", "follow"):
                st = self._follower.get_state()
                return jsonify({"distance": st.get("distance", -1)})
            # manual/voice 模式下没有线程在测距，直接调 measure()
            if self._ultrasonic:
                d = self._ultrasonic.measure()
                return jsonify({"distance": round(d, 1) if d > 0 else -1})
            return jsonify({"distance": -1})

        @app.route("/api/mode", methods=["POST"])
        def api_mode():
            data = request.get_json(silent=True) or {}
            mode = data.get("mode", "manual")
            if mode in ("manual", "auto", "voice", "follow"):
                # follow 前置检查: 与 AICar.set_mode 一致，避免 WebServer._mode 已改但 AICar 拒绝导致状态不一致
                # 失败时返回 diagnose() 的具体原因，让用户知道该装 dlib / 下模型 / 录入主人
                if mode == "follow" and self._face_recognizer and not self._face_recognizer.is_ready():
                    diag = self._face_recognizer.diagnose()
                    return jsonify({
                        "status": "error",
                        "msg": f"人脸识别未就绪: {diag['reason']}",
                        "detail": diag.get("detail", ""),
                        "diagnosis": diag,
                    }), 400
                # 不直接设 self._mode；交给 AICar.set_mode 处理，它会回调 self.set_mode 同步
                # 这样保证 WebServer._mode 和 AICar._mode 永远一致
                if self._on_mode_change:
                    self._on_mode_change(mode)
                else:
                    self._mode = mode
                return jsonify({"status": "ok", "mode": self._mode})
            return jsonify({"status": "error", "msg": "无效模式"}), 400

        @app.route("/api/mode")
        def get_mode():
            return jsonify({"mode": self._mode})

        @app.route("/api/detect")
        def api_detect():
            # 审查 bug: 非 auto 模式下 _latest_detections 不更新，返回 stale 数据
            if not self._vision or self._mode != "auto":
                return jsonify({"detections": []})
            with self._frame_lock:
                return jsonify({"detections": self._latest_detections})

        @app.route("/api/status")
        def api_status():
            """综合状态接口，减少前端轮询次数"""
            # 不在此处调用 ultrasonic.measure() — 它会阻塞 50-100ms (5次采样)，
            # 与 follower 线程竞争超声波锁。改为读 follower/auto-pilot 最近测得的距离。
            dist = -1
            if self._follower:
                st = self._follower.get_state()
                # follower 状态里有 dist 字段时优先用
                dist = st.get("distance", -1)
            status = {
                "mode": self._mode,
                "distance": dist,
                "speed": self._motor.get_speed(),
                "face_ready": self._face_recognizer.is_ready() if self._face_recognizer else False,
            }
            if self._follower:
                status["follow_state"] = self._follower.get_state()
            return jsonify(status)

        # ========== 主人管理 API ==========

        @app.route("/api/face/diagnose")
        def face_diagnose():
            """诊断人脸识别就绪状态，返回具体未就绪原因"""
            if not self._face_recognizer:
                return jsonify({"status": "error", "msg": "人脸识别模块未加载"}), 503
            return jsonify({"status": "ok", "diagnosis": self._face_recognizer.diagnose()})

        @app.route("/api/owner/list")
        def owner_list():
            if not self._face_recognizer:
                return jsonify({"status": "error", "msg": "人脸识别模块未加载"}), 503
            return jsonify({
                "status": "ok",
                "owners": self._face_recognizer.list_owners(),
                "ready": self._face_recognizer.is_ready(),
            })

        @app.route("/api/owner/register", methods=["POST"])
        def owner_register():
            if not self._face_recognizer:
                return jsonify({"status": "error", "msg": "人脸识别模块未加载"}), 503
            data = request.get_json(silent=True) or {}
            name = (data.get("name") or "").strip()
            if not name:
                return jsonify({"status": "error", "msg": "名字不能为空"}), 400
            owner_id = self._face_recognizer.register_owner(name)
            if owner_id == "exists":
                return jsonify({"status": "exists", "msg": f"主人 '{name}' 已存在"}), 409
            if owner_id:
                return jsonify({"status": "ok", "owner_id": owner_id, "name": name})
            return jsonify({"status": "error", "msg": "注册失败"}), 500

        @app.route("/api/owner/capture", methods=["POST"])
        def owner_capture():
            """采集当前帧的人脸 embedding 到指定主人"""
            if not self._face_recognizer:
                return jsonify({"status": "error", "msg": "人脸识别模块未加载"}), 503
            # 采集只检查模型就绪 (_initialized)，不要求已有 owner/embedding
            # (is_ready 要求已有 embedding，新注册的主人 embeds=[] 过不了)
            if not self._face_recognizer._initialized:
                return jsonify({"status": "error", "msg": "人脸识别未初始化 (检查 dlib 安装/模型文件)"}), 503
            data = request.get_json(silent=True) or {}
            owner_id = data.get("owner_id")
            if not owner_id:
                return jsonify({"status": "error", "msg": "缺少 owner_id"}), 400
            frame = self._get_current_frame()
            if frame is None:
                return jsonify({"status": "error", "msg": "摄像头未就绪"}), 503
            result = self._face_recognizer.capture_and_save_embedding(owner_id, frame)
            if result["ok"]:
                return jsonify({"status": "ok", "msg": result["msg"]})
            # 审查 bug: 之前统一返回 400，应区分 503/404/400/500
            msg = result["msg"]
            if "未初始化" in msg:
                code = 503
            elif "未找到主人" in msg:
                code = 404
            elif "保存失败" in msg:
                code = 500
            else:
                code = 400
            return jsonify({"status": "error", "msg": msg}), code

        @app.route("/api/owner/delete", methods=["POST"])
        def owner_delete():
            if not self._face_recognizer:
                return jsonify({"status": "error", "msg": "人脸识别模块未加载"}), 503
            data = request.get_json(silent=True) or {}
            owner_id = data.get("owner_id")
            if not owner_id:
                return jsonify({"status": "error", "msg": "缺少 owner_id"}), 400
            # 如果正在 follow 被删除的主人，先切回 manual 避免车误报"主人走丢了"
            if self._follower:
                st = self._follower.get_state()
                if st.get("following") and st.get("target_name"):
                    # 找到要删除的主人名字
                    owners = self._face_recognizer.list_owners()
                    del_name = next((o["name"] for o in owners if o["id"] == owner_id), None)
                    if del_name and st.get("target_name") == del_name:
                        if self._on_mode_change:
                            self._on_mode_change("manual")
                        else:
                            self._mode = "manual"
            ok = self._face_recognizer.delete_owner(owner_id)
            # 审查 bug: 之前删除失败仍返回 HTTP 200，前端无反馈
            if ok:
                return jsonify({"status": "ok"})
            return jsonify({"status": "error", "msg": "删除失败 (主人不存在或文件系统错误)"}), 404

        @app.route("/api/follow_state")
        def follow_state():
            if not self._follower:
                return jsonify({"status": "error", "msg": "跟随模块未加载"}), 503
            return jsonify({"status": "ok", "state": self._follower.get_state()})

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    def _get_current_frame(self):
        # camera.capture() 内部已有锁，这里直接调用即可
        if self._camera_csi:
            return self._camera_csi.capture()
        return None

    def _generate_frames(self):
        import cv2
        while True:
            try:
                frame = self._get_current_frame()
                if frame is not None:
                    # auto 模式: 叠加视觉避障分析 + 物体检测
                    # 注意: 控制线程已在 _auto_pilot_loop 里调过 analyze/detect，
                    # 这里再调一次是为了 web 端可视化；如性能不足可后续改为读共享缓存
                    if self._mode == "auto":
                        if self._vision_obs:
                            try:
                                analysis = self._vision_obs.analyze(frame)
                                if analysis.get("ok"):
                                    frame = self._vision_obs.draw_overlay(frame, analysis)
                            except Exception:
                                pass
                        if self._vision:
                            try:
                                detections = self._vision.detect(frame)
                                with self._frame_lock:
                                    self._latest_detections = detections
                                if detections:
                                    frame = self._vision.draw_detections(frame, detections)
                            except Exception:
                                pass

                    # follow 模式: 复用 follower 线程的检测结果 (避免重复跑 dlib HOG)
                    # follower.get_state() 返回 last_faces / last_ids (只读)
                    elif self._mode == "follow":
                        if self._face_recognizer and self._face_recognizer.is_ready():
                            st = self._follower.get_state() if self._follower else {}
                            faces = st.get("last_faces") or []
                            ids = st.get("last_ids") or []
                            if faces:
                                try:
                                    frame = self._face_recognizer.draw_detections(frame, faces, ids)
                                except Exception:
                                    pass
                        else:
                            # 未就绪时在画面上提示 (RGB 红色)
                            cv2.putText(frame, "Face recognizer not ready",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 0, 0), 2)

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
            except Exception as e:
                # 单帧失败不能让整个视频流断 (浏览器会重连造成雪崩)
                print(f"[WebServer] _generate_frames 异常: {e}")
                try:
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    _, jpeg = cv2.imencode('.jpg', blank)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           jpeg.tobytes() + b'\r\n')
                except Exception:
                    pass

            time.sleep(0.05)

    def start(self):
        # 审查 bug: 之前 threaded 参数从未使用，签名误导。删除它。
        # Flask app.run 默认 threaded=True 已足够处理并发请求。
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
  /* 自适应尺寸单位 — 基于视口短边 vmin (放大方向/云台按钮) */
  --dpad-btn: min(26vmin, 150px);
  --gimbal-btn: min(20vmin, 110px);
  --rotate-w: min(22vmin, 120px);
  --rotate-h: min(12vmin, 70px);
}
html, body {
  width:100%; height:100%; overflow:hidden;
  font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
  background:var(--bg); color:var(--text);
  touch-action:manipulation; user-select:none; -webkit-user-select:none;
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
  display:flex; gap:6px; justify-content:center; align-items:center;
}
/* 测距徽章 (放在旋转行左侧, 横排显示) */
.rotate-row .dist-badge {
  margin-right:auto;  /* 靠左对齐 */
  background:var(--accent2); padding:6px 10px; border-radius:10px;
  font-size:13px; color:var(--green); white-space:nowrap;
  min-width:72px; text-align:center; flex:0 0 auto;
}
/* 注册/管理按钮列 (放在云台左侧) */
.owner-action-col {
  display:flex; flex-direction:column; gap:6px; flex:0 0 auto;
}
.owner-action-btn {
  width:min(16vmin, 80px); height:min(8vmin, 44px);
  border:none; border-radius:8px; background:var(--accent2);
  color:var(--text); font-size:13px; cursor:pointer;
  touch-action:manipulation; transition:all 0.08s;
  border:1px solid rgba(255,255,255,0.08);
}
.owner-action-btn:active { background:var(--btn-active); transform:scale(0.92); }
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
  /* 滑块周围禁用浏览器默认触摸平移，手指溢出滑块时不会拖动页面 */
  touch-action:none;
}
.speed-row {
  display:flex; align-items:center; gap:6px; width:100%;
  touch-action:none;
}
.speed-slider {
  flex:1; -webkit-appearance:none; appearance:none;
  height:24px; border-radius:8px; outline:none;
  background:linear-gradient(to right, var(--accent2), var(--accent));
  /* touch-action:none 让滑块完全接管触摸，禁止浏览器默认平移/滚动，
     这样手指拖出滑块边缘时不会触发"拖动整个页面"。
     之前用 pan-x 仍允许水平平移，手指右滑出滑块后父元素会接管导致页面被拖动。 */
  touch-action:none;
  pointer-events:auto;
}
.speed-slider::-webkit-slider-thumb {
  -webkit-appearance:none; width:34px; height:34px;
  border-radius:50%; background:#fff; cursor:grab;
  box-shadow:0 2px 8px rgba(0,0,0,0.5); border:3px solid var(--accent);
  /* 扩大触摸热区，让手指更容易抓住 thumb，减少滑出概率 */
  padding:6px; margin-top:-5px;
}
.speed-slider:active::-webkit-slider-thumb { cursor:grabbing; transform:scale(1.15); }
.speed-slider::-moz-range-thumb {
  width:34px; height:34px; border-radius:50%; background:#fff;
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

/* 跟随状态行 (目标/状态, 常驻显示在管理按钮下方) */
.follow-info-row {
  width:100%; padding:3px 10px; font-size:12px; line-height:1.5;
  background:rgba(15,52,96,0.3); border-radius:6px;
}
.follow-info-row span { color:#00e676; font-weight:bold; }

/* ===== 弹窗 (注册/管理主人) ===== */
.modal-overlay {
  display:none; position:fixed; inset:0; z-index:9998;
  background:rgba(0,0,0,0.7); backdrop-filter:blur(4px);
  align-items:center; justify-content:center;
}
.modal-overlay.show { display:flex; }
.modal-box {
  background:var(--panel); border-radius:16px; padding:20px;
  width:min(90vw, 380px); max-height:80vh; overflow-y:auto;
  border:1px solid var(--accent2); box-shadow:0 8px 32px rgba(0,0,0,0.5);
}
.modal-title {
  font-size:16px; font-weight:bold; margin-bottom:14px;
  text-align:center; color:var(--accent);
}
.modal-input {
  width:100%; padding:10px 12px; border:1px solid var(--accent2);
  border-radius:8px; background:var(--bg); color:var(--text);
  font-size:14px; margin-bottom:12px;
}
.modal-btn-row { display:flex; gap:8px; }
.modal-btn {
  flex:1; padding:10px; border:none; border-radius:8px;
  font-size:14px; cursor:pointer; transition:all 0.15s;
}
.modal-btn-primary { background:var(--accent); color:#fff; }
.modal-btn-primary:active { transform:scale(0.95); }
.modal-btn-secondary { background:var(--btn); color:var(--text); border:1px solid var(--accent2); }
.modal-btn-secondary:active { transform:scale(0.95); }
.modal-owner-list {
  display:flex; flex-direction:column; gap:8px; margin-bottom:12px;
}
.modal-owner-item {
  display:flex; align-items:center; gap:8px;
  padding:10px; background:rgba(255,255,255,0.04); border-radius:8px;
}
.modal-owner-name { flex:1; font-size:14px; }
.modal-owner-name small { color:#888; font-size:11px; }
.modal-owner-capture {
  padding:6px 12px; border:none; border-radius:6px;
  background:var(--accent2); color:#fff; font-size:12px; cursor:pointer;
}
.modal-owner-del {
  padding:6px 12px; border:none; border-radius:6px;
  background:#c62828; color:#fff; font-size:12px; cursor:pointer;
}
.modal-empty { color:#666; font-size:13px; text-align:center; padding:20px 0; }

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
      <button data-act="fl" ontouchend="event.preventDefault();toggleDir(-70,70,0,this)" onclick="toggleDir(-70,70,0,this)">↖</button>
      <button data-act="f" ontouchend="event.preventDefault();toggleDir(0,100,0,this)" onclick="toggleDir(0,100,0,this)">↑</button>
      <button data-act="fr" ontouchend="event.preventDefault();toggleDir(70,70,0,this)" onclick="toggleDir(70,70,0,this)">↗</button>
      <button data-act="sl" ontouchend="event.preventDefault();toggleDir(-100,0,0,this)" onclick="toggleDir(-100,0,0,this)">←</button>
      <button class="stop-btn" ontouchend="event.preventDefault();stopCar(this)" onclick="stopCar(this)">⏹</button>
      <button data-act="sr" ontouchend="event.preventDefault();toggleDir(100,0,0,this)" onclick="toggleDir(100,0,0,this)">→</button>
      <button data-act="bl" ontouchend="event.preventDefault();toggleDir(-70,-70,0,this)" onclick="toggleDir(-70,-70,0,this)">↙</button>
      <button data-act="b" ontouchend="event.preventDefault();toggleDir(0,-100,0,this)" onclick="toggleDir(0,-100,0,this)">↓</button>
      <button data-act="br" ontouchend="event.preventDefault();toggleDir(70,-70,0,this)" onclick="toggleDir(70,-70,0,this)">↘</button>
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

    <!-- 旋转 (测距徽章放在左转按钮左边, 靠左对齐, 横排显示) -->
    <div class="section-label">原地旋转</div>
    <div class="rotate-row">
      <span class="dist-badge" id="distDisplay">📡 --</span>
      <button class="rotate-btn" ontouchend="event.preventDefault();toggleRotate(0,0,-100,this)" onclick="toggleRotate(0,0,-100,this)">
        <span class="icon">⟲</span>左转
      </button>
      <button class="rotate-btn" ontouchend="event.preventDefault();toggleRotate(0,0,100,this)" onclick="toggleRotate(0,0,100,this)">
        <span class="icon">⟳</span>右转
      </button>
    </div>

    <div class="divider"></div>

    <!-- 速度 -->
    <div class="speed-section">
      <div class="speed-label">速度调节</div>
      <div class="speed-row">
        <span class="speed-value" id="speedDisplay">30%</span>
        <span style="font-size:11px;color:#555;">慢</span>
        <input type="range" class="speed-slider" id="speedSlider" min="20" max="100" value="30"
               oninput="updateSpeed(this.value)" onchange="commitSpeed()">
        <span style="font-size:11px;color:#555;">快</span>
      </div>
    </div>

    <div class="divider"></div>

    <!-- 注册/管理按钮 + 云台 (水平排列: 按钮列在左, 云台在右) -->
    <div class="section-label">云台控制</div>
    <div style="display:flex;align-items:center;justify-content:center;gap:8px;">
      <div class="owner-action-col">
        <button class="owner-action-btn" ontouchend="event.preventDefault();openRegisterModal()" onclick="openRegisterModal()">注册</button>
        <button class="owner-action-btn" ontouchend="event.preventDefault();openManageModal()" onclick="openManageModal()">管理</button>
      </div>
      <div class="gimbal-grid">
      <div></div>
      <button ontouchstart="gimbalTilt(-10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalTilt(-10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">↑</button>
      <div></div>
      <button ontouchstart="gimbalPan(-10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalPan(-10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">←</button>
      <button class="gimbal-center" ontouchend="event.preventDefault();gimbalCenter()" onclick="gimbalCenter()">归中</button>
      <button ontouchstart="gimbalPan(10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalPan(10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">→</button>
      <div></div>
      <button ontouchstart="gimbalTilt(10,this)" ontouchend="gimbalRelease(this)" onmousedown="gimbalTilt(10,this)" onmouseup="gimbalRelease(this)" onmouseleave="gimbalRelease(this)">↓</button>
      <div></div>
    </div>
    </div><!-- /注册管理按钮+云台 flex -->
    <!-- 目标 (云台下方, 常驻显示) -->
    <div class="follow-info-row">目标: <span id="followTarget">-</span></div>
    <!-- 状态 (目标下方, 常驻显示) -->
    <div class="follow-info-row">状态: <span id="followMsg">待机</span></div>

    <div class="divider"></div>

    <!-- 模式 -->
    <div class="section-label">运行模式</div>
    <div class="mode-row">
      <button class="mode-btn active" id="modeManual" ontouchend="event.preventDefault();setMode('manual')" onclick="setMode('manual')">🕹手动</button>
      <button class="mode-btn" id="modeAuto" ontouchend="event.preventDefault();setMode('auto')" onclick="setMode('auto')">🤖自动</button>
      <button class="mode-btn" id="modeVoice" ontouchend="event.preventDefault();setMode('voice')" onclick="setMode('voice')">🎤语音</button>
      <button class="mode-btn" id="modeFollow" ontouchend="event.preventDefault();setMode('follow')" onclick="setMode('follow')">👣跟随</button>
    </div>

    <!-- 人脸识别状态 (跟随模式依赖) -->
    <div class="section-label">人脸识别 <span id="faceStatus" style="font-size:0.7em;color:#999;"></span></div>

  </div>
  </div><!-- /main-body -->
</div>

<!-- ===== 注册主人弹窗 ===== -->
<div class="modal-overlay" id="registerModal">
  <div class="modal-box">
    <div class="modal-title">注册新主人</div>
    <input type="text" id="modalOwnerName" class="modal-input" placeholder="请输入主人名字">
    <div class="modal-btn-row">
      <button class="modal-btn modal-btn-secondary" ontouchend="event.preventDefault();closeModal('registerModal')" onclick="closeModal('registerModal')">取消</button>
      <button class="modal-btn modal-btn-primary" ontouchend="event.preventDefault();registerOwner()" onclick="registerOwner()">注册</button>
    </div>
  </div>
</div>

<!-- ===== 管理主人弹窗 ===== -->
<div class="modal-overlay" id="manageModal">
  <div class="modal-box">
    <div class="modal-title">主人管理</div>
    <div id="modalOwnerList" class="modal-owner-list"></div>
    <div class="modal-btn-row">
      <button class="modal-btn modal-btn-secondary" ontouchend="event.preventDefault();closeModal('manageModal')" onclick="closeModal('manageModal')">关闭</button>
    </div>
  </div>
</div>

<script>
let speedValue = 30;
let currentPan = 90, currentTilt = 90;
let activeDirBtn = null;
let activeRotateBtn = null;
let speedSendTimer = null;
let currentMove = {x:0, y:0, rotation:0};  // 当前运动方向，sendSpeed 用它保持运动
let currentMode = 'manual';  // 审查 bug: 追踪当前模式，非 manual 时不发键盘/方向控制请求

// 防触摸+鼠标双触发 (ontouchend+onclick / ontouchstart+onmousedown):
// 部分移动浏览器 touch 事件后仍会合成 click/mouse 事件，导致：
//   - toggle 类按钮被切换两次回到原状 = "点了没反应"
//   - 云台按钮增量执行两次，偏移加倍
//   - 模式/注册等非 toggle 按钮虽功能幂等，但发两次请求浪费
// 用 350ms 内同 key 去重兜底 (人手快速双击间隔通常 > 350ms)。
// key 可以是 DOM 元素 (按钮) 或字符串 (如 mode 名)。
let _lastInputKey = null;
let _lastInputTime = 0;
function _guardDoubleFire(key) {
  if (!key) return false;
  const now = Date.now();
  if (key === _lastInputKey && now - _lastInputTime < 350) return true;
  _lastInputKey = key;
  _lastInputTime = now;
  return false;
}

// 离开 manual 模式时清理运动状态 (审查 bug: 之前不清理，切回 manual 调滑块会意外移动)
function clearMoveState() {
  if (activeDirBtn) { activeDirBtn.classList.remove('active'); activeDirBtn = null; }
  if (activeRotateBtn) { activeRotateBtn.classList.remove('active'); activeRotateBtn = null; }
  currentMove = {x:0, y:0, rotation:0};
}

// ===== 点击切换方向 (点一下持续运动，再点一下停止) =====
function toggleDir(x, y, r, btn) {
  if (_guardDoubleFire(btn)) return;
  if (navigator.vibrate) navigator.vibrate(15);

  // 点同一个按钮 = 停止
  if (activeDirBtn === btn) {
    btn.classList.remove('active');
    activeDirBtn = null;
    stopCar(null);
    return;
  }

  // 审查 bug: 非 manual 模式下方向控制会被服务端 ignore，但 stopCar 会触发 /api/stop 急停切回 manual
  // 所以非 manual 时只更新 UI 不发请求
  if (currentMode !== 'manual') return;

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
  if (_guardDoubleFire(btn)) return;
  if (navigator.vibrate) navigator.vibrate(15);

  // 点同一个按钮 = 停止
  if (activeRotateBtn === btn) {
    btn.classList.remove('active');
    activeRotateBtn = null;
    stopCar(null);
    return;
  }

  // 审查 bug: 非 manual 模式下不发请求 (同 toggleDir)
  if (currentMode !== 'manual') return;

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
  if (_guardDoubleFire(btn)) return;
  if (btn) { btn.classList.add('active'); if(navigator.vibrate) navigator.vibrate(20); setTimeout(()=>btn.classList.remove('active'),200); }
  clearMoveState();
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
  // 审查 bug: 非 manual 模式下 /api/control 会被 ignore，不发请求避免无效流量
  if (currentMode !== 'manual') return;
  // 用当前运动方向 + 新速度发送，不会意外停车
  fetch('/api/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({x:currentMove.x, y:currentMove.y, rotation:currentMove.rotation, speed:s})
  }).catch(()=>{});
}

// ===== 云台 =====
async function gimbalPan(delta, btn) {
  if (_guardDoubleFire(btn)) return;
  btn.classList.add('active');
  if (navigator.vibrate) navigator.vibrate(10);
  currentPan = Math.max(0, Math.min(180, currentPan + delta));
  await fetch('/api/servo', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({pan: currentPan})
  }).catch(()=>{});
}

async function gimbalTilt(delta, btn) {
  if (_guardDoubleFire(btn)) return;
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
  if (_guardDoubleFire('gimbal:center')) return;
  currentPan = 90; currentTilt = 90;
  await fetch('/api/servo', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({pan:90, tilt:90})
  }).catch(()=>{});
}

// ===== 模式 =====
async function setMode(mode) {
  if (_guardDoubleFire('mode:' + mode)) return;
  // 先更新 UI，再发请求（避免 await 阻塞 UI 响应）
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
  if (navigator.vibrate) navigator.vibrate(15);
  // 审查 bug: 离开 manual 时清理运动状态，避免切回后调滑块意外移动
  if (mode !== 'manual') clearMoveState();
  try {
    const r = await fetch('/api/mode', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({mode})
    });
    const d = await r.json();
    // 审查 bug: 服务端拒绝时 (如 follow 未就绪) 回滚 UI 并提示
    if (d.status !== 'ok') {
      // follow 未就绪时展示具体诊断原因 + 修复建议
      if (d.detail) {
        alert(d.msg + '\\n\\n' + d.detail);
      } else {
        alert(d.msg || '模式切换失败');
      }
      syncMode();  // 立即拉回正确状态
    } else {
      currentMode = mode;
    }
  } catch(e) { alert('网络错误'); }
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
    // 审查 bug: 检测到模式变化时，若离开 manual 则清理运动状态
    if (mode !== currentMode) {
      if (mode !== 'manual') clearMoveState();
      currentMode = mode;
    }
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1));
    if (btn) btn.classList.add('active');
    // 目标/状态行已常驻显示 (移到管理按钮下方)，无需按模式切换显隐
  } catch(e) {}
}
setInterval(syncMode, 1000);

// ===== 主人管理 (弹窗) =====
function openRegisterModal() {
  document.getElementById('modalOwnerName').value = '';
  document.getElementById('registerModal').classList.add('show');
  setTimeout(() => document.getElementById('modalOwnerName').focus(), 100);
}
function openManageModal() {
  document.getElementById('manageModal').classList.add('show');
  refreshOwners();
}
function closeModal(id) {
  document.getElementById(id).classList.remove('show');
}
// 点击弹窗背景关闭
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.modal-overlay').forEach(ov => {
    ov.addEventListener('click', function(e) {
      if (e.target === this) this.classList.remove('show');
    });
  });
});

async function refreshOwners() {
  try {
    const r = await fetch('/api/owner/list');
    const d = await r.json();
    const statusEl = document.getElementById('faceStatus');
    if (statusEl) {
      statusEl.textContent = d.ready ? `(已就绪, ${d.owners.length}人)` : '(未就绪)';
      statusEl.style.color = d.ready ? '#00e676' : '#999';
    }
    const listEl = document.getElementById('modalOwnerList');
    if (!listEl) return;
    // 用 DOM API 构造元素，避免 innerHTML 字符串拼接的 XSS 风险
    listEl.innerHTML = '';
    if (!d.owners || d.owners.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'modal-empty';
      empty.textContent = '尚未录入主人';
      listEl.appendChild(empty);
      return;
    }
    d.owners.forEach(o => {
      const div = document.createElement('div');
      div.className = 'modal-owner-item';
      const span = document.createElement('span');
      span.className = 'modal-owner-name';
      span.textContent = o.name + ' ';
      const small = document.createElement('small');
      small.textContent = `(${o.samples}样本)`;
      span.appendChild(small);
      const capBtn = document.createElement('button');
      capBtn.className = 'modal-owner-capture';
      capBtn.textContent = '采集';
      capBtn.addEventListener('click', () => captureOwner(o.id, o.name));
      const delBtn = document.createElement('button');
      delBtn.className = 'modal-owner-del';
      delBtn.textContent = '删除';
      delBtn.addEventListener('click', () => deleteOwner(o.id));
      div.append(span, capBtn, delBtn);
      listEl.appendChild(div);
    });
  } catch(e) {}
}

async function registerOwner() {
  const input = document.getElementById('modalOwnerName');
  const name = (input.value || '').trim();
  if (!name) { alert('请输入名字'); return; }
  try {
    const r = await fetch('/api/owner/register', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name})
    });
    const d = await r.json();
    if (d.status === 'ok') {
      input.value = '';
      closeModal('registerModal');
      await refreshOwners();
      alert(`已注册 "${name}"，请点"管理"打开列表，站到摄像头前点"采集"3次`);
    } else {
      alert('注册失败: ' + (d.msg || ''));
    }
  } catch(e) { alert('网络错误'); }
}

async function captureOwner(ownerId, ownerName) {
  try {
    const r = await fetch('/api/owner/capture', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({owner_id: ownerId})
    });
    const d = await r.json();
    if (d.status === 'ok') {
      await refreshOwners();
    } else {
      alert(d.msg || '采集失败');
    }
  } catch(e) { alert('网络错误'); }
}

async function deleteOwner(ownerId) {
  if (!confirm('确定删除该主人?')) return;
  try {
    const r = await fetch('/api/owner/delete', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({owner_id: ownerId})
    });
    const d = await r.json();
    if (d.status !== 'ok') {
      alert(d.msg || '删除失败');
    }
    await refreshOwners();
  } catch(e) { alert('网络错误'); }
}

// 跟随状态轮询
async function updateFollowState() {
  try {
    const r = await fetch('/api/follow_state');
    const d = await r.json();
    if (d.status !== 'ok') return;
    const s = d.state;
    const msgEl = document.getElementById('followMsg');
    const tgtEl = document.getElementById('followTarget');
    if (msgEl) msgEl.textContent = s.msg || '待机';
    if (tgtEl) tgtEl.textContent = s.target_name || '-';
  } catch(e) {}
}

// 初始化 + 定期刷新
refreshOwners();
setInterval(refreshOwners, 5000);    // 主人列表 5s 刷新一次
setInterval(updateFollowState, 500); // 跟随状态 500ms 刷新

// ===== 防止缩放/双击 =====
// 审查 bug: 之前在 document 上全局 preventDefault touchmove，
// 在 iOS Safari / 微信内置等移动浏览器上会打断触摸事件链，
// 导致所有 onclick 按钮点击无反应、滑块 oninput 不触发。
// 防滚动改用 CSS: html,body 设 touch-action:manipulation + overflow:hidden；
// 滑块靠 .speed-slider 的 touch-action:pan-x 正常拖动。
// 所有点击类按钮统一加 ontouchend (触摸) + onclick (鼠标) 双绑定，
// 配合 _guardDoubleFire 去重，不依赖浏览器的 click 合成。
// gesturestart/dblclick 仍需 JS 阻止 (CSS 无法禁用 iOS 双指/双击缩放)。
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
  // 审查 bug: 非 manual 模式下松开按键会触发 /api/stop 急停切回 manual
  // 非 manual 时键盘控制完全静默 (空格急停除外，在 keydown 里单独处理)
  if (currentMode !== 'manual') return;
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
