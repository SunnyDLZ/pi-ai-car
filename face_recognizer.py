"""
face_recognizer.py - 主人识别模块

基于 dlib + face_recognition 的人脸识别 (128D embedding + 余弦相似度)。
识别流程:
  1. dlib HOG 人脸检测 (CPU 友好, ~50ms/帧)
  2. 68 点关键点定位
  3. face_recognition 128D 编码
  4. 与主人库 embedding 比对 (余弦相似度)

主人库结构 (data/owners/, 已加入 .gitignore 绝不上传):
  data/owners/
    ├── registry.json                 # 主人 ID → 名字映射
    ├── owner_001_<name>/
    │   ├── meta.json                 # {"name": "张三", "created": "2026-07-20T..."}
    │   ├── embedding_001.npy         # 128D 向量
    │   ├── embedding_002.npy
    │   └── embedding_003.npy         # OWNER_SAMPLES_PER_PERSON 个样本
    └── owner_002_<name>/

降级策略:
  - dlib/face_recognition 装不上 → 初始化失败，模块不可用
  - 模型文件缺失 → 同上
  - 主人库为空 → 可识别但所有识别结果都是 "未识别到主人"
"""

from __future__ import annotations

import os
import json
import threading
import numpy as np
from datetime import datetime
from config import (
    OWNERS_DIR,
    FACE_LANDMARK_MODEL,
    FACE_RECOGNITION_MODEL,
    FACE_MATCH_THRESHOLD,
    OWNER_SAMPLES_PER_PERSON,
)

# 项目根目录 (用于把 config 里的相对路径转为绝对路径，避免 systemd 启动时 cwd 不对找不到模型)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RESOLVED_OWNERS_DIR = OWNERS_DIR if os.path.isabs(OWNERS_DIR) else os.path.join(_BASE_DIR, OWNERS_DIR)
_RESOLVED_LANDMARK = FACE_LANDMARK_MODEL if os.path.isabs(FACE_LANDMARK_MODEL) else os.path.join(_BASE_DIR, FACE_LANDMARK_MODEL)
_RESOLVED_RECOGNITION = FACE_RECOGNITION_MODEL if os.path.isabs(FACE_RECOGNITION_MODEL) else os.path.join(_BASE_DIR, FACE_RECOGNITION_MODEL)


class FaceRecognizer:
    """主人识别器"""

    def __init__(self):
        self._initialized = False
        self._detector = None       # dlib HOG 人脸检测器
        self._predictor = None      # dlib 68 点关键点
        self._encoder = None        # dlib ResNet 128D 编码器
        self._owners = []           # [{id, name, embeddings: [np.ndarray, ...]}, ...]
        self._lock = threading.Lock()

    def init(self):
        """加载 dlib 模型和主人库

        任一外部依赖缺失会优雅降级为 _initialized=False，不影响其他模块。
        """
        try:
            import dlib
        except ImportError:
            print("[FaceRecognizer] 未安装 dlib，主人识别功能不可用")
            print("[FaceRecognizer] 安装: pip install dlib face_recognition")
            return False

        # 检查模型文件 (用绝对路径，避免 systemd 启动时 cwd 不对找不到)
        if not os.path.exists(_RESOLVED_LANDMARK):
            print(f"[FaceRecognizer] 缺失模型: {_RESOLVED_LANDMARK}")
            print(f"[FaceRecognizer] 下载: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            return False
        if not os.path.exists(_RESOLVED_RECOGNITION):
            print(f"[FaceRecognizer] 缺失模型: {_RESOLVED_RECOGNITION}")
            print(f"[FaceRecognizer] 下载: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
            return False

        try:
            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(_RESOLVED_LANDMARK)
            self._encoder = dlib.face_recognition_model_v1(_RESOLVED_RECOGNITION)
        except Exception as e:
            # 半初始化回滚，避免遗留状态引起调试困惑
            self._detector = None
            self._predictor = None
            self._encoder = None
            print(f"[FaceRecognizer] 模型加载失败: {e}")
            return False

        # 加载主人库
        os.makedirs(_RESOLVED_OWNERS_DIR, exist_ok=True)
        self._owners_dir = _RESOLVED_OWNERS_DIR
        self._load_registry()

        self._initialized = True
        print(f"[FaceRecognizer] 初始化完成，已加载 {len(self._owners)} 个主人")
        return True

    # ===================== 主人库管理 =====================

    def _load_registry(self):
        """从磁盘加载主人库"""
        self._owners = []
        if not os.path.isdir(_RESOLVED_OWNERS_DIR):
            return

        for owner_dir_name in sorted(os.listdir(_RESOLVED_OWNERS_DIR)):
            owner_path = os.path.join(_RESOLVED_OWNERS_DIR, owner_dir_name)
            if not os.path.isdir(owner_path):
                continue
            meta_path = os.path.join(owner_path, "meta.json")
            if not os.path.exists(meta_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                embeddings = []
                for fname in sorted(os.listdir(owner_path)):
                    if fname.startswith("embedding_") and fname.endswith(".npy"):
                        # 单文件损坏不应连累整个 owner
                        try:
                            emb = np.load(os.path.join(owner_path, fname), allow_pickle=False)
                            embeddings.append(emb)
                        except Exception as e:
                            print(f"[FaceRecognizer] 跳过损坏样本 {owner_dir_name}/{fname}: {e}")
                if not embeddings:
                    print(f"[FaceRecognizer] 主人 {owner_dir_name} 无有效样本，跳过")
                    continue
                self._owners.append({
                    "id": owner_dir_name,
                    "name": meta.get("name", owner_dir_name),
                    "path": owner_path,
                    "embeddings": embeddings,
                })
            except Exception as e:
                print(f"[FaceRecognizer] 加载主人 {owner_dir_name} 失败: {e}")

    def list_owners(self):
        """列出所有已注册主人"""
        with self._lock:
            return [{"id": o["id"], "name": o["name"], "samples": len(o["embeddings"])} for o in self._owners]

    def register_owner(self, name):
        """注册新主人 (创建空目录)，返回 owner_id

        Args:
            name: 主人名字 (会用于目录名，做安全过滤)

        Returns:
            str: owner_id (目录名)，失败返回 None；同名主人已存在返回 "exists"
        """
        with self._lock:
            # 重名检查
            for o in self._owners:
                if o["name"] == name:
                    return "exists"

            # 安全过滤名字 → 文件名 (只保留字母数字下划线横线 + CJK 基本区/扩展A区)
            safe_name = "".join(c for c in name if (
                c.isalnum() or c in ("_", "-")
                or 0x4e00 <= ord(c) <= 0x9fff
                or 0x3400 <= ord(c) <= 0x4dbf
            ))
            if not safe_name:
                safe_name = "owner"
            # 找一个不冲突的 ID (限制最大重试 10000 次，防异常累积)
            owner_path = None
            owner_id = None
            for idx in range(1, 10001):
                candidate_id = f"owner_{idx:03d}_{safe_name}"
                candidate_path = os.path.join(_RESOLVED_OWNERS_DIR, candidate_id)
                try:
                    os.makedirs(candidate_path, exist_ok=False)  # 原子创建，防 TOCTOU
                    owner_id = candidate_id
                    owner_path = candidate_path
                    break
                except FileExistsError:
                    continue
            if owner_path is None:
                print(f"[FaceRecognizer] 注册失败: 找不到可用 ID")
                return None

            try:
                meta = {"name": name, "created": datetime.now().isoformat()}
                with open(os.path.join(owner_path, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                self._owners.append({
                    "id": owner_id,
                    "name": name,
                    "path": owner_path,
                    "embeddings": [],
                })
                print(f"[FaceRecognizer] 注册主人: {name} (id={owner_id})")
                return owner_id
            except Exception as e:
                print(f"[FaceRecognizer] 注册失败: {e}")
                return None

    def delete_owner(self, owner_id):
        """删除主人

        Returns:
            bool: True=删除成功, False=主人不存在或文件删除失败
        """
        import shutil
        with self._lock:
            for i, o in enumerate(self._owners):
                if o["id"] == owner_id:
                    # 审查 bug: 之前用 ignore_errors=True，rmtree 部分失败时静默返回，
                    # 内存里已 pop 但磁盘残留，重启后 _load_registry 重新扫描导致主人"复活"。
                    # 现在不忽略错误: rmtree 失败则不 pop，保持内存与磁盘一致。
                    try:
                        shutil.rmtree(o["path"])
                    except Exception as e:
                        print(f"[FaceRecognizer] 删除失败 (磁盘文件): {e}")
                        return False
                    self._owners.pop(i)
                    print(f"[FaceRecognizer] 删除主人: {owner_id}")
                    return True
            return False

    def capture_and_save_embedding(self, owner_id, frame):
        """采集一帧并保存到主人库

        Returns:
            dict: {"ok": bool, "msg": str}
        """
        if not self._initialized:
            return {"ok": False, "msg": "模块未初始化"}

        owner = None
        with self._lock:
            for o in self._owners:
                if o["id"] == owner_id:
                    owner = o
                    break
        if owner is None:
            return {"ok": False, "msg": f"未找到主人 {owner_id}"}

        # 检测人脸 + 编码
        result = self._detect_and_encode(frame, for_enrollment=True)
        if not result["ok"]:
            return {"ok": False, "msg": result["msg"]}

        faces = result["faces"]
        if not faces:
            return {"ok": False, "msg": "画面中未检测到人脸"}
        # 严格校验: 录入时只允许画面中有一张人脸，避免录错人 (身后大脸会被误录)
        if len(faces) > 1:
            return {"ok": False, "msg": f"画面中检测到 {len(faces)} 张人脸，请确保画面中只有您一人"}
        emb = faces[0]["embedding"]

        # embedding 有效性校验 (极端光照/检测框异常可能产生 nan/inf)
        if not np.all(np.isfinite(emb)):
            return {"ok": False, "msg": "embedding 异常 (光照不足或检测异常)，请重试"}

        # 保存 (sample_idx 计算 + np.save + 内存 append 全部在同一锁内，避免并发覆盖文件)
        try:
            with self._lock:
                # 扫描目录已有最大编号 + 1，防止手动删除文件后 len != 实际文件数 导致重号覆盖
                existing = []
                for fname in os.listdir(owner["path"]):
                    if fname.startswith("embedding_") and fname.endswith(".npy"):
                        try:
                            existing.append(int(fname[len("embedding_"):-len(".npy")]))
                        except ValueError:
                            continue
                sample_idx = (max(existing) + 1) if existing else 1
                fname = f"embedding_{sample_idx:03d}.npy"
                np.save(os.path.join(owner["path"], fname), emb)
                owner["embeddings"].append(emb)
            print(f"[FaceRecognizer] 保存 {owner['name']} 的第 {sample_idx} 个样本")
            return {"ok": True, "msg": f"已采集 {sample_idx}/{OWNER_SAMPLES_PER_PERSON}", "samples": sample_idx}
        except Exception as e:
            return {"ok": False, "msg": f"保存失败: {e}"}

    # ===================== 识别 =====================

    def _detect_and_encode(self, frame, for_enrollment=False):
        """检测人脸并计算 128D embedding

        Args:
            frame: RGB 图像
            for_enrollment: True=录入模式 (upsample=1 检测小脸更准, jittering=10 embedding 更鲁棒)
                            False=识别模式 (upsample=0 加速, jittering=0)

        Returns:
            dict: {"ok": bool, "faces": [{"box": (x,y,w,h), "embedding": np.ndarray}], "msg": str}
        """
        if not self._initialized:
            return {"ok": False, "faces": [], "msg": "模块未初始化"}

        try:
            # dlib 期望 RGB 输入
            if frame.ndim != 3 or frame.shape[2] != 3:
                return {"ok": False, "faces": [], "msg": "图像格式错误"}

            # 录入用 upsample=1 (慢但准, 小脸不漏), 跟随识别用 upsample=0 (快, ~10FPS 必需)
            upsample = 1 if for_enrollment else 0
            # 录入用 jittering=10 (dlib 推荐, 数据增强让 embedding 对姿态/光照更鲁棒)
            jittering = 10 if for_enrollment else 0
            dets = self._detector(frame, upsample)
            if not dets:
                return {"ok": True, "faces": [], "msg": "未检测到人脸"}

            faces = []
            for det in dets:
                shape = self._predictor(frame, det)
                emb = np.array(self._encoder.compute_face_descriptor(frame, shape, jittering))
                # dlib.rectangle → (x, y, w, h)
                box = (det.left(), det.top(), det.width(), det.height())
                faces.append({"box": box, "embedding": emb})

            return {"ok": True, "faces": faces, "msg": f"检测到 {len(faces)} 张脸"}
        except Exception as e:
            return {"ok": False, "faces": [], "msg": f"识别异常: {e}"}

    def detect_faces(self, frame):
        """公开方法: 只检测人脸位置 (不识别身份)

        Returns:
            list[dict]: [{"box": (x,y,w,h), "embedding": np.ndarray}, ...] 失败返回 []
        """
        result = self._detect_and_encode(frame)
        if result["ok"]:
            return result["faces"]
        return []

    def identify(self, face):
        """识别人脸身份

        Args:
            face: detect_faces 返回的 dict (含 embedding)，None 安全

        Returns:
            str or None: 匹配到的主人名，未识别返回 None
        """
        if not self._initialized:
            return None
        if not isinstance(face, dict) or "embedding" not in face:
            return None

        emb = face["embedding"]
        # 锁内只做浅拷贝快照，锁外做 O(N*M) 余弦计算，避免阻塞 register/delete/capture
        with self._lock:
            if not self._owners:
                return None
            snapshot = [(o["name"], list(o["embeddings"])) for o in self._owners]

        best_name = None
        best_sim = 0.0
        for name, samples in snapshot:
            for sample in samples:
                sim = self._cosine_similarity(emb, sample)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
        if best_sim >= FACE_MATCH_THRESHOLD:
            return best_name
        return None

    @staticmethod
    def _cosine_similarity(a, b):
        """余弦相似度 (0~1)"""
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        # 浮点比较用 1e-12 容差，防 norm 极小但非零导致除法溢出
        if norm < 1e-12:
            return 0.0
        return float(np.dot(a, b) / norm)

    def is_ready(self):
        """模块是否就绪 (initialized + 至少一个 owner 有有效样本)"""
        with self._lock:
            return self._initialized and any(o["embeddings"] for o in self._owners)

    def draw_detections(self, frame, faces, identifications=None):
        """在画面上绘制人脸框和识别结果 (用于 web 端可视化)

        注意: frame 来自 picamera2 RGB888 配置，是 RGB 顺序。
        cv2.rectangle/putText 直接写入像素值，颜色元组按 RGB 解释。
        所以绿色=(0,255,0)，红色=(255,0,0)。
        之前 bug: 红色写成 (0,0,255)，在 RGB 帧上实际渲染为蓝色。
        """
        import cv2
        if frame is None:
            return frame
        identifications = identifications or []
        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            name = identifications[i] if i < len(identifications) else None
            color = (0, 255, 0) if name else (255, 0, 0)  # RGB: 绿=识别到, 红=未知
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = name if name else "未知"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
