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

        # 检查模型文件
        if not os.path.exists(FACE_LANDMARK_MODEL):
            print(f"[FaceRecognizer] 缺失模型: {FACE_LANDMARK_MODEL}")
            print(f"[FaceRecognizer] 下载: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            return False
        if not os.path.exists(FACE_RECOGNITION_MODEL):
            print(f"[FaceRecognizer] 缺失模型: {FACE_RECOGNITION_MODEL}")
            print(f"[FaceRecognizer] 下载: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
            return False

        try:
            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(FACE_LANDMARK_MODEL)
            self._encoder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)
        except Exception as e:
            print(f"[FaceRecognizer] 模型加载失败: {e}")
            return False

        # 加载主人库
        os.makedirs(OWNERS_DIR, exist_ok=True)
        self._load_registry()

        self._initialized = True
        print(f"[FaceRecognizer] 初始化完成，已加载 {len(self._owners)} 个主人")
        return True

    # ===================== 主人库管理 =====================

    def _load_registry(self):
        """从磁盘加载主人库"""
        self._owners = []
        if not os.path.isdir(OWNERS_DIR):
            return

        for owner_dir_name in sorted(os.listdir(OWNERS_DIR)):
            owner_path = os.path.join(OWNERS_DIR, owner_dir_name)
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
                        emb = np.load(os.path.join(owner_path, fname))
                        embeddings.append(emb)
                if not embeddings:
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
            str: owner_id (目录名)，失败返回 None
        """
        with self._lock:
            # 安全过滤名字 → 文件名 (只保留字母数字下划线中文)
            safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-") or "\u4e00" <= c <= "\u9fff")
            if not safe_name:
                safe_name = "owner"
            # 找一个不冲突的 ID
            idx = 1
            while True:
                owner_id = f"owner_{idx:03d}_{safe_name}"
                owner_path = os.path.join(OWNERS_DIR, owner_id)
                if not os.path.exists(owner_path):
                    break
                idx += 1

            try:
                os.makedirs(owner_path, exist_ok=False)
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
        """删除主人"""
        import shutil
        with self._lock:
            for i, o in enumerate(self._owners):
                if o["id"] == owner_id:
                    try:
                        shutil.rmtree(o["path"], ignore_errors=True)
                        self._owners.pop(i)
                        print(f"[FaceRecognizer] 删除主人: {owner_id}")
                        return True
                    except Exception as e:
                        print(f"[FaceRecognizer] 删除失败: {e}")
                        return False
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
        result = self._detect_and_encode(frame)
        if not result["ok"]:
            return {"ok": False, "msg": result["msg"]}

        # 取最大的人脸作为目标 (假设同一帧只有该主人)
        faces = result["faces"]
        if not faces:
            return {"ok": False, "msg": "画面中未检测到人脸"}
        target = max(faces, key=lambda f: (f["box"][2] * f["box"][3]))
        emb = target["embedding"]

        # 保存
        try:
            sample_idx = len(owner["embeddings"]) + 1
            fname = f"embedding_{sample_idx:03d}.npy"
            np.save(os.path.join(owner["path"], fname), emb)
            with self._lock:
                owner["embeddings"].append(emb)
            print(f"[FaceRecognizer] 保存 {owner['name']} 的第 {sample_idx} 个样本")
            return {"ok": True, "msg": f"已采集 {sample_idx}/{OWNER_SAMPLES_PER_PERSON}", "samples": sample_idx}
        except Exception as e:
            return {"ok": False, "msg": f"保存失败: {e}"}

    # ===================== 识别 =====================

    def _detect_and_encode(self, frame):
        """检测人脸并计算 128D embedding

        Args:
            frame: RGB 图像

        Returns:
            dict: {"ok": bool, "faces": [{"box": (x,y,w,h), "embedding": np.ndarray}], "msg": str}
        """
        if not self._initialized:
            return {"ok": False, "faces": [], "msg": "模块未初始化"}

        try:
            import dlib
            # dlib 期望 RGB 输入
            if frame.ndim != 3 or frame.shape[2] != 3:
                return {"ok": False, "faces": [], "msg": "图像格式错误"}

            # HOG 检测 (upsample=1 提高小脸检测率，约 2x 慢)
            dets = self._detector(frame, 1)
            if not dets:
                return {"ok": True, "faces": [], "msg": "未检测到人脸"}

            faces = []
            for det in dets:
                shape = self._predictor(frame, det)
                # 128D encoding; jittering=0 加速 (录入时可加 10 增强鲁棒)
                emb = np.array(self._encoder.compute_face_descriptor(frame, shape, 0))
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
            face: detect_faces 返回的 dict (含 embedding)

        Returns:
            str or None: 匹配到的主人名，未识别返回 None
        """
        if not self._initialized or "embedding" not in face:
            return None

        emb = face["embedding"]
        with self._lock:
            if not self._owners:
                return None
            best_name = None
            best_sim = 0.0
            for owner in self._owners:
                # 与该主人的所有样本比对，取最高相似度
                for sample in owner["embeddings"]:
                    sim = self._cosine_similarity(emb, sample)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = owner["name"]
        # 阈值判定
        if best_sim >= FACE_MATCH_THRESHOLD:
            return best_name
        return None

    @staticmethod
    def _cosine_similarity(a, b):
        """余弦相似度 (0~1)"""
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        # embedding 已归一化, 这里 float() 防溢出
        return float(np.dot(a, b) / norm)

    def is_ready(self):
        return self._initialized and len(self._owners) > 0

    def draw_detections(self, frame, faces, identifications=None):
        """在画面上绘制人脸框和识别结果 (用于 web 端可视化)"""
        import cv2
        if frame is None:
            return frame
        identifications = identifications or []
        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            name = identifications[i] if i < len(identifications) else "?"
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = name if name else "未知"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
