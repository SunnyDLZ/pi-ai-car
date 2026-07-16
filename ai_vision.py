"""
ai_vision.py - AI 视觉识别模块

基于 OpenCV DNN + MobileNet SSD (COCO 数据集)
支持: 物体检测 (人、车、动物等 20 类)
"""

import os
import urllib.request
import numpy as np
import cv2
from config import DETECTION_CONFIDENCE, COCO_CLASSES, \
    MOBILENET_PROTOTXT, MOBILENET_MODEL


# MobileNet SSD 模型下载地址 (Caffe)
MODEL_URLS = {
    "prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
}


class AIVision:
    """AI 视觉识别"""

    def __init__(self):
        self._net = None
        self._classes = COCO_CLASSES
        self._initialized = False

    def init(self):
        """加载 MobileNet SSD 模型 (必要时自动下载)"""
        model_dir = os.path.dirname(MOBILENET_MODEL)
        os.makedirs(model_dir, exist_ok=True)

        # 检查模型文件是否存在
        proto_path = MOBILENET_PROTOTXT
        model_path = MOBILENET_MODEL

        if not os.path.exists(proto_path):
            print("[AIVision] 下载模型配置文件...")
            urllib.request.urlretrieve(MODEL_URLS["prototxt"], proto_path)

        if not os.path.exists(model_path):
            print("[AIVision] 下载模型权重 (~30MB)...")
            urllib.request.urlretrieve(MODEL_URLS["caffemodel"], model_path)
            print("[AIVision] 模型下载完成")

        # 加载模型
        self._net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._initialized = True
        print("[AIVision] MobileNet SSD 模型加载完成")
        return True

    def detect(self, frame):
        """对图像进行物体检测

        Args:
            frame: RGB 图像 (numpy.ndarray)

        Returns:
            list[dict]: 检测结果列表，每个元素:
                {
                    "label": str,    # 类别名称
                    "class_id": int, # 类别 ID
                    "confidence": float, # 置信度 0~1
                    "box": (x, y, w, h) # 边界框 (像素)
                }
        """
        if not self._initialized or self._net is None:
            return []

        h, w = frame.shape[:2]

        # 构建 blob 输入
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self._net.setInput(blob)
        detections = self._net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < DETECTION_CONFIDENCE:
                continue

            class_id = int(detections[0, 0, i, 1])
            if class_id >= len(self._classes):
                continue

            # 边界框坐标 (归一化 → 像素)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # 裁剪到图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            results.append({
                "label": self._classes[class_id],
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "box": (x1, y1, x2 - x1, y2 - y1),
            })

        return results

    def draw_detections(self, frame, detections):
        """在图像上绘制检测框和标签

        Args:
            frame: RGB 图像
            detections: detect() 返回的结果列表

        Returns:
            numpy.ndarray: 标注后的图像
        """
        for det in detections:
            x, y, w, h = det["box"]
            label = f"{det['label']} {det['confidence']:.2f}"

            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 10, y), (0, 255, 0), -1)

            # 绘制标签文字
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame
