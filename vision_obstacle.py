"""
vision_obstacle.py - 视觉避障分析模块

纯 OpenCV 图像分析 (无新依赖)，用于辅助超声波避障:
  - 弥补超声波 15° 波束角的左右盲区
  - 检测低矮障碍物 (超声波可能从上方越过)
  - 提供"左/中/右"三段通行性判断，规划避障后新路线

原理:
  摄像头俯视前方时，画面下半部分是地面。地面纹理相对均匀，
  障碍物 (墙、桌腿、玩具) 与地面在亮度/梯度上显著不同。
  通过边缘检测 + 形态学运算提取障碍区域，按画面左中右三段
  统计障碍像素占比，得到通行性判断。

注意:
  - 该模块是"辅助"而非"替代"超声波。最终决策由 _auto_pilot_loop
    融合两者做出 (config.VISION_TRUST_LEVEL 控制信任度)。
  - 强光/暗光/纯色地毯等极端环境可能误判，融合决策会兜底。
"""

import numpy as np
import cv2
from config import (
    VISION_OBSTACLE_ROI_Y_START,
    VISION_OBSTACLE_BLOCK_RATIO,
    VISION_OBSTACLE_MIN_AREA,
)


class VisionObstacle:
    """视觉避障分析"""

    def __init__(self):
        self._initialized = False

    def init(self):
        """无外部资源需要加载，初始化恒成功"""
        self._initialized = True
        print("[VisionObstacle] 视觉避障分析模块就绪")
        return True

    def analyze(self, frame):
        """分析画面通行性

        Args:
            frame: RGB 图像 (numpy.ndarray)，来自 picamera2

        Returns:
            dict:
                {
                    "left_blocked": bool,    # 左 1/3 是否阻塞
                    "center_blocked": bool,  # 中 1/3 是否阻塞
                    "right_blocked": bool,   # 右 1/3 是否阻塞
                    "left_ratio": float,     # 左段障碍像素占比 0~1
                    "center_ratio": float,
                    "right_ratio": float,
                    "suggested_dir": str,    # "left"/"center"/"right"/"backward"
                    "ok": bool,              # 分析是否成功 (False 表示无有效数据)
                }
        """
        if not self._initialized or frame is None:
            return self._empty_result()

        try:
            h, w = frame.shape[:2]
            if h < 10 or w < 10:
                return self._empty_result()

            # 1. 取画面底部 ROI (近地区域)
            # 越靠下对应越近的地面，障碍物在这里最显眼
            y_start = int(h * VISION_OBSTACLE_ROI_Y_START)
            roi = frame[y_start:, :, :] if frame.ndim == 3 else frame[y_start:, :]

            # 2. 转灰度 + 高斯模糊降噪
            if roi.ndim == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = roi
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # 3. Canny 边缘检测 — 障碍物边缘密集，地面边缘稀疏
            edges = cv2.Canny(gray, 50, 150)

            # 4. 形态学闭运算 — 把相邻边缘连成块，便于找轮廓
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 5. 找轮廓并填充 — 障碍物轮廓内填充为实心
            mask = np.zeros_like(edges)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

            # 6. 按画面左中右三段统计障碍像素占比
            third_w = mask.shape[1] // 3
            if third_w < 5:
                return self._empty_result()

            left_mask = mask[:, :third_w]
            center_mask = mask[:, third_w:2 * third_w]
            right_mask = mask[:, 2 * third_w:]

            left_ratio = self._block_ratio(left_mask, third_w, 0)
            center_ratio = self._block_ratio(center_mask, third_w, 1)
            right_ratio = self._block_ratio(right_mask, third_w, 2)

            # 7. 判定阻塞 + 推荐方向
            left_blocked = left_ratio > VISION_OBSTACLE_BLOCK_RATIO
            center_blocked = center_ratio > VISION_OBSTACLE_BLOCK_RATIO
            right_blocked = right_ratio > VISION_OBSTACLE_BLOCK_RATIO

            suggested = self._suggest_direction(left_blocked, center_blocked, right_blocked)

            return {
                "left_blocked": left_blocked,
                "center_blocked": center_blocked,
                "right_blocked": right_blocked,
                "left_ratio": round(left_ratio, 3),
                "center_ratio": round(center_ratio, 3),
                "right_ratio": round(right_ratio, 3),
                "suggested_dir": suggested,
                "ok": True,
            }
        except Exception as e:
            print(f"[VisionObstacle] 分析异常: {e}")
            return self._empty_result()

    @staticmethod
    def _block_ratio(mask_segment, third_w, segment_idx):
        """计算单段内障碍像素占比

        小轮廓噪点已在前面 findContours 后通过 drawContours(FILLED) 合并，
        这里直接用段内非零像素 / 段面积，简化高效。
        """
        if mask_segment.size == 0:
            return 0.0
        non_zero = cv2.countNonZero(mask_segment)
        total = mask_segment.shape[0] * mask_segment.shape[1]
        return float(non_zero) / total if total > 0 else 0.0

    @staticmethod
    def _suggest_direction(left_blocked, center_blocked, right_blocked):
        """根据三段阻塞情况推荐避障方向"""
        if not center_blocked:
            return "center"  # 中部畅通，直行
        # 中部阻塞，看左右哪边畅通
        if not left_blocked and not right_blocked:
            return "left"  # 双侧畅通，默认左转 (习惯右行交通避让)
        if not left_blocked:
            return "left"
        if not right_blocked:
            return "right"
        return "backward"  # 三面都堵，倒车

    @staticmethod
    def _empty_result():
        return {
            "left_blocked": False,
            "center_blocked": False,
            "right_blocked": False,
            "left_ratio": 0.0,
            "center_ratio": 0.0,
            "right_ratio": 0.0,
            "suggested_dir": "center",
            "ok": False,
        }

    def draw_overlay(self, frame, analysis):
        """在画面上叠加可视化分析结果 (用于 web 端调试)

        在底部 ROI 画出左中右三段分界线，阻塞段标红，畅通段标绿。
        """
        if frame is None or not analysis.get("ok"):
            return frame

        h, w = frame.shape[:2]
        y_start = int(h * VISION_OBSTACLE_ROI_Y_START)
        third_w = w // 3

        # 三段分界线
        for i in range(1, 3):
            x = i * third_w
            cv2.line(frame, (x, y_start), (x, h), (255, 255, 0), 1)

        # ROI 上边线
        cv2.line(frame, (0, y_start), (w, y_start), (255, 255, 0), 1)

        # 每段顶部标注阻塞/畅通
        colors = [
            (0, 0, 255) if analysis["left_blocked"] else (0, 255, 0),
            (0, 0, 255) if analysis["center_blocked"] else (0, 255, 0),
            (0, 0, 255) if analysis["right_blocked"] else (0, 255, 0),
        ]
        labels = [
            f"L:{analysis['left_ratio']:.2f}",
            f"C:{analysis['center_ratio']:.2f}",
            f"R:{analysis['right_ratio']:.2f}",
        ]
        for i in range(3):
            cx = i * third_w + third_w // 2
            cv2.putText(frame, labels[i], (cx - 30, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

        # 推荐方向
        cv2.putText(frame, f"->{analysis['suggested_dir']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame
