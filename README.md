# AI 智能小车

用树莓派 + USB 摄像头打造的 AI 小车。

## 硬件
- 树莓派（Raspberry Pi OS）
- USB 摄像头（带麦克风）
- 电机驱动板（待定）
- 底盘（待定）

## 项目结构
```
ai-car/
├── camera/          # 摄像头模块
│   ├── photo.py     # 拍照
│   ├── video.py     # 录像
│   └── detect.py    # 物体检测
├── motor/           # 电机控制
├── ai/              # AI 视觉/语音
└── main.py          # 主程序
```
