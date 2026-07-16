#!/bin/bash
# AI 小车一键启动脚本
# 使用方式: bash start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚗 AI 小车 - 启动中..."
echo ""

# 1. 确保 pigpiod 在运行 (舵机需要)
if ! pgrep -x pigpiod > /dev/null; then
    echo "[1/3] 启动 pigpiod 守护进程..."
    sudo pigpiod
    sleep 1
else
    echo "[1/3] pigpiod 已在运行"
fi

# 2. 激活虚拟环境
echo "[2/3] 激活 Python 环境..."
if [ ! -d ".venv" ]; then
    echo "  创建虚拟环境..."
    python3 -m venv --system-site-packages .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 3. 启动主程序
echo "[3/3] 启动主控程序..."
echo ""
echo "========================================"
echo "   请在浏览器打开:"
echo "   http://$(hostname -I | awk '{print $1}'):5000"
echo "========================================"
echo ""

python3 main.py
