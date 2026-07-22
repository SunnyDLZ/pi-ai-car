#!/bin/bash
# AI 小车一键启动脚本
# 使用方式: sudo bash start.sh
# (需要 sudo: RPi.GPIO 访问 /dev/mem 需要 root 权限)

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
    # 审查 bug: pip install 失败会留下半成品 .venv，下次启动跳过安装直接崩
    if ! pip install -r requirements.txt; then
        echo "[!] 依赖安装失败，清理半成品 .venv"
        deactivate 2>/dev/null || true
        rm -rf .venv
        exit 1
    fi
else
    source .venv/bin/activate
fi

# 3. 启动主程序
echo "[3/3] 启动主控程序..."
echo ""
echo "========================================"
echo "   请在浏览器打开:"
echo "   http://$(hostname -I | awk '{print $1}'):2222"
echo "========================================"
echo ""

# 审查 bug: 之前用 python3 main.py 未加 sudo，RPi.GPIO 访问 GPIO 需要 root 权限
# 用 exec 让信号直接到达 python 进程 (Ctrl+C 能正确触发 cleanup)
exec python3 main.py
