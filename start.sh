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

# 3. 启动主程序 (崩溃自动重启)
echo "[3/3] 启动主控程序..."
echo ""
echo "========================================"
echo "   请在浏览器打开:"
echo "   http://$(hostname -I | awk '{print $1}'):2222"
echo "========================================"
echo ""

# 崩溃自愈: dlib 人脸识别在底层 C++ 层有 SIGSEGV (exit 139) 风险，
# 代码内已加线程锁防护，此处再加一层兜底 — 异常退出时自动重启，
# 避免小车在运行中"死机"失控。
# 注意 set +e: 主程序非零退出不能触发 set -e 直接终结本脚本
set +e
while true; do
    python3 main.py
    rc=$?
    # 0=正常退出, 130=Ctrl+C (SIGINT) → 不重启，正常结束
    if [ $rc -eq 0 ] || [ $rc -eq 130 ]; then
        break
    fi
    echo ""
    echo "[!] 主程序异常退出 (code $rc)，3 秒后自动重启... (再按 Ctrl+C 彻底退出)"
    sleep 3
done
