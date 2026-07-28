#!/bin/bash
# AI 小车一键启动脚本
# 使用方式: sudo bash start.sh
# (需要 sudo: RPi.GPIO 访问 /dev/mem 需要 root 权限)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ========== 热点配置 ==========
# 启动小车服务时同时开启树莓派热点，手机直连热点传输控制更稳定。
# 关闭服务时同步关闭热点。
HOTSPOT_SSID="car"
HOTSPOT_PASS="13828287650"
HOTSPOT_CONN="ai-car-hotspot"   # nmcli 连接名 (固定，便于启停)

start_hotspot() {
    echo "[热点] 开启 WiFi 热点 ($HOTSPOT_SSID)..."
    # 若已有同名连接，先删除重建 (确保配置正确)
    nmcli connection delete "$HOTSPOT_CONN" 2>/dev/null || true
    # 创建热点连接 (ifname wlan0, 自动分配 IP 10.42.0.1)
    nmcli connection add type wifi ifname wlan0 con-name "$HOTSPOT_CONN" \
        autoconnect no ssid "$HOTSPOT_SSID" \
        wifi-sec.key-mgmt wpa-psk \
        wifi-sec.psk "$HOTSPOT_PASS" \
        ipv4.method shared \
        ipv6.method ignore 2>/dev/null
    nmcli connection up "$HOTSPOT_CONN" 2>/dev/null || true
    echo "[热点] 热点已开启: SSID=$HOTSPOT_SSID 密码=$HOTSPOT_PASS"
    echo "[热点] 手机连接后访问: http://10.42.0.1:2222"
}

stop_hotspot() {
    echo "[热点] 关闭 WiFi 热点..."
    nmcli connection down "$HOTSPOT_CONN" 2>/dev/null || true
    nmcli connection delete "$HOTSPOT_CONN" 2>/dev/null || true
    echo "[热点] 热点已关闭"
}

# 确保脚本真正退出时关闭热点 (仅绑 EXIT，不绑 INT/TERM)。
# 之前 bug: 绑 INT 后，用户在重启间隔 sleep 3 期间按 Ctrl+C 会触发 INT trap
# 关闭热点，但 set +e 让脚本继续循环重启 main.py，热点却不重开 → 手机断连。
# 现在只绑 EXIT: Ctrl+C 在 python3 main.py 期间退出 (rc=130) → break 循环
# → 脚本退出 → EXIT trap 关热点。重启间隔的 sleep 被 Ctrl+C 中断后 set +e
# 继续循环，热点保持开启 (不会误关)。
trap stop_hotspot EXIT

echo "🚗 AI 小车 - 启动中..."
echo ""

# 1. 确保 pigpiod 在运行 (舵机需要)
if ! pgrep -x pigpiod > /dev/null; then
    echo "[1/4] 启动 pigpiod 守护进程..."
    sudo pigpiod
    sleep 1
else
    echo "[1/4] pigpiod 已在运行"
fi

# 2. 开启 WiFi 热点 (手机直连传输控制更稳定)
echo "[2/4] 开启 WiFi 热点..."
start_hotspot

# 3. 激活虚拟环境
echo "[3/4] 激活 Python 环境..."
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

# 4. 启动主程序 (崩溃自动重启)
echo "[4/4] 启动主控程序..."
echo ""
echo "========================================"
echo "   控制面板地址:"
echo "   热点直连: http://10.42.0.1:2222"
echo "   局域网:   http://$(hostname -I | awk '{print $1}'):2222"
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
