#!/bin/bash
# wifi-ap-check.sh
# 启动时检测 WiFi 连接状态，未连接则开启 AP 热点
# 用途: 没有可用 WiFi 的环境下也能通过 AP 热点连接树莓派控制小车
#
# 部署方式 (systemd 服务开机自启):
#   sudo cp scripts/wifi-ap-check.sh /usr/local/bin/
#   sudo chmod +x /usr/local/bin/wifi-ap-check.sh
#   sudo tee /etc/systemd/system/wifi-ap-check.service > /dev/null <<'UNIT'
#   [Unit]
#   Description=WiFi Check & AP Fallback
#   After=network-online.target
#   Wants=network-online.target
#
#   [Service]
#   Type=oneshot
#   ExecStart=/usr/local/bin/wifi-ap-check.sh
#   RemainAfterExit=yes
#
#   [Install]
#   WantedBy=multi-user.target
#   UNIT
#   sudo systemctl daemon-reload
#   sudo systemctl enable wifi-ap-check.service
#
# 手动测试: sudo bash scripts/wifi-ap-check.sh

AP_SSID="car"
AP_PASSWORD="raspberry"
AP_CON_NAME="car-hotspot"
AP_CHANNEL=6                 # 2.4GHz channel 1/6/11 互不重叠，6 居中较稳
AP_IFACE="wlan0"             # 树莓派板载 WiFi 接口名
GRACE_PERIOD=15              # 等 WiFi 客户端模式尝试关联已知热点的宽限期 (秒)
LOG_TAG="wifi-ap-check"

log() { logger -t "$LOG_TAG" "$1"; echo "[$LOG_TAG] $1"; }

# 0. 依赖检查 — 缺少任一工具直接退出，避免执行到最后才失败看不出根因
if ! command -v nmcli >/dev/null 2>&1; then
    log "错误: 未找到 nmcli，请安装 NetworkManager (sudo apt install network-manager)"
    exit 1
fi
if ! command -v iwgetid >/dev/null 2>&1; then
    log "错误: 未找到 iwgetid，请安装 wireless-tools (sudo apt install wireless-tools)"
    exit 1
fi
if ! ip link show "$AP_IFACE" >/dev/null 2>&1; then
    log "错误: 未找到 WiFi 接口 $AP_IFACE"
    exit 1
fi

# 1. 等待 network-online.target 就绪
#    systemd 已通过 After= 依赖等待，此循环兼容 cron/手动调用场景
log "等待 network-online.target..."
for _ in $(seq 1 30); do
    if systemctl is-active --quiet network-online.target; then
        log "network-online.target 已就绪"
        break
    fi
    sleep 2
done

# 2. 宽限期: 给 WiFi 客户端模式留时间尝试关联已保存的热点
#    network-online.target 只代表"网络栈就绪"，不代表 WiFi 已关联成功
#    实测树莓派 WiFi 关联+DHCP 获取 IP 通常需要 5~10s，这里留 15s 余量
log "等待 ${GRACE_PERIOD}s WiFi 关联宽限期..."
sleep $GRACE_PERIOD

# 3. 检测 WiFi 是否已连接到某个 SSID 且拿到 IPv4 地址
#    iwgetid -r 只能判定是否关联到 AP，但 DHCP 失败时仍无 IP 无法访问
#    所以再补一层 ip -4 addr 检查，确保真正可用才退出
SSID=$(iwgetid -r 2>/dev/null)
WLAN_IP=$(ip -4 -o addr show "$AP_IFACE" 2>/dev/null | awk '{print $4}' | head -n1)
if [ -n "$SSID" ] && [ -n "$WLAN_IP" ]; then
    log "WiFi 已连接到 '$SSID' (IP=$WLAN_IP)，无需启动 AP"
    exit 0
fi
if [ -n "$SSID" ]; then
    log "WiFi 已关联 '$SSID' 但未获取 IP，等待 5s 后再检查..."
    sleep 5
    WLAN_IP=$(ip -4 -o addr show "$AP_IFACE" 2>/dev/null | awk '{print $4}' | head -n1)
    if [ -n "$WLAN_IP" ]; then
        log "DHCP 完成，IP=$WLAN_IP，无需启动 AP"
        exit 0
    fi
fi

# 4. WiFi 未连接 → 启动 AP 热点
log "WiFi 未连接，准备启动 AP 热点..."

# 使用 NetworkManager (树莓派 OS Bookworm 默认) 创建/激活 AP
# ipv4.method shared 会让树莓派做 DHCP 服务器，自身 IP 默认 10.42.0.1
if nmcli con show "$AP_CON_NAME" >/dev/null 2>&1; then
    # 配置已存在 (之前创建过)。检查 SSID 是否与脚本中的一致，
    # 不一致时删除重建，避免用户改了 AP_SSID 但脚本仍用旧配置
    EXISTING_SSID=$(nmcli -g 802-11-wireless.ssid con show "$AP_CON_NAME" 2>/dev/null)
    if [ -n "$EXISTING_SSID" ] && [ "$EXISTING_SSID" != "$AP_SSID" ]; then
        log "AP 配置 SSID='$EXISTING_SSID' 与脚本 SSID='$AP_SSID' 不一致，删除重建..."
        nmcli con delete "$AP_CON_NAME" >/dev/null 2>&1
    fi
fi

if nmcli con show "$AP_CON_NAME" >/dev/null 2>&1; then
    # 配置已存在 (且 SSID 一致)，直接激活
    log "AP 连接配置已存在，直接激活..."
    if nmcli con up "$AP_CON_NAME"; then
        AP_IP=$(ip -4 -o addr show "$AP_IFACE" 2>/dev/null | awk '{print $4}' | head -n1)
        log "AP 已启动: SSID=$AP_SSID  IP=${AP_IP:-未知}"
    else
        log "AP 激活失败"
        exit 1
    fi
else
    # 首次创建 AP 连接配置
    log "创建 AP 连接配置..."
    if nmcli con add type wifi ifname "$AP_IFACE" con-name "$AP_CON_NAME" \
        autoconnect no ssid "$AP_SSID" \
        wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$AP_PASSWORD" \
        802-11-wireless.mode ap 802-11-wireless.band bg \
        802-11-wireless.channel "$AP_CHANNEL" \
        ipv4.method shared; then
        if nmcli con up "$AP_CON_NAME"; then
            AP_IP=$(ip -4 -o addr show "$AP_IFACE" 2>/dev/null | awk '{print $4}' | head -n1)
            log "AP 已创建并启动: SSID=$AP_SSID  密码=$AP_PASSWORD  IP=${AP_IP:-未知}"
            log ">>> 用手机/电脑连接 WiFi '$AP_SSID'，然后 ssh pi@${AP_IP:-10.42.0.1} <<<"
        else
            log "AP 配置创建成功但启动失败"
            exit 1
        fi
    else
        log "AP 连接配置创建失败 (请确认 NetworkManager 已安装并在运行)"
        exit 1
    fi
fi
