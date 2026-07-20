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
GRACE_PERIOD=15  # 等 WiFi 客户端模式尝试关联已知热点的宽限期 (秒)
LOG_TAG="wifi-ap-check"

log() { logger -t "$LOG_TAG" "$1"; echo "[$LOG_TAG] $1"; }

# 1. 等待 network-online.target 就绪
#    systemd 已通过 After= 依赖等待，此循环兼容 cron/手动调用场景
log "等待 network-online.target..."
for i in $(seq 1 30); do
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

# 3. 检测 WiFi 是否已连接到某个 SSID
#    iwgetid -r 返回当前关联的 SSID，未连接则输出空
SSID=$(iwgetid -r 2>/dev/null)
if [ -n "$SSID" ]; then
    log "WiFi 已连接到 '$SSID'，无需启动 AP"
    exit 0
fi

# 4. WiFi 未连接 → 启动 AP 热点
log "WiFi 未连接，准备启动 AP 热点..."

# 使用 NetworkManager (树莓派 OS Bookworm 默认) 创建/激活 AP
# ipv4.method shared 会让树莓派做 DHCP 服务器，自身 IP 默认 10.42.0.1
if nmcli con show "$AP_CON_NAME" >/dev/null 2>&1; then
    # 配置已存在 (之前创建过)，直接激活
    log "AP 连接配置已存在，直接激活..."
    if nmcli con up "$AP_CON_NAME"; then
        log "AP 已启动: SSID=$AP_SSID  IP=10.42.0.1"
    else
        log "AP 激活失败"
        exit 1
    fi
else
    # 首次创建 AP 连接配置
    log "创建 AP 连接配置..."
    if nmcli con add type wifi ifname wlan0 con-name "$AP_CON_NAME" \
        autoconnect no ssid "$AP_SSID" \
        wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$AP_PASSWORD" \
        802-11-wireless.mode ap 802-11-wireless.band bg \
        ipv4.method shared; then
        if nmcli con up "$AP_CON_NAME"; then
            log "AP 已创建并启动: SSID=$AP_SSID  密码=$AP_PASSWORD  IP=10.42.0.1"
            log ">>> 用手机/电脑连接 WiFi '$AP_SSID'，然后 ssh pi@10.42.0.1 <<<"
        else
            log "AP 配置创建成功但启动失败"
            exit 1
        fi
    else
        log "AP 连接配置创建失败 (请确认 NetworkManager 已安装并在运行)"
        exit 1
    fi
fi
