from pymavlink import mavutil
import time

# Подключение
master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
master.wait_heartbeat()
print("✅ Подключение установлено.")

# Смещения
offset_north = 5.0
offset_east = 3.0
offset_down = 0.0
yaw = 0.0

# Маска: игнорируем всё кроме позиции и yaw
type_mask = int(0b100111111000)

# Отправка
master.mav.send(
    mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        0,  # или int(time.monotonic() * 1000) % 2**32
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        offset_north,
        offset_east,
        offset_down,
        0, 0, 0,
        0, 0, 0,
        yaw,
        0
    )
)

print("Ожидание LOCAL_POSITION_NED...")
while True:
    msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=5)
    if msg:
        print(f"[RX] LOCAL_POSITION_NED: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
    else:
        print("❌ Нет данных LOCAL_POSITION_NED (возможно, EKF не готов)")
# Проверка получения позиции
