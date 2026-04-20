#!/usr/bin/env python3
# pi_mode_listener.py
# Этот скрипт слушает MAVLink-канал и выводит на консоль все поступившие NAMED_VALUE_INT("CMD_MODE", ...).

import time
from pymavlink import mavutil

# ===== КОНФИГУРАЦИЯ =====
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600

def main():
    # 1) Подключаемся к Pixhawk по MAVLink
    master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
    master.wait_heartbeat()
    print("[Python] Подключено к Pixhawk (Heartbeat).")

    # 2) Вечный цикл: приём сообщений NAMED_VALUE_INT
    while True:
        # Блокируемся, пока не придёт NAMED_VALUE_INT или таймаут 1 сек
        msg = master.recv_match(type='NAMED_VALUE_INT', blocking=True, timeout=1)
        if msg:
            # В официальных сообщениях NAMED_VALUE_INT поле 'name' хранит до 10 символов (bytes)
            name = msg.name.decode('utf-8').rstrip('\x00')
            value = msg.value

            if name == "CMD_MODE":
                if value == 1:
                    print(f"[Python] Received CMD_MODE=1 → MODE_LETTERS")
                elif value == 2:
                    print(f"[Python] Received CMD_MODE=2 → MODE_ARUCO")
                elif value == 0:
                    print(f"[Python] Received CMD_MODE=0 → MODE_OFF")
                else:
                    print(f"[Python] Received CMD_MODE={value} → Неизвестный режим")
        else:
            # Если в течение 1 с не пришло NAMED_VALUE_INT — просто продолжаем ждать
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Python] Завершение по Ctrl+C")

