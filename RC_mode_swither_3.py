from pymavlink import mavutil
import subprocess
import time
import os
import signal

# Пути к скриптам
SCRIPT_ARUCO = "/home/pi/tests/Part_land_3.py"
SCRIPT_LETTERS = "/home/pi/tests/skan_letters_rp4_3.py"

active_proc = None
active_mode = "NONE"

def stop_proc(proc):
    if proc and proc.poll() is None:
        print("🛑 Остановка процесса...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(1)

def start_proc(script_path):
    print(f"🚀 Запуск: {script_path}")
    return subprocess.Popen(["python3", script_path], preexec_fn=os.setsid)

def send_status(text):
    try:
        master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, text.encode('utf-8'))
        print(f"📡 Status → Mission Planner: {text}")
    except Exception as e:
        print(f"Ошибка при отправке статуса: {e}")

# Подключение к Pixhawk
print("🔌 Подключение к Pixhawk...")
master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
master.wait_heartbeat()
print("✅ Связь с Pixhawk установлена")

print("🎮 Тумблер RC6: [1000=OFF | 1500=LETTERS | 2000=ARUCO]")

while True:
    msg = master.recv_match(type="RC_CHANNELS", blocking=True, timeout=1)
    if msg:
        rc6 = msg.chan6_raw
        print(f"RC6: {rc6} | Активный режим: {active_mode}")

        if rc6 > 1800 and active_mode != "ARUCO":
            stop_proc(active_proc)
            active_proc = start_proc(SCRIPT_ARUCO)
            active_mode = "ARUCO"
            send_status("Mode: ARUCO")

        elif 1300 < rc6 < 1700 and active_mode != "LETTERS":
            stop_proc(active_proc)
            active_proc = start_proc(SCRIPT_LETTERS)
            active_mode = "LETTERS"
            send_status("Mode: LETTERS")

        elif rc6 < 1200 and active_mode != "NONE":
            stop_proc(active_proc)
            active_proc = None
            active_mode = "NONE"
            send_status("Mode: OFF")

    time.sleep(0.2)
