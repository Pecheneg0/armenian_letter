from pymavlink import mavutil
import subprocess
import time
import os
import signal

# Пути к скриптам
SCRIPT_ARUCO = "/home/pi/tests/Part_land_2.py"
SCRIPT_LETTERS = "/home/pi/tests/skan_letters_rp4_3.py"

# Активные процессы
aruco_proc = None
letters_proc = None

# Остановка процесса
def stop_proc(proc):
    if proc and proc.poll() is None:
        print("🛑 Остановка процесса...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(1)

# Запуск процесса
def start_proc(script_path):
    print(f"🚀 Запуск: {script_path}")
    return subprocess.Popen(["python3", script_path], preexec_fn=os.setsid)

# Подключение к Pixhawk
print("🔌 Подключение к Pixhawk...")
master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
master.wait_heartbeat()
print("✅ Подключение установлено")

# Флаги для контроля
triggered_aruco = False
triggered_letters = False

print("🎮 RC6 → ArUco | RC7 → Armenian Letters")
while True:
    msg = master.recv_match(type="RC_CHANNELS", blocking=True, timeout=1)
    if msg:
        rc6 = msg.chan6_raw
        rc7 = msg.chan7_raw

        # ▶️ RC6 активен → ArUco режим
        if rc6 > 1800 and not triggered_aruco:
            print("📡 Получена команда на запуск ArUco")
            stop_proc(letters_proc)
            letters_proc = None
            if not aruco_proc or aruco_proc.poll() is not None:
                aruco_proc = start_proc(SCRIPT_ARUCO)
            triggered_aruco = True
            triggered_letters = False

        elif rc6 < 1500:
            triggered_aruco = False

        # ▶️ RC7 активен → Armenian Letters режим
        if rc7 > 1800 and not triggered_letters:
            print("📡 Получена команда на запуск Armenian Letter Scan")
            stop_proc(aruco_proc)
            aruco_proc = None
            if not letters_proc or letters_proc.poll() is not None:
                letters_proc = start_proc(SCRIPT_LETTERS)
            triggered_letters = True
            triggered_aruco = False

        elif rc7 < 1500:
            triggered_letters = False

    time.sleep(0.2)
