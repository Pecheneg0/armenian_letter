from pymavlink import mavutil
import numpy as np
import math
import logging
import time


MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600

class PrecisionLandingSystem:
    def __init__(self):
        self.master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
        self.master.wait_heartbeat()
        print(f"Подключение установлено {self.master.target_system}")

        self.min_altitude = 0.2
        self.pos_tolerance = 0.1
        self.drone_in_vtol = False
        self.mode_guided = False

        self.test_sequence = [
            {"n": 0, "e": 0},
            {"n": 0, "e": 0},
            {"n": 0, "e": 0},
            {"n": 0, "e": 0},
            {"n": -0.02, "e": 0},
            {"n": 0, "e": 0},
            {"n": 0, "e": 0}

        ]
        self.test_index = 0
        self.land_command_sent = False

    def calculate_offset(self, dx, dy):
        x_cam = dx
        y_cam = dy
        yaw_deg = self.get_yaw()
        yaw_rad = np.radians(yaw_deg)
        R = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)]
        ])
        offset = R @ np.array([x_cam, y_cam])
        return offset[0], offset[1]
    def get_yaw(self):
        try:
            msg = self.master.recv_match(type='ATTITUDE', blocking = True)
            if msg:
                yaw = math.degrees(msg.yaw)
                logging.info(f"Курс: {yaw:.1f}")
                print(f"Курс : {yaw:.1f}")
                return yaw
        except Exception as e:
            logging.error(f"Ошибка получения курса {e}")
        return 0.0

    def get_current_altitude(self):
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking = True)
        return msg.relative_alt / 1000
    
    def move_to_offset(self, dx, dy, target_alt=None):
        current_alt = self.get_current_altitude()

        if target_alt is None:
            target_alt = current_alt

        self.master.mav.set_position_target_local_ned_send(
            10,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
            int(0b100111111000),
            dx, dy, target_alt,
            0, 0, 0, 0, 0, 0, 0, 0 )


    def switch_to_vtol_mode(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
            0, 3, 0, 0, 0, 0, 0, 0

        )
        print("Переход в режим VTOL ")
        self.drone_in_vtol = True


    def execute_landing(self): # переписать функцию.
        print ("\n=== Начало процедуры посадки ===")
        start_time = time.time()
        last_command_time = 0
        command_interval = 2
        try:
            if not self.mode_guided:
                try:
                    self.master.mav.command_long_send (
                        self.master.target_system,
                        self.master.target_component,
                        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                        0,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        15, 0, 0, 0, 0, 0
                        )
                    self.mode_guided = True
                except Exception as e:
                    logging.error(f"Ошибка установки режима : {e}")


            while not self.land_command_sent:
                altitude = self.get_current_altitude()
                print(f"Текущая высота : {altitude:.2f} m")
                current_time = time.time()
                if current_time - last_command_time < command_interval:
                    time.sleep(0.01)
                    continue

                test_data = self.test_sequence[self.test_index]
                x_offset = test_data["n"]
                y_offset = test_data["e"]

                n_offset, e_offset = self.calculate_offset(x_offset, y_offset)

                self.test_index = (self.test_index + 1) % len(self.test_sequence)

                print(f"Тестовые смещения : Север {n_offset}, Восток : {e_offset}")
                if altitude > self.min_altitude :

                    if abs(n_offset) < self.pos_tolerance and abs(e_offset) < self.pos_tolerance:
                        print("Малые смещения, снижение")
                        altitude = self.get_current_altitude()
                        if altitude > 3:
                            new_alt = 0.8
                            self.move_to_offset(n_offset, e_offset, new_alt)
                        elif altitude <= 3 :
                            new_alt = 0.2
                            self.move_to_offset(n_offset, e_offset, new_alt)

                    else:
                        print(f"Коррекция : Север={n_offset}, восток = {e_offset}")
                        self.move_to_offset(n_offset, e_offset, 0)
                    time.sleep(2)

                else :
                    if abs(n_offset) < self.pos_tolerance and abs(e_offset) < self.pos_tolerance:
                        self.master.set_mode("QLAND")
                        print("Комаанда посадки отправлена")
                        self.land_command_sent = True
                    else:
                        print(f"Коррекция : Север={n_offset}, восток = {e_offset}")
                        self.move_to_offset(n_offset, e_offset, 0)
            time.sleep(0.05)

        except Exception as e:
            print (f"Критическая ошибка : {str(e)}")


if __name__ == "__main__":
    try:

        lander = PrecisionLandingSystem()
        lander.execute_landing()
    except KeyboardInterrupt:
        print("Программа завершене пользователем")
    except Exception as e :
        print (f"Фатальная ошибка {str(e)}")

