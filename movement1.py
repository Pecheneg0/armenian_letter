from pymavlink import mavutil 
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600

the_connection = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)

the_connection.wait_heartbeat()
print ("Heartbeat from system (system %u component %u)" %
     (the_connection.target_system, the_connection.target_component))


the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message (10, the_connection.target_system, the_connection.target_component, 
                                     mavutil.mavlink.MAV_FRAME_LOCAL_NED, int(0b100111111000) 40, 0, -10, 0, 0, 0, 0, 0, 0, 1.57, 0))


print("Ожидание LOCAL_POSITION_NED...")
while True:
    msg = the_connection.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=5)
    if msg:
        print(f"[RX] LOCAL_POSITION_NED: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
    else:
        print("❌ Нет данных LOCAL_POSITION_NED (возможно, EKF не готов)")


