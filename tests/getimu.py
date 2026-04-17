import serial
import struct

# Open the serial port
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

def parse_imu():
    while True:
        # Search for the start byte (0x55)
        if ser.read() == b'\x55':
            # Read the rest of the header and data
            # Adjust the '12' depending on your specific board's packet size
            data = ser.read(12) 
            
            if len(data) == 12:
                # Unpack 6 signed shorts ('hhhhhh') 
                # These represent Accel X,Y,Z and Gyro X,Y,Z
                try:
                    vals = struct.unpack('>hhhhhh', data)
                    
                    ax, ay, az = vals[0:3]
                    gx, gy, gz = vals[3:6]
                    
                    print(f"Accel: {ax}, {ay}, {az} | Gyro: {gx}, {gy}, {gz}")
                except struct.error:
                    pass

try:
    print("Reading raw IMU stream...")
    parse_imu()
except KeyboardInterrupt:
    ser.close()