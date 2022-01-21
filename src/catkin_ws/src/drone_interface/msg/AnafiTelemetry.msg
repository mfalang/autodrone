# Telemetry data from Anafi drone. Published at 5 Hz as this is the
# update frequency of the drone output

Header header

# Attitude in Euler angles in degrees
float64 roll
float64 pitch
float64 yaw # 0 deg = north, 90 deg = east

# Body velocity in m/s
float64 vx # positive forward
float64 vy # positive right
float64 vz # positive down

# Relative altitude
float64 relative_altitude # negative, as in NED

# Gimbal attitude
float64 gimbal_roll
float64 gimbal_pitch # 0 at horizon, positive upwards
float64 gimbal_yaw

# Battery information
float64 battery_percentage
string battery_health
string battery_serial_number

# Flying state
string flying_state
