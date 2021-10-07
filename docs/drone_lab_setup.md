# Setup for testing in the drone lab

## Set up model in Measurement4
1. Physical setup
    1. Add markers to the drone
    2. Make sure drone starts in the origin of the world corrdinate frame (mark 
    on the floor)
2. Measurement4 setup on lab PC
    1. Start "Measurement4" program and choose the project
    2. Click on "Capture..." under "Capture"
    3. Click "Cancel"
    4. Click 3D view
    5. Shift-click and drag to select the markers so that they are white (if 
    they are not already blue and have a body name, then skip this and the next
    step)
    6. Right click on one of the markers and click "Define rigid body 
    (6DOF)/Current Frame"
3. Aligning local and world frame
    1. Click on "Tools"
    2. Choose "6DOF Tracking"
    3. Choose the model
    4. Click "Translate" to translate the body along one of the axis or 
    similarly "Rotate" to rotate. The local coordinate system should line up 
    with the world  coordinate system as well as possible in order to get the 
    most accurate data.


## Set up streaming of tracking data to laptop
1. Connect laptop to ethernet
2. On laptop, if not done: assign static IP address to computer
   1. Go to Settings/Network
   2. Click the gear icon under "Wired"
   3. Under the IPv4 tab, fill in an IP address and netmask

                Address: 192.168.0.50 // for example
                Netmask: 24

    4. Choose "Apply"
    5. Verify that the address is correct by running `ifconfig` or `ip a`
3. Synchronize clocks of lab PC and laptop
    1. On laptop, install network time protocol tools

                sudo apt install ntpdate

    2. On laptop, Synchronize with lab PC using and repeat until difference is 
    less than 1ms

                sudo ntpdate 192.168.0.41
4. Set up environment variables on lab PC
    1. Go to "System properties" and press "Environmental variables"
    2. Under "System variables", make sure the two following variables are set 
    as follows

                ROS_IP = 192.168.0.41   // IP of the lab computer
                ROS_MASTER_URI = http://192.168.0.50:11311  // IP of laptop

5. Start streaming data to ROS topics
    1. On lab PC, open windows terminal
    2. Make sure the current folder is `C:\Users\ntnua\ros_arena_ws`
    3. Run the following two commands

                devel\setup.bat
                roslaunch mocap_qualisys qualisys.launch

    4. The topics from qualisys should now be available on the laptop (check 
    using `rostopic list`) 

## After completed work
1. Unplug power cable to the cameras (connector underneath the monitor)