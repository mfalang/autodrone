# autodrone
My project thesis, aiming to take off and land a drone autonomously on a moving
platform.

## Usage

To start the Sphinx simulator run

```
./sphinx/start.sh land
```

To run a program in the simulator, first start the Olympe environment in a 
terminal

```
source olympe/start.sh
source src/catkin_ws/devel/setup.bash
rosrun melodic_sandbox simple_mission.py
```
