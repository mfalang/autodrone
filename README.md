# autodrone
Implementation of my master's thesis in cybernetics and robotics at NTNU. The
goal of the project is to take off and land a drone on a moving platform.

## Installation

### Pulling repository

The models are stored using Git LFS, and must therefore be pulled in order for
the simulator to work. First install Git LFS as explained
[here](http://arfc.github.io/manual/guides/git-lfs) and then run

```
cd autodrone
git lfs install
git lfs pull
```

### Installing Ros Melodic, Olympe SDK and Sphinx simulator

These scripts worked in the fall of 2021 prior to the new software releases from
Parrot, but have not been tested since.

```
cd autodrone
./ros/melodic/install.sh
./olympe/install.sh
./sphinx/install.sh
```

### Installing Python requirements

```
pip3 install -r requirements.txt
```

## Usage

Start Tmux

```
./tmux_open.sh
```

To start the Sphinx simulator run

```
./sphinx/start.sh land
```

To start the drone interface run

```
roslaunch drone_interface anafi_<sim/real>.launch
roslaunch drone_interface anafi_real_skycontroller.launch // if connecting using a skycontroller
```

Perception algorithms
```
roslaunch darknet_ros anafi_<sim/real> // needed for dnnCV to work
roslaunch perception tcv_<sim/real>
roslaunch_perception_dnnCV_<sim/real>
roslaunch perception ekf_<sim/real>  // also starts DNNCV and TCV systems
```

Control algorithms
```
roslaunch control mission_control.launch // the main system
roslaunch control lab_test.launch // some preset missions to run
roslaunch control track_helipad.launch // test tracking helipad only
```

Ground truth
```
roslaunch ground_truth sphinx_ros_bridge.launch // for GT in simulator
roslaunch ground_truth coordinate_transform_<sim/real> // for GT data in body
frame. Sim version requires sphinx_ros_bridge to run
```

Most parameters in all the different ros packages are set in their own config
files.