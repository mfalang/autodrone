# autodrone
My project thesis, aiming to take off and land a drone autonomously on a moving
platform.

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
