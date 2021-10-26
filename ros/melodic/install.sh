#!/bin/bash

# Script to install ROS Melodic
# Commands taken from http://wiki.ros.org/melodic/Installation/Ubuntu

SCRIPT_DIR=$(dirname "$(realpath $0)")

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}========== Installing Ros Melodic ==========${NC}"

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install -y curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

sudo apt update
sudo apt install -y ros-melodic-desktop-full

echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

sudo apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential

sudo apt install -y python-rosdep

sudo rosdep init
rosdep update

echo -e "${GREEN}Building catkin workspace and updating ~/.bashrc${NC}"

source $HOME/.bashrc

cd $SCRIPT_DIR/../../src/catkin_ws
catkin_make

echo "source `pwd`/devel/setup.bash" >> ~/.bashrc

echo -e "${GREEN}Done. Remember to source ~/.bashrc${NC}"
