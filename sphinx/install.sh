#!/bin/bash

# Script to install Parrot Sphinx simulator, UUV simulator, and custom models

SCRIPT_DIR=$(dirname "$(realpath $0)")

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}========== Installing Simulator and models ==========${NC}"

read -p "Install Parrot Sphinx? [Y/n] " -n 1 -r reply
echo
if [[ $reply != "n" ]]; then
    echo -e "${GREEN}Installing Parrot Sphinx${NC}"
    # Source: https://developer.parrot.com/docs/sphinx/installation.html

    echo "deb http://plf.parrot.com/sphinx/binary `lsb_release -cs`/" | sudo tee /etc/apt/sources.list.d/sphinx.list > /dev/null
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 508B1AE5
    sudo apt update
    sudo apt install -y parrot-sphinx    
    sudo systemctl enable firmwared
fi

read -p "Install UUV Simulator? [Y/n] " -n 1 -r reply
echo
if [[ $reply != "n" ]]; then
    echo -e "${GREEN}Installing UUV Simulator${NC}"
    # Source: https://github.com/uuvsimulator/uuv_simulator

    sudo apt install -y ros-melodic-uuv-simulator
fi

read -p "Install custom models? [Y/n] " -n 1 -r reply
echo
if [[ $reply != "n" ]]; then
    echo -e "${GREEN}Installing custom models${NC}"

    cp -r $SCRIPT_DIR/models/ $HOME/.gazebo
fi

echo -e "${GREEN}========== Done ==========${NC}"



