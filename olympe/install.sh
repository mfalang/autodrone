#!/bin/bash

# Script to install Olympe SDK
# Commands taken from https://developer.parrot.com/docs/olympe/installation.html
# with some modifications which were found to be needed

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}========== Installing Parrot Olympe ==========${NC}"

echo -e "${GREEN}\nInstalling script dependencies${NC}"
sudo apt update
sudo apt install -y curl python python-pip
sudo pip3 install --upgrade pip

echo -e "${GREEN}\nInstalling Repo tool${NC}"
mkdir -p ~/.bin
PATH="${HOME}/.bin:${PATH}"
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/.bin/repo
chmod a+rx ~/.bin/repo

echo -e "${GREEN}\nCloning source code${NC}"
cd /home/$USER
mkdir -p code/parrot-groundsdk
cd code/parrot-groundsdk
repo init -u https://github.com/Parrot-Developers/groundsdk-manifest.git
repo sync

echo -e "${GREEN}\nInstalling dependencies${NC}"
./products/olympe/linux/env/postinst

echo -e "${GREEN}\nBuilding Olympe${NC}"
./build.sh -p olympe-linux -A all final -j

echo -e "${GREEN}\nFixing Python path in Olympe navtive wrapper script${NC}"
sed -i '/# Update python path/!b;n;c\PYTHONPATH=${PYTHONPATH}:${SYSROOT}/usr/lib/python/site-packages' ./out/olympe-linux/final/native-wrapper.sh

echo -e "${GREEN}\nInstalling Olympe requirements in current Python environment${NC}"
pip install -r "./packages/olympe/requirements.txt"

echo -e "${GREEN}\nRemoving Repo tool${NC}"
rm -rf ~/.bin

echo -e "${GREEN}========== Installation finished ==========${NC}"
