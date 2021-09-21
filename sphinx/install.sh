#!/bin/bash

# Script to install Parrot Sphinx simulator
# Commands taken from https://developer.parrot.com/docs/sphinx/installation.html

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}========== Installing Parrot Sphinx ==========${NC}"

echo "deb http://plf.parrot.com/sphinx/binary `lsb_release -cs`/" | sudo tee /etc/apt/sources.list.d/sphinx.list > /dev/null
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 508B1AE5
sudo apt update
sudo apt install parrot-sphinx