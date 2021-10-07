#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

source "$HOME/code/parrot-groundsdk/out/olympe-linux/final/native-wrapper.sh"

echo -e "${GREEN}========== Started Parrot Olympe environment ==========${NC}"

echo -e "System: `lsb_release -sd` (`lsb_release -sc`)"

echo -e "ROS version: `rosversion -d`"

echo -e "Python version: `python3 -V`"
