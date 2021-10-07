#!/bin/bash

usage="Usage: $(basename "$0") <test-name>"

if [ $# -ne 1 ]
  then
    echo $usage
    exit
fi

TEST_NAME=$1
SCRIPT_DIR=$(dirname "$(realpath $0)")

DATE=$(date +%d-%m-%Y)
OUTPUT_DIR=$SCRIPT_DIR/../../out/$DATE/$TEST_NAME
mkdir -p $OUTPUT_DIR


rosbag record -O $OUTPUT_DIR/anafi_data.bag \
    /anafi/attitude \
    /anafi/velocity_body \
    /anafi/flying_state \
    /anafi/battery_data \
    /anafi/gimbal_attitude \
    /anafi/gps_data \
    # /anafi/image_rect_color \
    /anafi/cmd/takeoff \
    /anafi/cmd/land \
    /anafi/cmd/set_position_relative

