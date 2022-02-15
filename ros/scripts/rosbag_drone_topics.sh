#!/bin/bash

usage="Usage: $(basename "$0") <test-name> <env (sim/real)>"

if [ $# -ne 2 ]
  then
    echo $usage
    exit
fi

TEST_NAME=$1
ENV=$2
SCRIPT_DIR=$(dirname "$(realpath $0)")


OUTPUT_DIR=$SCRIPT_DIR/../../out/rosbag/$ENV/$TEST_NAME
mkdir -p $OUTPUT_DIR

if [ -e $OUTPUT_DIR/*.bag ]
then
    OLD_DIR=$OUTPUT_DIR/old
    echo "Moving old bagfile into "$OLD_DIR""
    mkdir -p $OLD_DIR
    mv $OUTPUT_DIR/*.bag $OLD_DIR
fi

TIME=$(date +%Y-%m-%d-%H-%M-%S)

rosbag record -O $OUTPUT_DIR/$TIME \
    /drone/out/telemetry \
    /drone/out/image_rect_color \
    /drone/out/gps \
    /ground_truth/helipad_frame/drone_pose \
    /ground_truth/body_frame/helipad_pose \
    /ground_truth/ned_frame/drone_pose \
    /ground_truth/ned_frame/helipad_pose \
    /darknet_ros/bounding_boxes \
