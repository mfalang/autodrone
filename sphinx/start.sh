#!/bin/bash

usage="Usage: $(basename "$0") <world (land/ocean)>"

declare -a worlds=("land" "ocean")

if [ $# -ne 1 ]
  then
    echo $usage
    exit
fi

world=$1

if [[ " ${worlds[*]} " =~ " $world " ]]; then
    script_dir=$(dirname "$(realpath $0)")
    sphinx $script_dir/worlds/$world.world
else
    echo $usage
fi
