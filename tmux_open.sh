#!/bin/bash

tmux new-session -d

tmux set -g mouse on

tmux send-keys "cd ~/code/autodrone" Enter

tmux split-window -v

tmux send-keys "roscd utilities" Enter
tmux send-keys "roslaunch output_saver "

tmux split-window -h

tmux send-keys "roslaunch control lab_test.launch"

tmux select-pane -t 0

tmux send-keys "roslaunch perception ekf_"

tmux split-window -h

tmux send-keys "roslaunch perception dnnCV_"

tmux new-window -n "anafi"

tmux send-keys "source ~/code/autodrone/olympe/start.sh" Enter
tmux send-keys "roslaunch drone_interface anafi_"

tmux split-window -h

tmux send-keys "./sphinx/start.sh land"

tmux select-window -t 0

tmux attach-session
