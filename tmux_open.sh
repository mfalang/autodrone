#!/bin/bash

tmux new-session -d

tmux set -g mouse on

tmux send-keys "cd ~/code/autodrone" Enter

tmux split-window -v

tmux split-window -h

tmux select-pane -t 0

tmux split-window -h

tmux new-window -n "anafi"

tmux send-keys "source ~/code/autodrone/olympe/start.sh" Enter
tmux send-keys "roslaunch drone_interface anafi_"

tmux split-window -h

tmux send-keys "./sphinx/start.sh land"

tmux split-window -v

tmux select-pane -t 0

tmux split-window -v

tmux select-window -t 0

tmux attach-session
