#!/bin/bash

# Simple script for running a Python script inside a Docker container.
# Currently everything is hard coded and this script is only here to
# avoid having to remember the arguments for docker run.
docker run -it --network host -v $(pwd)/src:/app olympe_env python simple_test.py
