#!/usr/bin/env python3

from os import times
from subprocess import PIPE, Popen
from threading  import Thread
import sys
import numpy as np
import re
from queue import Queue, Empty

# Process the output from the file
def process_output(out, queue):
    
    next_pos = 0
    
    for line in iter(out.readline, b''):
        line = line.decode("utf-8")
        print(line)
        if next_pos and next_pos <= 3:
            number = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
            if "x" in line:
                pos1[0] = float(number)
            if "y" in line:
                pos1[1] = float(number)
            if "z" in line:
                pos1[2] = float(number)
            next_pos += 1
        elif "position" in line:
            next_pos = 1
        else:    
            next_pos = 0
    
        # if ".worldPosition" in line:
        #     number = re.findall(r"[-+]?\d*\.\d+|\d+", line)[1]
        #     if ".x" in line:
        #         pos1[0] = float(number)
        #     if ".y" in line:
        #         pos1[1] = float(number)
        #     if ".z" in line:
        #         pos1[2] = float(number)


        # if ".worldAttitude" in line:
        #     number = re.findall(r"[-+]?\d*\.\d+|\d+", line)[1]
        #     if ".x" in line:
        #         att1[0] = float(number)
        #     if ".y" in line:
        #         att1[1] = float(number)
        #     if ".z" in line:
        #         att1[2] = float(number)
                
        queue.put(line)
    out.close()

if __name__ == "__main__":
    pos1 = np.zeros(3)
    att1 = np.zeros(3)
    q = Queue()

    # Run the command
    ON_POSIX = 'posix' in sys.builtin_module_names
    command = "parrot-gz topic -e /gazebo/land/pose/info | grep -E -A 12 " \
        "'time" \
        "|name: \"helipad\"" \
        "|name: \"anafi4k\"'" 
    p = Popen(command, stdout=PIPE, bufsize=1, close_fds=ON_POSIX, shell=True)

    
    msg = {}
    next_pos = 0

    entry = {
        "timestamp": {
            "sec": None,
            "nsec": None
        }, 
        "anafi": {
            "position": {
                "x": None,
                "y": None,
                "z": None
            },
            "orientation": {
                "x": None,
                "y": None,
                "z": None,
                "w": None
            },
        },
        "helipad": {
            "position": {
                "x": None,
                "y": None,
                "z": None
            },
            "orientation": {
                "x": None,
                "y": None,
                "z": None,
                "w": None
            },
        }
    }
    
    timestamp_data_gather_done = False
    anafi_data_gather_done = False
    helipad_data_gather_done = False

    timestamp_index = 0
    anafi_index = 0
    helipad_index = 0

    for line in iter(p.stdout.readline, b''):
        line = line.decode("utf-8")

        # First find timestamp as this will be the first value
        if timestamp_index > 0 and timestamp_index <= 3:
            # Gather data
            if timestamp_index < 3:
                line = line.split()
                entry["timestamp"][line[0][:-1]] = int(line[1])
                timestamp_index += 1
            # Timestamp index is 3 and there is no more data to gather
            else:
                timestamp_index = 0
                timestamp_data_gather_done = True
        elif "time" in line:
            timestamp_index = 1

        # Make sure to only search for data after a timestamp is found to make
        # sure that the data belong to the correct timestamp
        if timestamp_data_gather_done == False:
            continue

        # Find Anafi data
        if anafi_index > 0 and anafi_index <= 12:
            # Skip over two first entries before position data
            if anafi_index < 3:
                anafi_index += 1
            # Parse anafi position
            elif anafi_index >= 3 and anafi_index <= 5:
                line = line.split()
                entry["anafi"]["position"][line[0][0]] = float(line[1])
                anafi_index += 1
            # Skip over two first entries before orientation data
            elif anafi_index < 8:
                anafi_index += 1
            # Parse anafi orientation
            elif anafi_index >= 8 and anafi_index <= 11:
                line = line.split()
                entry["anafi"]["orientation"][line[0][0]] = float(line[1])
                anafi_index += 1
            # Anafi index is 12 and there is no more data to gather
            else:
                anafi_index = 0
                anafi_data_gather_done = True
        elif "anafi" in line:
            anafi_index = 1
        
        # Find helipad data
        if helipad_index > 0 and helipad_index <= 12:
            # Skip over two first entries before position data
            if helipad_index < 3:
                helipad_index += 1
            # Parse helipad position
            elif helipad_index >= 3 and helipad_index <= 5:
                line = line.split()
                entry["helipad"]["position"][line[0][0]] = float(line[1])
                helipad_index += 1
            # Skip over two first entries before orientation data
            elif helipad_index < 8:
                helipad_index += 1
            # Parse helipad orientation
            elif helipad_index >= 8 and helipad_index <= 11:
                line = line.split()
                entry["helipad"]["orientation"][line[0][0]] = float(line[1])
                helipad_index += 1
            # helipad index is 12 and there is no more data to gather
            else:
                helipad_index = 0
                helipad_data_gather_done = True
        elif "helipad" in line:
            helipad_index = 1
        
        if timestamp_data_gather_done and anafi_data_gather_done and helipad_data_gather_done:
            print(entry)
            break

        # # Sample that works on only anafi data
        # if next_pos and next_pos <= 3:
        #     number = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
        #     msg[line[line.find(":")-1]] = float(number)
        #     next_pos += 1
        #     if next_pos == 4:
        #         print(msg)
        #         msg = {}
        #         next_pos = 0
        # elif "position" in line:
        #     next_pos = 1
    
    sys.exit()

    # Create a thread which dies with main program
    t = Thread(target=process_output, args=(p.stdout, q))
    t.daemon = True 
    t.start()

    for i in range(100):
        try:  
            line = q.get_nowait()
        except Empty:
            # Clear out the queue
            q.queue.clear()

        print("Position: {}, {}, {}".format(pos1[0], pos1[1], pos1[2]))
        print("Orientation: {}, {}, {}".format(att1[0], att1[1], att1[2]))

    print("System Exiting\n")
    sys.exit(0)