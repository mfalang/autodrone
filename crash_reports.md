# Crash reports

## 01.04.2022 - 09.31.41 - NTNU Drone Lab

### Code run
- attitude_reference_evaluator using the linear drag model
- darknet_ros
- perception dnn_cv
- perception ekf
- output_saver
- ground_truth coordinate_transform

### Behaviour
Drone seemed to behave as expected in the beginning, but then after a few seconds started going
backwards without stopping. Cancelled the mission by forcing the drone to land with the land-button
on the remote controller, but due to the horizontal velocity of the drone it hit the bottom of the
net in the drone lab. Motors turned off fast.

### Damage
As the motors turned off very fast, there seems to be only minor dents in a few of the
propellers.

### Causes
Unknown. Could be faulty velocity measurements, as this has caused a similar response in the
simulator.

### Takeaways
- Check that the velocity measurements look ok using "rostopic echo" while the drone is hovering
before starting any mission
- Manually land the drone faster if there is any sign of misbehaviour