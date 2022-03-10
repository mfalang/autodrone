# This file is used to evaluate the different methods in attitude_reference_generator.py
# to determine which is most suited

from unittest import skip
import numpy as np

# Load velocity data to be used as reference and attitude data used for evaluation
data_folder = "2022-03-10/14-30-00"
vel_refs_filename = f"../../../../../out/{data_folder}/ground_truths/drone_velocity_body_frame.txt"
vel_refs = np.loadtxt(vel_refs_filename, skiprows=1)
att_refs_filename = f"../../../../../out/{data_folder}/ground_truths/helipad_pose_body_frame.txt"
att_refs = np.loadtxt(att_refs_filename, skiprows=1)

