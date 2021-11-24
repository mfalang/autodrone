
import enum
import numpy as np

dists = np.loadtxt("test_corner_dists.txt")

matching_lengths = []

for i, dists_i in enumerate(dists):
    matching_lengths_i = []
    for j, dist_j in enumerate(dists_i):
        if i != j:
            matches = np.array(np.where(np.isclose(dists, dist_j, atol=1))).T
            matching_lengths_i.append(matches)
    matching_lengths.append(matching_lengths_i)

pass