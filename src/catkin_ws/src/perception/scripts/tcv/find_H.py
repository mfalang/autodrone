
import numpy as np

dists = np.loadtxt("test_corner_dists.txt")

# Index of the 4 largest distances
inds = np.unravel_index(np.argsort(-dists, axis=None)[:4], dists.shape)

arrow_idx = np.bincount(inds[0]).argmax()

H_corner_idns = inds[0][np.not_equal(inds[0], arrow_idx)]

assert np.array_equal(np.sort(inds[0]), np.sort(inds[1]))

pass