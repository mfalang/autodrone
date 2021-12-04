
import numpy as np

import homography

class PoseRecovery():

    def __init__(self, K):
        self.K = K

    def recover_pose(self, features_image, features_metric):

        # uv is image pixel location (origin = upper left)
        uv = features_image.copy()
        uv1 = homography.homogenize(uv)

        # xy is image coordinates with origin = center
        xy_homgen = np.linalg.solve(self.K, uv1)
        xy1 = homography.homogeneous_normalize(xy_homgen)
        xy = homography.dehomogenize(xy_homgen)

        # XY are the metric coordinates in the helipad frame
        XY = features_metric[0:2].copy()
        XY1 = homography.homogenize(XY)
        XY01 = features_metric.copy()

        H = homography.estimate_H_linear(xy, XY)

        T1, T2 = homography.decompose_H(H)
        T = homography.choose_decomposition(T1, T2, XY01)

        R = T[:3, :3]
        t = T[:3, 3]

        uv_Rt = homography.reproject_using_Rt(self.K, R, t, XY01)

        R_LM, t_LM = homography.estimate_Rt_ls(
            xy, XY01, R, t, self.K, uv, num_iterations=1000, debug=False)
        T_LM = homography.create_T_from_Rt(R_LM, t_LM)
        # print(f"T_LM:\n{T_LM}")
        angles = homography.rotation_matrix2euler_angles(R_LM)
        print(f"Roll {angles[0]*180/np.pi} (in deg)")
        print(f"Pitch {angles[1]*180/np.pi} (in deg)")
        print(f"Yaw {angles[2]*180/np.pi} (in deg)")



def main():
    pass

if __name__ == "__main__":
    main()