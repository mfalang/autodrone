
import numpy as np
import matplotlib.pyplot as plt

import homography

class PoseRecovery():

    def __init__(self, K):
        self.K = K

    def find_homography(self, features_image, features_metric):
        # uv is image pixel location (origin = upper left)
        uv = features_image.copy()
        uv1 = homography.homogenize(uv)

        # xy is image coordinates with origin = center
        xy_homgen = np.linalg.solve(self.K, uv1)
        xy = homography.dehomogenize(xy_homgen)

        # XY are the metric coordinates in the helipad frame
        XY = features_metric[0:2].copy()

        H = homography.estimate_H_linear(xy, XY)

        return H

    def find_R_t(self, features_image, features_metric, H):
        XY01 = features_metric.copy()

        T1, T2 = homography.decompose_H(H)
        T = homography.choose_decomposition(T1, T2, XY01)

        R = T[:3, :3]
        t = T[:3, 3]

        return R, t

    def optimize_R_t(self, features_image, features_metric, R, t):

        # uv is image pixel location (origin = upper left)
        uv = features_image.copy()
        uv1 = homography.homogenize(uv)

        # xy is image coordinates with origin = center
        xy_homgen = np.linalg.solve(self.K, uv1)
        xy = homography.dehomogenize(xy_homgen)

        # XY are the metric coordinates in the helipad frame
        XY01 = features_metric.copy()

        R, t = homography.estimate_Rt_ls(
                    xy, XY01, R.copy(), t.copy(), self.K, uv, num_iterations=1000, debug=False)

        return R, t

    def get_pose_from_R_t(self, R, t):

        orientation = homography.rotation_matrix2euler_angles(R)*180/np.pi
        pos = t

        pose = np.hstack((pos, orientation))

        return pose

    def evaluate_reprojection(self, image, features_image, features_metric, H, R, t, R_LM, t_LM):

        # uv is image pixel location (origin = upper left)
        uv = features_image.copy()

        # XY are the metric coordinates in the helipad frame
        XY = features_metric[0:2].copy()
        XY1 = homography.homogenize(XY)
        XY01 = features_metric.copy()

        # For H
        uv_H = homography.reproject_using_H(self.K, H, XY1)
        err_H = homography.reprojection_error(uv, uv_H)
        print("Reprojection error using H:", err_H)

        # For R,t from DLT
        uv_Rt = homography.reproject_using_Rt(self.K, R, t, XY01)
        err_Rt = homography.reprojection_error(uv, uv_Rt)
        print("Reprojection error using R and t:", err_Rt)

        # For R,t from LM
        uv_Rt_LM = homography.reproject_using_Rt(self.K, R_LM, t_LM, XY01)
        err_Rt_LM = homography.reprojection_error(uv, uv_Rt_LM)
        print("Reprojection error using R and t from LM:", err_Rt_LM)

        plt.figure()
        plt.imshow(image)
        plt.scatter(*uv, s=150, marker='o', c='white', edgecolor='black', label="True")
        plt.scatter(*uv_H, s=30, marker='o', c='yellow',
                    edgecolors='black', label="Reprojected using H")
        plt.scatter(*uv_Rt, s=30, marker='o', c='cyan',
                    edgecolors='black', label="Reprojected using R and t")
        plt.scatter(*uv_Rt_LM, s=30, marker='o', c='red',
                    edgecolors='black', label="Reprojected using LM")

def main():
    pass

if __name__ == "__main__":
    main()