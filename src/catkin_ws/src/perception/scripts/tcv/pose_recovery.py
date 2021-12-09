
import cv2 as cv
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
                    xy, XY01, R.copy(), t.copy(), self.K, uv, num_iterations=100, debug=False)

        return R, t

    def get_pose_from_R_t(self, R, t):

        orientation = homography.rotation_matrix2euler_angles(R)*180/np.pi
        pos = t

        pose = np.hstack((pos, orientation))

        return pose

    def evaluate_reprojection(self, image, features_image, features_metric, H, R, t, R_LM, t_LM):

        T_LM = homography.create_T_from_Rt(R_LM, t_LM)

        angles = homography.rotation_matrix2euler_angles(R_LM)*180/np.pi

        print(f"Pos: {t_LM} orientation: {angles}")

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
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.scatter(*uv, s=150, marker='o', c='white', edgecolor='black', label="True")
        plt.scatter(*uv_H, s=30, marker='o', c='yellow',
                    edgecolors='black', label="Reprojected using H")
        plt.scatter(*uv_Rt, s=30, marker='o', c='cyan',
                    edgecolors='black', label="Reprojected using R and t")
        plt.scatter(*uv_Rt_LM, s=30, marker='o', c='red',
                    edgecolors='black', label="Reprojected using LM")

        homography.draw_frame(self.K, T_LM, scale=0.1)

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(XY[0,:], XY[1,:], np.zeros(XY.shape[1]), '.') # Draw markers in 3D
        pO = np.linalg.inv(T_LM)@np.array([0,0,0,1]) # Compute camera origin
        pX = np.linalg.inv(T_LM)@np.array([0.1,0,0,1]) # Compute camera X-axis
        pY = np.linalg.inv(T_LM)@np.array([0,0.1,0,1]) # Compute camera Y-axis
        pZ = np.linalg.inv(T_LM)@np.array([0,0,0.1,1]) # Compute camera Z-axis
        plt.plot([pO[0], pZ[0]], [pO[1], pZ[1]], [pO[2], pZ[2]], color='blue') # Draw camera Z-axis
        plt.plot([pO[0], pY[0]], [pO[1], pY[1]], [pO[2], pY[2]], color='green') # Draw camera Y-axis
        plt.plot([pO[0], pX[0]], [pO[1], pX[1]], [pO[2], pX[2]], color='red') # Draw camera X-axis
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([0, 2])
        ax.set_xlabel('X')
        ax.set_zlabel('Y')
        ax.set_ylabel('Z')

        plt.legend()
        plt.show()

def main():
    # Test reprojection using
    import corner_detector

    detector_config = {
        "max_corners" : 13,
        "quality_level" : 0.001,
        "min_distance" : 10,
        "block_size" : 3,
        "gradient_size" : 3,
        "k" : 0.04
    }
    detector = corner_detector.CornerDetector(detector_config)
    features_metric = np.loadtxt("../../data/helipad_dists_origin_center_enu_metric.txt")

    K = np.array([
            [941.22, 0, 580.66],
            [0, 932.66, 375.35],
            [0, 0, 1]
    ])
    pose_recoverer = PoseRecovery(K)

    img = cv.imread("test_images/test1.png")

    img_processed = detector.preprocess_image(img)

    corners = detector.find_corners_shi_tomasi(img_processed)

    features_image = detector.find_arrow_and_H(corners, features_metric)

    H = pose_recoverer.find_homography(features_image, features_metric)
    R, t = pose_recoverer.find_R_t(features_image, features_metric, H)
    R_LM, t_LM = pose_recoverer.optimize_R_t(features_image, features_metric, R, t)

    pose_recoverer.evaluate_reprojection(img, features_image, features_metric,
        H, R, t, R_LM, t_LM
    )

    pose = pose_recoverer.get_pose_from_R_t(R_LM, t_LM)

if __name__ == "__main__":
    main()