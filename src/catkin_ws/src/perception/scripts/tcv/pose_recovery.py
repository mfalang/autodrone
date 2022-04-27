
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

        R_ned, t_ned = self.camera_to_ned_frame(R, t)

        orientation = homography.rotation_matrix2euler_angles(R_ned)*180/np.pi
        psi_offset = -90
        orientation[2] = (orientation[2] - psi_offset + 180) % 360 - 180

        pos = t_ned

        pose = np.hstack((pos, orientation))

        # # Convert pose from camera frame with ENU coordinates, to NED coordinates

        # pose_ned = np.zeros_like(pose_camera)
        # pose_ned[0] = pose_camera[1]
        # pose_ned[1] = pose_camera[0]
        # pose_ned[2] = -pose_camera[2]
        # pose_ned[3] = pose_camera[3]
        # pose_ned[4] = pose_camera[4]
        # pose_ned[5] = pose_camera[5]

        # camera_offset_x_mm = 100
        # camera_offset_y_mm = 0
        # camera_offset_z_mm = 0

        # pose_ned[0] -= camera_offset_x_mm / 1000
        # pose_ned[1] -= camera_offset_y_mm / 1000
        # pose_ned[2] -= camera_offset_z_mm / 1000

        return pose

    def camera_to_ned_frame(self, R, t):

        # Rotate 90 deg (counter clockwise around z)
        rad = np.deg2rad(90)
        c = np.cos(rad)
        s = np.sin(rad)

        Rz = np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

        # Rotate 180 degrees around x-axis
        rad = np.deg2rad(180)
        c = np.cos(rad)
        s = np.sin(rad)

        Rx = np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])

        R_ned = R @ Rz @ Rx

        # t_ned = Rx @ Rz @ t

        t_ned = np.zeros_like(t)
        t_ned[0] = t[1]
        t_ned[1] = -t[0]
        t_ned[2] = -t[2]

        # t_ned = t

        return R_ned, t_ned

    def evaluate_reprojection(self, image, features_image, features_metric, H, R, t, R_LM, t_LM):


        angles = homography.rotation_matrix2euler_angles(R_LM)*180/np.pi

        # print(f"Pos: {t_LM} orientation: {angles}")

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
        print(f"Reprojection error using R and t: {err_Rt}, average: {np.average(err_Rt)}")

        # For R,t from LM
        uv_Rt_LM = homography.reproject_using_Rt(self.K, R_LM, t_LM, XY01)
        err_Rt_LM = homography.reprojection_error(uv, uv_Rt_LM)
        print(f"Reprojection error using R and t from LM: {err_Rt_LM}, average: {np.average(err_Rt_LM)}")

        # R_LM_ned, t_LM_ned = self.camera_to_ned_frame(R_LM, t_LM)

        T_LM = homography.create_T_from_Rt(R_LM, t_LM)

        plt.figure()
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.scatter(*uv, s=150, marker='o', c='white', edgecolor='black', label="True")
        # plt.scatter(*uv_H, s=30, marker='o', c='yellow',
        #             edgecolors='black', label="Reprojected using H")
        plt.scatter(*uv_Rt, s=30, marker='o', c='cyan',
                    edgecolors='black', label="Reprojected using R and t")
        plt.scatter(*uv_Rt_LM, s=30, marker='o', c='red',
                    edgecolors='black', label="Reprojected using LM")

        plt.title("Reprojections of pose with and without optimization")

        homography.draw_frame(self.K, T_LM, scale=0.1)
        plt.legend()

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(XY[0,:], XY[1,:], np.zeros(XY.shape[1]), '.', markersize=10) # Draw markers in 3D
        pO = np.linalg.inv(T_LM)@np.array([0,0,0,1]) # Compute camera origin
        pX = np.linalg.inv(T_LM)@np.array([0.4,0,0,1]) # Compute camera X-axis
        pY = np.linalg.inv(T_LM)@np.array([0,0.4,0,1]) # Compute camera Y-axis
        pZ = np.linalg.inv(T_LM)@np.array([0,0,0.4,1]) # Compute camera Z-axis
        plt.plot([pO[0], pZ[0]], [pO[1], pZ[1]], [pO[2], pZ[2]], color='blue', linewidth=2) # Draw camera Z-axis
        plt.plot([pO[0], pY[0]], [pO[1], pY[1]], [pO[2], pY[2]], color='green', linewidth=2) # Draw camera Y-axis
        plt.plot([pO[0], pX[0]], [pO[1], pX[1]], [pO[2], pX[2]], color='red', linewidth=2) # Draw camera X-axis
        # plt.plot([0, 0], [0, 0], [0, 0.4], color='blue', linewidth=2) # Draw camera Z-axis
        # plt.plot([0, 0], [0, 0.4], [0, 0], color='green', linewidth=2) # Draw camera Y-axis
        # plt.plot([0, 0.4], [0, 0], [0, 0], color='red', linewidth=2) # Draw camera X-axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-0.5, 1.5])
        ax.set_xlabel('X')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')

        plt.title("Camera pose relative to detected features")
        # plt.show()

def main():
    # Test reprojection using
    import os
    import sys
    import yaml
    import time
    import feature_detector

    shi_tomasi_config = {
        "max_corners" : 13,
        "quality_level" : 0.0001,
        "min_distance" : 1,
        "block_size" : 7,
        "gradient_size" : 17,
        "k" : 0.04,
        "use_harris_detector": True
    }

    hough_circle_config = {
        "bilateral_diameter": 9,
        "dp": 1,
        "gaussian_kernel": 5,
        "max_radius": 500,
        "median_kernel": 11,
        "method": 3,
        "min_dist": 1000,
        "min_radius": 50,
        "param1": 40,
        "param2": 70,
        "use_bilateral_blur": True,
        "use_gaussian_blur": True,
        "use_median_blur": True
    }

    detector = feature_detector.FeatureDetector(shi_tomasi_config, hough_circle_config)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        with open(f"{script_dir}/../../config/tcv_config.yaml") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit()

    features_metric = np.loadtxt(
            f"{script_dir}/../../{config['feature_dists_metric']['path']}"
    )
    # features_metric = np.loadtxt("../../data/helipad_dists_origin_center_enu_metric.txt")

    K = np.loadtxt(
            f"{script_dir}/../../{config['camera_matrix']['path']}"
        )

    pose_recoverer = PoseRecovery(K)

    img = cv.imread("test_images/test4.png")
    mask = detector.create_helipad_mask(img, show_masked_img=True)
    # img_processed = detector.preprocess_image(img)

    corners = detector.find_corners_shi_tomasi(img, mask)
    # detector.show_corners_found(img, corners, color="red")

    features_image = detector.find_arrow_and_H(corners, features_metric)

    detector.show_known_points(img, features_image)
    cv.waitKey()

    H = pose_recoverer.find_homography(features_image, features_metric)
    start = time.time()
    R, t = pose_recoverer.find_R_t(features_image, features_metric, H)
    print(f"Recovering R and t took {time.time() - start} seconds")
    start = time.time()
    R_LM, t_LM = pose_recoverer.optimize_R_t(features_image, features_metric, R, t)
    print(f"Optmizing R and t took {time.time() - start} seconds")

    pose_recoverer.evaluate_reprojection(img, features_image, features_metric,
        H, R, t, R_LM, t_LM
    )

    R_LM_ned, t_LM_ned = pose_recoverer.camera_to_ned_frame(R_LM, t_LM)

    pose = pose_recoverer.get_pose_from_R_t(R_LM_ned, t_LM_ned)

    print(f"NED pose: Pos: {pose[:3]} Orientation: {pose[3:]}")

    plt.show()

if __name__ == "__main__":
    main()