
import cv2 as cv
import numpy as np

HSV_SIM_GREEN = [120, 100, 30]

HUE_MARGIN = 15
SAT_MARGIN = 15
VAL_MARGIN = 15

HUE_LOW_GREEN = max(0, HSV_SIM_GREEN[0] - HUE_MARGIN)
HUE_HIGH_GREEN = min(360, HSV_SIM_GREEN[0] + HUE_MARGIN)

SAT_LOW_GREEN = 85
SAT_HIGH_GREEN = 100

VAL_LOW_GREEN = 15
VAL_HIGH_GREEN = 60

class CornerDetector():

    def __init__(self, shi_tomasi_config):
        self._max_corners = shi_tomasi_config["max_corners"]
        self._quality_level = shi_tomasi_config["quality_level"]
        self._min_distance = shi_tomasi_config["min_distance"]
        self._block_size = shi_tomasi_config["block_size"]
        self._gradient_size = shi_tomasi_config["gradient_size"]
        self._k = shi_tomasi_config["k"]

        self.fast_feature_dector = cv.FastFeatureDetector_create()

    def preprocess_image(self, img):
        # Make image grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        blur = cv.medianBlur(img, 5)

        blur = cv.GaussianBlur(blur,(5,5),0)
        blur = cv.bilateralFilter(blur,9,75,75)

        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

        # low_green = np.array([HUE_LOW_GREEN, SAT_LOW_GREEN, VAL_LOW_GREEN])
        # high_green = np.array([HUE_HIGH_GREEN, SAT_HIGH_GREEN, VAL_HIGH_GREEN])

        hue_margin = 60
        sat_margin = 60
        val_margin = 60

        low_green = np.array([85-hue_margin,255-sat_margin,127-val_margin])
        high_green = np.array([85+hue_margin,255+sat_margin,127+val_margin])

        mask = cv.inRange(hsv, low_green, high_green)

        img_segmented = cv.bitwise_and(gray, gray, mask=mask)
        img_segmented = cv.medianBlur(img_segmented, 5)
        img_segmented = cv.GaussianBlur(img_segmented,(5,5),0)
        img_segmented = cv.bilateralFilter(img_segmented,9,75,75)

        # cv.imshow("mask", mask)
        cv.imshow("segmented", img_segmented)

        # # Find circle in helipad
        # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, gray.shape[0] / 8,
        #                        param1=200, param2=50,
        #                        minRadius=0, maxRadius=0)

        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         center = (i[0], i[1])
        #         # circle center
        #         cv.circle(img, center, 1, (0, 100, 100), 3)
        #         # circle outline
        #         radius = i[2]
        #         cv.circle(img, center, radius, (255, 0, 255), 3)

        # cv.imshow("Circles", img)
        output = gray

        return output

    def find_corners_fast(self, img):
        key_points = self.fast_feature_dector.detect(img, None)

        corners = np.array([key_points[idx].pt for idx in range(0, len(key_points))]).reshape(-1, 1, 2)

        return corners

    def find_corners_shi_tomasi(self, img):

        corners = cv.goodFeaturesToTrack(img, self._max_corners, self._quality_level,
            self._min_distance, None, blockSize=self._block_size,
            gradientSize=self._gradient_size, useHarrisDetector=False, k=self._k
        )

        return corners

    def find_H(self, img, corners):

        if corners.shape[0] != 13:
            print(f"Not enough points to determine H uniquely ({corners.shape[0]}/13)")
            return None

        dists = np.zeros((13,13))

        for i, xy_i in enumerate(corners.reshape(corners.shape[0], 2)):
            print(f"{i} distances: ", end="")
            for j, xy_j in enumerate(corners.reshape(corners.shape[0], 2)):
                print(f"{j}: {cv.norm(xy_i, xy_j)} ", end="")
                dists[i,j] = cv.norm(xy_i, xy_j)
            print()

        np.savetxt("test_corner_dists.txt", dists)

        # for i in range(corners.shape[0]):
        #     current_corner = i
        #     for j in range(corner.shape[0]):
        #         dist = cv.norm(current_corner, )

    def show_corners_found(self, img, corners, color):
        image = np.copy(img)

        if color == "red":
            c = (0,0,255)
        elif color == "blue":
            c = (255,0,0)

        for i in range(corners.shape[0]):
            center = (int(corners[i,0,0]), int(corners[i,0,1]))

            text_face = cv.FONT_HERSHEY_DUPLEX
            text_scale = 0.5
            text_thickness = 1
            text = f"{i}"
            text_offset = 10

            text_size, _ = cv.getTextSize(text, text_face, text_scale, text_thickness)
            text_origin = (
                int(center[0] - text_size[0] / 2) + text_offset,
                int(center[1] + text_size[1] / 2) - text_offset
            )

            cv.circle(image, center, 4, c, cv.FILLED)
            cv.putText(image, text, text_origin, text_face, text_scale, (127,255,127), text_thickness, cv.LINE_AA)

        cv.imshow("Detected corners", image)

def main():
    config = {
        "max_corners" : 13,
        "quality_level" : 0.001,
        "min_distance" : 10,
        "block_size" : 3,
        "gradient_size" : 3,
        "k" : 0.04
    }

    corner_detector = CornerDetector(config)

    img = cv.imread("test_images/half_circle_5.png")
    img_gray = corner_detector.preprocess_image(img)
    cv.imshow("test", img_gray)
    # corners = corner_detector.find_corners_harris(img_gray)
    # corner_detector.show_corners_found(img, corners)
    corners = corner_detector.find_corners_shi_tomasi(img_gray)
    corner_detector.show_corners_found(img, corners, color="red")
    cv.waitKey()

if __name__ == "__main__":
    main()