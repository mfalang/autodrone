
import cv2 as cv
import numpy as np

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

        # Reduce the noise to avoid false circle detection
        gray = cv.medianBlur(gray, 5)

        # Find circle in helipad
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, gray.shape[0] / 8,
                               param1=100, param2=200,
                               minRadius=0, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(img, center, radius, (255, 0, 255), 3)

        cv.imshow("Circles", img)
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

    def show_corners_found(self, img, corners, color):
        image = np.copy(img)

        if color == "red":
            c = (0,0,255)
        elif color == "blue":
            c = (255,0,0)

        for i in range(corners.shape[0]):
            cv.circle(image, (int(corners[i,0,0]), int(corners[i,0,1])), 4, c, cv.FILLED)

        cv.imshow("Detected corners", image)

def main():
    config = {
        "max_corners" : 13,
        "quality_level" : 0.01,
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