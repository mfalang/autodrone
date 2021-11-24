
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

    def color_segment_image(self, img):

        # Make image grayscale
        output = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        return output

    def find_corners_shi_tomasi(self, img):
        corners = cv.goodFeaturesToTrack(img, self._max_corners, self._quality_level, self._min_distance, None, blockSize=self._block_size, gradientSize=self._gradient_size, useHarrisDetector=False, k=self._k)

        # corners = cv.goodFeaturesToTrack(img, self._max_corners, self._quality_level,
            # self._min_distance, None, blockSize=self._block_size,
            # gradientSize=self._gradient_size, useHarrisDetector=False, k=self._k
        # )

        return corners

    def show_corners_found(self, img, corners):
        image = np.copy(img)

        for i in range(corners.shape[0]):
            cv.circle(image, (int(corners[i,0,0]), int(corners[i,0,1])), 4, (0,0,255), cv.FILLED)

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

    img = cv.imread("test_images/test1.png")
    img_gray = corner_detector.color_segment_image(img)
    cv.imwrite("from_corner.png", img_gray)
    import sys
    sys.exit()
    cv.imshow("test", img_gray)
    # corners = corner_detector.find_corners_harris(img_gray)
    # corner_detector.show_corners_found(img, corners)
    corners = corner_detector.find_corners_shi_tomasi(img_gray)
    corner_detector.show_corners_found(img, corners)
    cv.waitKey()

if __name__ == "__main__":
    main()