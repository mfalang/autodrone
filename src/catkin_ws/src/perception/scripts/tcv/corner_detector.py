
import cv2 as cv
import numpy as np

class CornerDetector():

    def __init__(self, max_corners, quality_level, min_distance, block_size, gradient_size, k):
        self._max_corners = max_corners
        self._quality_level = quality_level
        self._min_distance = min_distance
        self._block_size = block_size
        self._gradient_size = gradient_size
        self._k = k

    def color_segment_image(self, img):

        # Make image grayscale
        output = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        return output

    def find_corners_shi_tomasi(self, img):
        corners = cv.goodFeaturesToTrack(img, self._max_corners, self._quality_level,
            self._min_distance, None, blockSize=self._block_size,
            gradientSize=self._gradient_size, useHarrisDetector=False, k=self._k
        )

        return corners

    def show_corners_found(self, img, corners):
        image = np.copy(img)

        for i in range(corners.shape[0]):
            cv.circle(image, (int(corners[i,0,0]), int(corners[i,0,1])), 4, (0,0,255), cv.FILLED)

        cv.imshow("Detected corners", image)
        cv.waitKey()

def main():
    max_corners = 13
    quality_level = 0.01
    min_distance = 10
    block_size = 3
    gradient_size = 3
    k = 0.04

    corner_detector = CornerDetector(
        max_corners,
        quality_level,
        min_distance,
        block_size,
        gradient_size,
        k
    )
    img = cv.imread("test_images/test1.png")
    img_gray = corner_detector.color_segment_image(img)
    # corners = corner_detector.find_corners_harris(img_gray)
    # corner_detector.show_corners_found(img, corners)
    corners = corner_detector.find_corners_shi_tomasi(img_gray)
    corner_detector.show_corners_found(img, corners)

if __name__ == "__main__":
    main()