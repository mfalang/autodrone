# A script to get the best parameters for the Shi-Tomasi corner detector

import os
import glob
import yaml
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial import distance

import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class CornerDetector(sklearn.base.BaseEstimator):

    # total number of elements in (720, 1280, 3) image after flattening
    IMAGE_NUM_ELEMENTS = 2764800

    def __init__(self, quality_level=0.01, min_distance=10, block_size=9, gradient_size=3,
        use_harris_detector=False, k=0.04, max_corners=13
    ):
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.gradient_size = gradient_size
        self.use_harris_detector = use_harris_detector
        self.k = k
        self.max_corners = max_corners

    def set_params(self, **params):
        return super().set_params(**params)

    def get_params(self, deep=True):
        return super().get_params(deep)

    def predict(self, X):

        y_pred = []

        for X_i in X:

            img = X_i[:self.IMAGE_NUM_ELEMENTS].reshape((720, 1280, 3))
            mask = X_i[self.IMAGE_NUM_ELEMENTS:].reshape((720, 1280))

            img_gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

            corners = cv.goodFeaturesToTrack(img_gray, self.max_corners, self.quality_level,
                self.min_distance, mask=mask, blockSize=self.block_size,
                gradientSize=self.gradient_size, useHarrisDetector=self.use_harris_detector, k=self.k
            )

            if corners is not None and corners.shape[0] == self.max_corners:
                y_pred_i = corners.reshape(corners.shape[0], 2)
            else:
                y_pred_i = np.zeros((13, 2))

            y_pred.append(y_pred_i)

        y_pred = np.array(y_pred)

        return y_pred

    def fit(self, X, y):
        return self

def prediction_error(y, y_pred):
    errors_all_images = []
    for y_pred_i, y_i in zip(y_pred, y):

        pred = y_pred_i.reshape((13,2))

        gt = y_i.reshape((13,2))

        classes = np.arange(len(pred))

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(pred, y=classes)
        gt_ind = knn.predict(gt)

        gt_sorted = gt[np.argsort(gt_ind)]
        pred_sorted = pred

        errors = np.diagonal(distance.cdist(pred_sorted, gt_sorted))

        mean_error = np.mean(errors)
        errors_all_images.append(mean_error)

    mean_error_all_images = np.mean(errors_all_images)

    return mean_error_all_images

def blur_image_gaussian(img):
    blur = cv.GaussianBlur(img.copy(),(5,5),0)
    return blur

def blur_image(img):
    blur = cv.medianBlur(img.copy(), 11)
    blur = cv.GaussianBlur(blur,(5,5),0)
    blur = cv.bilateralFilter(blur,9,75,75)

    return blur

def hough_circle(img, use_matplotlib=True):

    blurred_img = blur_image(img)

    gray = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 1000,
                        param1=50, param2=50,
                        minRadius=10, maxRadius=700)
    output = img.copy()
    if circles is not None and len(circles) == 1:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # Create circle mask
        (x, y, r) = circles[0]
        circle_mask = np.zeros((720,1280), np.uint8)
        cv.circle(circle_mask, (x,y), int(r * 1.30), (255, 0, 255), cv.FILLED)
        img_masked = cv.bitwise_and(img, img, mask=circle_mask)

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(output, (x, y), r, (0, 255, 0), 4)
            cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        return img_masked, circle_mask, (x,y,r)
    else:
        return None

def get_circle_mask(img):
    img = blur_image_gaussian(img)
    _, circle_mask, _ = hough_circle(img)

    return circle_mask

def create_XY():
    images = [(cv.imread(file), file) for file in sorted(glob.glob("../test_images/real/*.jpg"))][:9]
    del images[3] # remove image that has not been cropped properly

    y_df = pd.read_csv("../test_images/real/corner_labels.csv", index_col=0)

    X = []
    y = []

    for img, filename in images:
        frame_id = os.path.basename(filename)
        if frame_id in y_df.index:
            y_i = y_df.loc[frame_id, :].to_numpy().reshape((13,2))
            y.append(y_i.flatten())
            mask = get_circle_mask(img)
            X.append(np.hstack((img.flatten(), mask.flatten())))

    X = np.array(X)
    y = np.array(y)

    return X, y

X, y = create_XY()

param_grid = [{"quality_level": [0.001, 0.01, 0.1, 1],
            "min_distance": [1, 5, 10],
            "block_size": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            "gradient_size": [1, 3, 5, 7, 9, 11, 13, 17, 19],
            "use_harris_detector": [False],
            "k": [0.04],
            "max_corners": [13]}]

scorer_function = make_scorer(prediction_error, greater_is_better=False)
grid = GridSearchCV(CornerDetector(), param_grid, scoring=scorer_function, verbose=51, n_jobs=10)
grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)
results.to_csv("results/corner_params_grid_search_results.csv")
with open("results/corner_params.yaml", "w+") as f:
    yaml.dump(grid.best_params_, f, default_flow_style=False)
print(f"Best parameters: {grid.best_params_}")
print(f"Score: {grid.best_score_}")
print(f"Best index: {grid.best_index_}")
