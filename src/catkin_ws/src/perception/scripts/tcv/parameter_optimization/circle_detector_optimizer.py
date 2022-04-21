# A script to get the best parameters for the Hough circle detector

import os
import glob
import yaml
import cv2 as cv
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

class CircleDetector(sklearn.base.BaseEstimator):

    def __init__(self, method=cv.HOUGH_GRADIENT, dp=1, min_dist=1000, param1=50,
        param2=50, min_radius=10, max_radius=700
    ):

        self.method = method
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def set_params(self, **params):
        return super().set_params(**params)

    def get_params(self, deep=True):
        return super().get_params(deep)

    def predict(self, X):

        y_pred = []

        for X_i in X:
            # TODO: Make this blurring dependent on learnable parameters
            img = X_i.reshape((720, 1280, 3))
            blurred_img = blur_image(img)
            img_gray = cv.cvtColor(blurred_img.copy(), cv.COLOR_BGR2GRAY)

            circles = cv.HoughCircles(img_gray, method=self.method, dp=self.dp,
                minDist=self.min_dist, param1=self.param1, param2=self.param2,
                minRadius=self.min_radius, maxRadius=self.max_radius
            )

            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            if circles is not None and len(circles) == 1:
                y_pred_i = np.array(circles[0])
            else:
                y_pred_i = np.zeros(3)

            y_pred.append(y_pred_i)

        y_pred = np.array(y_pred)

        return y_pred

    def fit(self, X, y):
        return self

    def prediction_error(self, y, y_pred, verbose=False):

        errors_all_images = []
        for y_pred_i, y_i in zip(y_pred, y):

            error_origin = np.linalg.norm(y_pred_i[:2] - y_i[:2])
            error_radius = np.abs(y_pred_i[2] - y_i[2])
            total_error = error_origin + error_radius
            errors_all_images.append(total_error)

            if verbose:
                print(f"Error origin: \t\t{error_origin:.2f} \t(pred: {(y_pred_i[0], y_pred_i[1])} gt: {(y_i[0], y_i[1])})")
                print(f"Error radius (abs): \t{error_radius:.2f} \t(pred: {y_pred_i[2]} gt: {y_i[2]})")

        mean_error_all_images = np.mean(errors_all_images)

        if verbose:
            print(f"Mean error all images: {mean_error_all_images}")
        return mean_error_all_images

def blur_image(img):
    blur = cv.medianBlur(img.copy(), 11)
    blur = cv.GaussianBlur(blur,(5,5),0)
    blur = cv.bilateralFilter(blur,9,75,75)

    return blur


def create_XY():
    images = [(cv.imread(file), file) for file in sorted(glob.glob("../test_images/real/*.jpg"))][:9]

    y_df = pd.read_csv("../test_images/real/circle_labels.csv", index_col=0)

    X = []
    y = []

    for img, filename in images:
        frame_id = os.path.basename(filename)
        if frame_id in y_df.index:
            y_i = y_df.loc[frame_id, :].to_numpy()
            y.append(y_i.flatten())
            X.append(img.flatten())

    X = np.array(X)
    y = np.array(y)

    return X, y

X, y = create_XY()

# param_grid = [{"method": [cv.HOUGH_GRADIENT, cv.HOUGH_STANDARD, cv.HOUGH_GRADIENT_ALT, cv.HOUGH_PROBABILISTIC, cv.HOUGH_MULTI_SCALE],
#             "dp": [0.5, 1, 2],
#             "min_dist": [10, 50, 100, 300, 500, 1000],
#             "param1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#             "param2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#             "min_radius": [5, 10, 20, 50, 100],
#             "max_radius": [100, 500, 700, 1000]}]

param_grid = [{"method": [cv.HOUGH_GRADIENT],
            "dp": [1],
            "min_dist": [1000],
            "param1": [30, 40, 50, 60, 70, 80],
            "param2": [30, 40, 50, 60, 70, 80],
            "min_radius": [10],
            "max_radius": [500, 700, 1000]}]

scorer_function = make_scorer(CircleDetector().prediction_error, greater_is_better=False)
grid = GridSearchCV(CircleDetector(), param_grid, scoring=scorer_function, verbose=10, n_jobs=10)
grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)
results.to_csv("results/circle_params_grid_search_results.csv")
with open("results/circle_params.yaml", "w+") as f:
    yaml.dump(grid.best_params_, f)
print(f"Best parameters: {grid.best_params_}")
print(f"Score: {grid.best_score_}")
print(f"Best index: {grid.best_index_}")