import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def compute_robot_transformation_matrix(a, b):
    print("a\n", a)
    print("b\n", b)
    lr = LinearRegression(fit_intercept=False).fit(a, b)
    return lr.coef_.T


def rgb_to_robot_coords(rgb_coords, transmatrix):
    assert len(rgb_coords.shape) <= 2
    if len(rgb_coords.shape) == 1:
        rgb_coords = np.array(rgb_coords[None])

    poly = PolynomialFeatures(2)
    rgb_coords = poly.fit_transform(rgb_coords)

    if transmatrix is not None:
        robot_coords = rgb_coords @ transmatrix
        return np.squeeze(robot_coords)


def compute_transform(robot_coords, rgb_coords):
    poly = PolynomialFeatures(2)
    temp = poly.fit_transform(rgb_coords)
    matrix = compute_robot_transformation_matrix(np.array(temp), robot_coords)
    print("matrix", matrix)
    residuals = (
        rgb_to_robot_coords(np.array(rgb_coords), matrix) - robot_coords)
    residuals = [np.linalg.norm(i) for i in residuals]
    print("residuals\n", residuals)
    return matrix
