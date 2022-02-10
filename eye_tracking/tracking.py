import cv2
import numpy as np
from scipy.spatial.distance import pdist


def find_n_contours(img, threshold, n, invert=False):
    """Finds the n largest contours in an image. Default assumes fish is dark against light."""
    # Threshold
    if invert:
        ret, threshed = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    else:
        ret, threshed = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour))
    contours.reverse()
    # Get largest contours
    contours = contours[:n]
    return contours


def contour_info(contour):
    """Computes centroid and angles of a contour using image moments"""
    moments = cv2.moments(contour)
    x, y = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    angle = 0.5 * np.arctan2(2 * moments["nu11"], (moments["nu20"] - moments["nu02"]))
    return x, y, angle


def assign_features(contour_centres):
    """Returns indicies of sb, left, right from an array of contour centres"""
    contour_centres = np.array(contour_centres)
    assert contour_centres.shape == (3, 2)
    distances = pdist(contour_centres)
    sb_idx = 2 - np.argmin(distances)
    eye_idxs = [i for i in range(3) if i != sb_idx]
    eye_vectors = contour_centres[eye_idxs] - contour_centres[sb_idx]
    cross_product = np.cross(*eye_vectors)
    if cross_product < 0:
        eye_idxs = eye_idxs[::-1]
    left_idx, right_idx = eye_idxs
    return sb_idx, left_idx, right_idx


def track_frame(frame, threshold):
    """Takes a grayscale frame and return centre and angle of swim bladder, left, and right eyes"""
    assert frame.ndim == 2
    contours = find_n_contours(frame, threshold, 3)
    contour_properties = [contour_info(contour) for contour in contours]
    contour_properties = np.array(contour_properties)
    sb, left, right = assign_features(contour_properties[:, :2])
    contour_properties = contour_properties[[sb, left, right]]
    return contour_properties
