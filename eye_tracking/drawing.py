import cv2
import numpy as np


def draw_circle(img, point, radius=3, color=(0, 0, 0), filled=True):
    """Draws a circle at the given point (modifies original image)"""
    x = int(point[0])
    y = int(point[1])
    if filled:
        thickness = -1
    else:
        thickness = 1
    cv2.circle(img, (x, y), radius, color, thickness)
    return img


def draw_line(img, p1, p2, color, thickness):
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    return img


def draw_eye(img, centre, angle, radius=3, length=20, color=(0, 0, 0), thickness=1):
    img = draw_circle(img, centre, radius, color)
    vector = np.array([length * np.cos(angle), length * np.sin(angle)])
    p1 = vector + centre
    p2 = -vector + centre
    img = draw_line(img, p1, p2, color, thickness)
    return img