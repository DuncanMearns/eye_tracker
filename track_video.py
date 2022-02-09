import numpy as np
import cv2
import math
import pickle


video_path = r"data/example_video.avi"
output_path = r"data/example_tracked.pickle"
cap = cv2.VideoCapture(video_path)
thresh = 100
show_tracking = False

tracking_output = []

while True:

    # Get frame
    ret, frame = cap.read()

    if ret:
        # Make grayscale and copy for drawing
        grayscale = frame[..., 0]
        show = frame.copy()

        # Threshold
        ret, threshed = cv2.threshold(grayscale, thresh, 255, cv2.THRESH_BINARY_INV)
        # Find contours
        contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour))
        contours.reverse()
        # Get largest contours
        contours = contours[:3]

        # Find contour centres
        centres = []
        for contour in contours:
            moments = cv2.moments(contour)
            c = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            centres.append(c)

        # Compute distances between contours
        distances = []
        for i in range(3):
            for j in range(i + 1, 3):
                c1 = centres[i]
                c2 = centres[j]
                x1, y1 = c1
                x2, y2 = c2
                d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distances.append(d)
        # Get swim bladder
        min_dist = min(distances)
        min_idx = distances.index(min_dist)
        sb_idx = 2 - min_idx
        sb = contours.pop(sb_idx)
        # Compute swim bladder centre
        moments = cv2.moments(sb)
        sb_c = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        # Draw sb centre
        cv2.circle(show, (int(sb_c[0]), int(sb_c[1])), 3, (255, 0, 0), -1)

        # Compute midpoint between eyes
        eye_centres = []
        for contour in contours:
            moments = cv2.moments(contour)
            c = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            eye_centres.append(c)
        eye_c_xs, eye_c_ys = zip(*eye_centres)
        mid_x = np.mean(eye_c_xs)
        mid_y = np.mean(eye_c_ys)
        mp = (mid_x, mid_y)
        # Draw midline
        cv2.line(show, (int(sb_c[0]), int(sb_c[1])), (int(mp[0]), int(mp[1])), (255, 0, 0), 1)

        # Compute orientation of fish
        dx = mp[0] - sb_c[0]
        dy = mp[1] - sb_c[1]
        fish_angle = math.atan2(dy, dx)  # POSITIVE ANGLES ARE CCW IN IMAGE
        if fish_angle < 0:
            fish_angle += (2 * math.pi)
        # Assign left and right eye
        if math.pi / 4 <= fish_angle < 3 * math.pi / 4:
            # DOWN
            left_i = eye_c_xs.index(max(eye_c_xs))
        elif 3 * math.pi / 4 <= fish_angle < 5 * math.pi / 4:
            # LEFT
            left_i = eye_c_ys.index(max(eye_c_ys))
        elif 5 * math.pi / 4 <= fish_angle < 7 * math.pi / 4:
            # UP
            left_i = eye_c_xs.index(min(eye_c_xs))
        else:
            # RIGHT
            left_i = eye_c_ys.index(min(eye_c_ys))
        # Get eye contours and centres
        eye_l = contours.pop(left_i)
        eye_l_c = eye_centres.pop(left_i)
        eye_r = contours.pop()
        eye_r_c = eye_centres.pop()
        # Draw centres
        cv2.circle(show, (int(eye_l_c[0]), int(eye_l_c[1])), 3, (0, 255, 0), -1)
        cv2.circle(show, (int(eye_r_c[0]), int(eye_r_c[1])), 3, (0, 0, 255), -1)

        # Compute eye angles
        eye_angles = []
        for contour in (eye_l, eye_r):
            moments = cv2.moments(contour)
            angle = 0.5 * np.arctan2(2 * moments["nu11"], (moments["nu20"] - moments["nu02"]))
            eye_angles.append(angle)
        left_angle, right_angle = eye_angles
        # Draw eye vectors
        l_vector = (20 * np.cos(left_angle), 20 * np.sin(left_angle))
        r_vector = (20 * np.cos(right_angle), 20 * np.sin(right_angle))
        cv2.line(show,
                 (int(eye_l_c[0] + l_vector[0]), int(eye_l_c[1] + l_vector[1])),
                 (int(eye_l_c[0] - l_vector[0]), int(eye_l_c[1] - l_vector[1])),
                 (0, 255, 0), 1)
        cv2.line(show,
                 (int(eye_r_c[0] + r_vector[0]), int(eye_r_c[1] + r_vector[1])),
                 (int(eye_r_c[0] - r_vector[0]), int(eye_r_c[1] - r_vector[1])),
                 (0, 0, 255), 1)

        # Show tracking
        if show_tracking:
            cv2.imshow("img", show)
            k = cv2.waitKey(1)
            if k in (27, 32, 13):
                break

        # Collect tracked data
        all_angles = (fish_angle, left_angle, right_angle)
        tracking_output.append(all_angles)

    else:
        break

with open(output_path, "wb") as f:
    pickle.dump(tracking_output, f)

print("TRACKING FINISHED")
