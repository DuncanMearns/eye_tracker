import cv2
from eye_tracking.drawing import draw_circle, draw_eye
from eye_tracking.tracking import track_frame
import numpy as np


colormap = {"sb": (255, 0, 0),
            "left": (0, 255, 0),
            "right": (0, 0, 255)}


def draw_tracking(img, tracking):
    sb, left, right = tracking
    # Draw stuff
    draw_circle(img, sb[:2], color=colormap["sb"])
    draw_eye(img, left[:2], left[2], color=colormap["left"])
    draw_eye(img, right[:2], right[2], color=colormap["right"])
    return img


def track_video(path, threshold=100, show_tracking=False):
    cap = cv2.VideoCapture(path)
    tracking_output = []
    while True:
        # Get frame
        ret, frame = cap.read()
        if ret:
            # Make grayscale and copy for drawing
            grayscale = frame[..., 0]
            # Do tracking
            tracking = track_frame(grayscale, threshold)
            # Collect tracked data
            tracking_output.append(tracking)
            # Show image
            if show_tracking:
                show = frame.copy()
                img = draw_tracking(show, tracking)
                cv2.imshow("img", img)
                k = cv2.waitKey(1)
                if k in (27, 32, 13):
                    break
        else:
            break
    tracking_output = np.array(tracking_output)
    return tracking_output


if __name__ == "__main__":
    video_path = r"data/example_video.avi"
    output_path = r"data/example_tracked.npy"
    tracking_output = track_video(video_path, threshold=100, show_tracking=True)
    np.save(output_path, tracking_output)
    print("TRACKING FINISHED", tracking_output.shape)
