import numpy as np
import pandas as pd


def read_eye_tracking(path):
    tracking = np.load(path)
    tracking = tracking.reshape(-1, np.multiply(*tracking.shape[1:]))
    df = pd.DataFrame(tracking, columns=["sb_x", "sb_y", "sb_angle",
                                         "left_x", "left_y", "left_angle",
                                         "right_x", "right_y", "right_angle"])
    return df
