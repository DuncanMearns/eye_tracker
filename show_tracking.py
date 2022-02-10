from matplotlib import pyplot as plt
from eye_tracking.io import read_eye_tracking

if __name__ == "__main__":
    tracking = read_eye_tracking(r"data/example_tracked.npy")
    plt.plot(tracking["left_angle"])
    plt.plot(tracking["right_angle"])
    plt.show()
