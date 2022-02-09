import pickle
from matplotlib import pyplot as plt
from track_video import output_path

with open(output_path, "rb") as f:
    tracking = pickle.load(f)

orientation, left, right = zip(*tracking)
plt.plot(left)
plt.plot(right)
plt.show()
