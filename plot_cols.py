import matplotlib.pyplot as plt

from utils import *
from constants import *

df = load_dataset()

x_labels = [i for i in range(NUM_SAMPLES)]
for col, values in df.iteritems():
    print("Plotting {}...".format(col))
    plt.plot(x_labels, sorted(values, reverse = True))
    plt.title("Distribution of {}".format(col))
    plt.xlabel("Samples")
    plt.ylabel("Value (normalized)")
    plt.savefig("./plots/{}_distribution.png".format(col))
    plt.figure()

# Uncomment for fun
# plt.show()
