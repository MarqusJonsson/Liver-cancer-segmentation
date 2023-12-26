import csv
import numpy as np
import matplotlib.pyplot as plt
import os

results_file = "exp/vicreg_aug_e10_unfreeze_loss_cce_lower_lr_label_smooth_0.1/results.csv"
plt_dir = "exp/vicreg_aug_e10_unfreeze_loss_cce_lower_lr_label_smooth_0.1/plts"

# Load data from CSV file
with open(results_file, "r") as f:
    reader = csv.reader(f, delimiter="|")
    headers = next(reader) # Get headers
    data = np.array(list(reader))

def to_float(x):
    try:
        return float(x)
    except ValueError:
        return 0.0

v_to_float = np.vectorize(to_float)
data = v_to_float(data)

# Get attribute names from headers
attributes = headers[1:]

if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)

# Plot data for each attribute
for i, attribute in enumerate(attributes):
    plt.figure()
    plt.plot(data[:, 0], data[:, i+1])
    plt.title(attribute)
    plt.xlabel(headers[0])
    plt.ylabel(headers[i+1])
    plt.savefig(plt_dir + "/" + attribute + ".png")