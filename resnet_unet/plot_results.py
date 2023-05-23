import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data from CSV file
with open("results.csv", "r") as f:
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

plt_dir = "exp/plts"
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)

# Plot data for each attribute
for i, attribute in enumerate(attributes):
    plt.figure()
    plt.plot(data[:, 0], data[:, i])
    plt.title(attribute)
    plt.xlabel(headers[0])
    plt.ylabel(headers[i+1])
    plt.savefig(plt_dir + "/" + attribute + ".png")