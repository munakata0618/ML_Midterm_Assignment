import numpy as np
import matplotlib.pyplot as plt

n = 100
points = 3 * (np.random.rand(n, 2) - 0.5)
radius = (points ** 2).sum(axis=1)
mask = (radius > 0.7 + 0.1 * np.random.randn(n)) & (radius < 2.2 + 0.1 * np.random.randn(n))
labels = 2 * mask - 1
x = points[~mask]
y = points[mask]

np.save("toy_points.npy", points)
np.save("toy_labels.npy", labels)
plt.scatter(x[:,0], x[:,1], alpha=0.8, label="X")
plt.scatter(y[:,0], y[:,1], alpha=0.8, label="Y")
plt.legend()
plt.show()
