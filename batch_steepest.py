import numpy as np
import matplotlib.pyplot as plt

X = np.load("toy_points.npy")
Y = np.load("toy_labels.npy")
eta = 0.01
w = np.array([0, 0])
step = np.arange(100)
loss_values = np.zeros(100)

def loss_function(w):
	global X, Y
	lam = 0.00001
	ind = -Y*np.dot(w, np.transpose(X))
	log = np.log(1 + np.exp(ind))
	return log.sum() + lam * w.dot(w)

def grad_J(w):
	global X, Y
	lam = 0.00001
	ind = -Y*np.dot(w, X.T)
	numer = -Y * np.exp(ind)
	denom = 1 + np.exp(ind)
	gradient = numer / denom * X.T
	return gradient.sum(axis=1) + 2*lam*w

for t in step:
	w = w - eta * grad_J(w)
	loss_values[t] = loss_function(w)

plt.plot(step, loss_values)
plt.show()