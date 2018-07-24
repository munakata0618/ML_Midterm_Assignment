import numpy as np
import matplotlib.pyplot as plt

X = np.load("toy_points.npy")
Y = np.load("toy_labels.npy")
w1 = np.array([0, 0])
w2 = np.array([0, 0])
eta = 0.01
step = np.arange(50)
loss_values = np.zeros(50)

def loss_function(w):
	global X, Y
	lam = 0.00001
	ind = -Y*np.dot(w, X.T)
	log = np.log(1 + np.exp(ind))
	return log.sum() + lam * w@w

def hessian_J(w):
	global X, Y
	lam = 0.00001
	goukei = np.zeros((2, 2))
	ind = -Y*np.dot(w, X.T)
	numer = -1 * np.exp(ind)
	denom = (1 + np.exp(ind)) ** 2
	coeff = numer / denom
	for c, x in zip(coeff, X):
		xt = x.reshape(2, 1)
		goukei += c + xt*x + 2*lam*np.eye(2,2)
	return goukei

def grad_J(w):
	global X, Y
	lam = 0.00001
	ind = -Y*np.dot(w, X.T)
	numer = -Y * np.exp(ind)
	denom = 1 + np.exp(ind)
	gradient = numer / denom * X.T
	return gradient.sum(axis=1) + 2*lam*w

for t in step:
	loss_values[t] = loss_function(w1)
	w1 = w1 - np.linalg.inv(hessian_J(w1)).dot(grad_J(w1))

plt.plot(step, loss_values, label="newton")

for t in step:
	loss_values[t] = loss_function(w2)
	w2 = w2 - eta * grad_J(w2)


plt.plot(step, loss_values, label="steepest")
plt.legend()
plt.title("learning rate = 0.01")
plt.savefig("compare2.pdf")
plt.show()