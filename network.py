import math
import numpy as np
from typing import List

class Layer:
	def __init__(self, dim: tuple):
		# self.weights = np.random.rand(*dim)
		self.weights = np.zeros(dim)
		self.biases = np.ones(dim)
		self.values = np.zeros(dim)

class Activations:
	@staticmethod
	def tanh(vector):
		return np.tanh(vector)

	@staticmethod
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def sigmoid_prime(a):
		return a * (1 - a)

class Network:
	def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_hidden_layers: int):
		print(input_dim, hidden_dim, num_classes, num_hidden_layers)
		self.layers = []

		if num_hidden_layers < 1:
			self.layers.append(Layer((input_dim, num_classes)))

			return

		# first hidden layer will be fed into from the input layer
		self.layers.append(Layer((input_dim, hidden_dim)))

		# hidden layers can feed into other hidden layers
		for i in range(1, num_hidden_layers):
			self.layers.append(Layer((hidden_dim, hidden_dim)))

		# hidden to output
		self.layers.append(Layer((hidden_dim, num_classes)))

	def feedforward(self, x):
		for layer in self.layers:
			print(layer.weights.shape, layer.values.shape)
		exit()
		prev_values = x
		for layer in self.layers:
			layer.values = Activations.sigmoid(np.dot(prev_values, layer.weights))# - layer.biases

			prev_values = layer.values

		for layer in self.layers:
			print(layer.weights.shape, layer.values.shape)
		exit()

		return prev_values

	def backprop(self, x, y):
		error = None
		# application of the chain rule to find slope of the loss fn with respect to layer weights
		for i in range(len(self.layers) - 1, -1, -1):
			if error is None:
				error = y - self.layers[i].values
			else:
				print(delta.shape, self.layers[i].weights.T.shape)
				# z_i error: how much our hidden layer weights contributed to output error
				error = delta.dot(self.layers[i].weights.T)

			# update the weights with the derivative of the loss fn (in this case, loss and activation are both sigmoid)

			# delta output sum = output layer's error margin * derivative of sigmoid activation fn
			delta = error * Activations.sigmoid_prime(self.layers[i].values)
			print(error.shape, self.layers[i].values.shape, Activations.sigmoid_prime(self.layers[i].values).shape, delta.shape)

			# adjust layer weights
			if i > 0:
				self.layers[i].weights += self.layers[i].values.T.dot(delta)
			else:
				self.layers[i].weights += x.T.dot(delta)

	def train(self, x, y):
		self.feedforward(x)
		self.backprop(x, y)

class DataPrep():
	def collect(self):
		pass


if __name__ == '__main__':
	dataset = DataPrep().collect()

	# X = (hours sleeping, hours studying), y = score on test
	X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
	y = np.array(([92], [86], [89]), dtype=float)

	# scale units
	X = X/np.amax(X, axis=0) # maximum of X array
	y = y/100 # max test score is 100

	NN = Network(2, 3, 1, 1)
	for i in range(1):
		# print("Input: \n" + str(X))
		# print("Actual Output: \n" + str(y))
		# print("Predicted Output: \n" + str(NN.feedforward(X)))
		print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward(X))))) # mean sum squared loss
		# print("\n")
		NN.train(X, y)

	# network = Network(64**2, 10, 2, 1)
	# for i, d in enumerate(dataset):
	# 	hidden = network.feedforward(d, i)