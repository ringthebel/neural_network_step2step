import numpy as np
from general_function import *

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.output = np.zeros(y.shape)
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y

    def feedforward(self):
        #
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        #update weight
        self.weights1 += d_weights1
        self.weights2 += d_weights2

class Predict:
    neural_network = NeuralNetwork
    def __init__(self, x, w1, w2):
        self.input = x
        self.w1 = w1
        self.w2 = w2

    def predict(self):
        predict = sigmoid(np.dot(sigmoid(np.dot(self.input, self.w1)), self.w2))
        return predict


