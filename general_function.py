import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds
def softmax(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A