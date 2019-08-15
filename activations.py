import numpy as np

class Activation:

    def __init__(self):
        pass

    def identity(self, x):
        return x

    def binarystep(self, x):
        return np.heaviside(x, 1)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def arctan(self, x):
        return np.arctan(x)

    def relu(self, x):
        return np.maximum(0,x)

