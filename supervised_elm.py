from SELM.activations import Activation
import numpy as np

class Supervised_ELM:

    def __init__(self, train_data, labels, activation, hidden_neurons):
        self.__train_data = train_data
        self.__labels = labels
        self.__activation = activation
        self.__hidden_neurons = hidden_neurons
        self.__nsamples , self.__ndim = self.__train_data.shape
        self.__input_weights = np.random.rand(self.__ndim, self.__hidden_neurons)
        self.bias = np.random.rand(self.__hidden_neurons)
        self.output_weights = None

    @property
    def train_data(self):
        return self.__train_data

    @property
    def activation(self):
        return self.__activation

    @property
    def hidden_neurons(self):
        return self.__hidden_neurons

    def hidden_layer_activations(self, data, bias):
        h = np.dot(data, self.__input_weights)
        h = np.add(h, bias)
        a = Activation()
        return getattr(a, self.__activation)(h)

    def train(self):
        bias = np.array([self.bias]*self.__nsamples)
        h = self.hidden_layer_activations(self.__train_data, bias)
        labelst = np.transpose(self.__labels)
        self.output_weights = np.dot(np.linalg.pinv(h), labelst)

    def test(self, test_data):
        predict = []
        nsamples = test_data.shape[0]
        bias = np.array([self.bias]*nsamples)
        h = self.hidden_layer_activations(test_data, bias)
        result = np.dot(h, self.output_weights)
        for i in result:
            if i < 0:
                predict.append(0)
            else:
                predict.append(1)
        return predict
