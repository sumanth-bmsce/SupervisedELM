import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, normalize, Binarizer

class Normalizer:

    def __init__(self, data):
        self.__data = data

    # Converts the data to [0, 1] range
    def minmax(self):
        min_max_scaler = MinMaxScaler()
        return np.array(min_max_scaler.fit_transform(self.__data))

    # Converts the data to [-1, 1] range
    def maxabs(self):
        max_abs_scalar = MaxAbsScaler()
        return np.array(max_abs_scalar.fit_transform(self.__data))

    # Apply l2 norm
    def l2norm(self):
        return np.array(normalize(self.__data, norm = 'l2'))

    # Convert the data to a numpy array of 0s and 1s
    def binarizer(self, thresh = 0.0, copy = True):
        bin = Binarizer(thresh, copy)
        return np.array(bin.transform(self.__data))






