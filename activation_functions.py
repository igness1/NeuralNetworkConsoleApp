import numpy as np

class ActivationFunctions:
    def unistep(self, x):
        if x < 0:
            return 0
        elif x == 0:
            return 0.5
        else:
            return 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tahn(self, x):
        return np.tanh(x)

    def deriviate_sigmoid(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def deriviate_tahn(self, x):
        return 1 - (self.tahn(x)) ** 2

    def ReLU(self, x):
        x = np.maximum(0, x)
        return x

    def deriviate_ReLU(self, x):
        r = self.ReLU(x)
        r = np.where(r > 0,1,0)
        return r

    def __init__(self):
        self.function_dict = {"sigmoid": self.sigmoid, "tahn": self.tahn, "ReLU": self.ReLU}
        self.deriviate_dict = {"sigmoid": self.deriviate_sigmoid, "tahn":self. deriviate_tahn, "ReLU": self.deriviate_ReLU}

