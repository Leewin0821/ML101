import numpy as np


class SigmoidActivator(object):
    @staticmethod
    def forward(weight):
        return 1.0 / (1.0 + np.exp(-weight))

    @staticmethod
    def backward(y):
        return y * (1 - y)


class ReluActivator(object):
    @staticmethod
    def forward(x):
        return max(x, 0)

    @staticmethod
    def backend(y):
        return 1 if y > 0 else 0


class IdentityActivator(object):
    @staticmethod
    def forward(weight):
        return weight

    @staticmethod
    def backend():
        return 1
