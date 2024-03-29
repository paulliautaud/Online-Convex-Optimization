import numpy as np


class LinearSVM:
    def __init__(self, m) -> None:
        self.w = np.random.randn(m)

    def loss(self, a, b, l):
        """
        Compute the loss for the data and provided parameters
        a (nxm) : data
        b (n) : labels
        l (float ): regularization parameter (lambda)
        """

        hinge = 1 - (a.dot(self.w) * b)
        hinge[np.where(hinge <= 0)] = 0
        return np.mean(hinge) + l / 2 * np.linalg.norm(self.w, 2)**2

    def gradLoss(self, a, b, l):
        """
        Compute the gradient of the loss wrt the current parameters and data
        x (m) : params
        a (nxm) : data
        b (n) : labels
        l (float) : regularization parameter (lambda)"""

        if b.shape != ():
            # reshape b to nxm to use term-by-term multiplication
            grad = - (np.repeat(b[:, np.newaxis], a.shape[1], 1)) * a
        else:
            grad = -b * a  # no need for repeat if n=1

        temp = 1 - (a.dot(self.w) * b)
        grad[np.where(temp <= 0)] = 0
        return np.mean(grad, 0) + l * self.w

    def instGradLoss(self, a, b, l):
        """
        Compute the gradient of the loss wrt the current parameters and data
        x (m) : params
        a (nxm) : data (number x size)
        b (n) : labels
        l (float) : regularization parameter (lambda)"""

        temp = 1 - a.dot(self.w)
        if b.shape != ():
            # reshape b to nxm to use term-by-term multiplication
            grad = - (np.repeat(b[:, np.newaxis], a.shape[1], 1)) * a
        else:

            grad = -b * a  # no need for repeat if n=1
        grad[np.where(temp <= 0)] = 0
        return grad + l * self.w

    def predict(self, data):
        """
        predict values using coeff x and new data data
        data (list) : the data for prediction
        """

        return np.sign(data.dot(self.w))
