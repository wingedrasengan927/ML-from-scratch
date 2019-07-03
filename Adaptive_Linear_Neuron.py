
import numpy as np

class Adaline(object):
    '''Adaline Classifier

    parameters
    -----------
    eta: float
        learning rate (between 0-1)
    n_iters: int
        number of iterations

    Attributes
    -----------
    w_: array-like 1d
        Weights
    cost_list: list
        The cost function value for every iteration
    log_transform: bool
        applies log transform to cost function to prevent overflow
    '''

    def __init__(self, eta=0.01, n_iters=1000, log_transform=False,
                shuffle=True, random_state=None):
        self.eta = eta
        self.n_iters = n_iters
        self.log_transform = log_transform

    def fit(self, X, y):
        '''Fit the classifier on the training data and optimize the weights

        Parameters
        -----------
        X: array-like, shape=[n_observations, n_samples]
            training data
        y: array-like, shape = [n_observations]
            target labels

        Returns
        -----------
        self object
        '''

        self.w_ = np.zeros(1+X.shape[1])

        self.cost_list = []

        for i in range(self.n_iters):
            output = self.weighed_input(X)
            errors = y - output
            self.w_[1:] += self.eta * (X.T @ errors)
            self.w_[0] += self.eta * errors.sum()
            if self.log_transform:
                cost = np.log((1/2) * (errors**2).sum())
            else:
                cost = (1/2) * (errors**2).sum()
            self.cost_list.append(cost)

        return self
    
    def weighed_input(self, X):
        "Returns the weighed Input"
        return (X @ self.w_[1:]) + self.w_[0]

    def predict(self, X):
        predicts = self.weighed_input(X)
        return np.where(predicts >= 0, 1, -1)
