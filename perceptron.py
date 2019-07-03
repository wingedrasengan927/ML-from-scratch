import numpy as np

class Perceptron(object):

    '''Perceptron Classifier.

    Parameters
    -----------
    eta: float
        Learning Rate (between 0 and 1)
    n_iters: int
        Number of Iterations

    Attributes
    ------------
    w_: 1d Array
        The weights after fitting
    errors_: list
        The total number of misclassifications in each epoch

    '''

    def __init__(self, eta=0.01, n_iters=1000):
        self.eta = eta
        self.n_iters = n_iters

    def fit(self, X, y):
        '''Fit training data

        Parameters
        ----------
        X: array-like, shape=[n_samples, n_features]
            The Features Data
            n_samples - Number of Observatons
            n_features - Number of Features
        y: array-like, shape=[n_samples]
            Target Labels

        Returns
        -----------
        self object

        '''

        self.w_ = np.zeros(1+X.shape[1])
        self.error_list = []

        for i in range(self.n_iters):
            predicts = self.predict(X)
            error = y - predicts
            self.w_[1:] += self.eta * (X.T @ error)
            self.w_[0] += self.eta * error.sum()
            total_error =  sum(error != 0)
            self.error_list.append(total_error)

        return self


    def weighed_input(self, X):
        "Calculate the net input"
        weighed_input = (X @ self.w_[1:]) + self.w_[0]
        return weighed_input
    
    def predict(self, X):
        "Predict the class lables"
        weighed_input = self.weighed_input(X)
        return np.where(weighed_input >= 0, 1, -1)

