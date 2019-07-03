import numpy as np

class AdalineSGD():
    '''Adaptive Linear Neuron classifier
    with stochastic gradient descent

    Parameters
    -----------
    eta: float
        Learning rate
    n_iters: int
        Number of Iterations

    Attributes
    ------------
    w_: 1d-array like
        weights
    cost_list: list
        list containing the cost function value for every epoch
    log_transform: bool
        Apply log transform to the cost function to prevent overflow
    shuffle: bool 
        Shuffling the training data for every epoch
    random_state: int (default: None)
        set random state for shuffling
    '''

    def __init__(self, eta=0.01, n_iters=1000, log_transform = False,
                shuffle=True, random_state = None):
        self.eta = eta
        self.n_iters = n_iters
        self.log_transform = log_transform
        self.shuffle = shuffle
        self.w_initialized = False
        self.random_state = random_state

        if self.random_state:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        '''Fit the data and get the weights

        Parameters
        -----------
        X: array like, sahpe=[n_observations, n_features]
            Training data
        y: 1d-array like, shape=[n_observations]
            target labels
        
        Returns
        --------
        self object
        '''

        self.initialize_weights(X.shape[1])
        self.cost_list = []

        for i in range(self.n_iters):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost=[]
            for xi, yi in zip(X, y):
                cost.append(self.update_weights(xi, yi))
            avg_cost = sum(cost)/X.shape[0]
            self.cost_list.append(avg_cost)

        return self
    
    def partial_fit(self, X, y):
        '''Fit training data without reinitializing the weights
        or update the weights as new data comes in (online learning)'''
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        if y.flatten().shape[0] > 1:
            for xi, yi in zip(X, y):
                self.update_weights(xi, yi)
        else:
            self.update_weights(X, y)

        return self


    def _shuffle(self, X, y):
        "Shuffle the training data"
        r = np.random.permutation(X.shape[0])
        return X[r], y[r]

    def update_weights(self, xi, yi):
        "Use adaline rule to update weights based off a single observation"
        output = self.weighted_input(xi)
        error = yi - output
        self.w_[1:] += self.eta * (xi.dot(error))
        self.w_[0] += self.eta * error
        cost = 0.5 * (error**2)
        return cost

         

    def initialize_weights(self, m):
        "Initialize the weights to zeros"
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def weighted_input(self, X):
        "Compute the weighted Input"
        return (X@self.w_[1:]) + self.w_[0]

    def predict(self, X):
        "Predict the class label"
        return np.where(self.weighted_input(X)>=0, 1, -1)

