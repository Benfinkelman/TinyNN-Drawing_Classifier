import numpy as np

class TinyNN:
    """
    A 1-neuron neural network for binary classification (A vs B).
    Output is P(class B).
    """

    def __init__(self, input_dim: int, lr: float = 0.1):
        self.lr = lr
        # weights: one per input pixel
        self.w = np.zeros(input_dim, dtype=np.float32)
        # bias: single scalar
        self.b = 0.0

    # turns values of z into probabilities
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # Return Probability of class B (returns a float)
    def predict_proba(self, x: np.ndarray) -> float:
        # this computes our input to sigmoid z = (w*x) + b
        z = np.dot(self.w, x) + self.b
        #plugs into sigmoid
        return float(self.sigmoid(z))
    
    #Gradient with respect to z -> dL/dz = p - y (=error)
    def gradients(self, x: np.ndarray, y: float):
        # make a prediction then find the error
        p = self.predict_proba(x)
        error = p - y #dL/dz

        dw = error * x
        db = error
        return dw, db

    # Perform a gradient descent update using ONE training example
    def train_step(self, x: np.ndarray, y: float):

        dw, db = self.gradients(x, y)

        # Update parameters wheights and biases based on the error calculated
        self.w -= self.lr * dw
        self.b -= self.lr * db

    # Train on all examples given an epoch
    def fit(self, X: np.ndarray, y: np.ndarray, epochs):

        N = X.shape[0]

        for epoch in range(epochs):
            for i in range(N):
                self.train_step(X[i], y[i])
    
    # Our average loss over the training
    ''' def average_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        for i in range(len(y)):
            p = self.predict_proba(X[i])
            total += self.loss(y[i], p)
        return total / len(y)
    '''

    #Loss function determines our predicted outcome vs preffered (1 or 0) returns this vlaue (Binary Cross Entropy)
    '''def loss(self, y_true: float, p_pred: float) -> float:
        eps = 1e-8  # prevents log(0)
        return - (y_true * np.log(p_pred + eps) + (1 - y_true) * np.log(1 - p_pred + eps))
    '''