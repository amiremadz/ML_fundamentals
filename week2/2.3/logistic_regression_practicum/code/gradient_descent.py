import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline



class GradientDescent(object):

    def __init__(self, cost, gradient, predict):
        '''
        INPUT: GradientDescent, function, function, function
        OUTPUT: None

        Initialize class variables. Takes three functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        predict: function to calculate the predicted values (0 or 1) for 
        the given data
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict = predict

    def run(self, X, y, alpha=0.01, num_iterations=100):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient descent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        #self.coeffs = np.zeros(X.shape[1])
        #self.coeffs = self.gradient(X, y, self.coeffs, alpha, num_iterations)
        #print "IN run"
        
        #Training the model
        
        rows, cols = X.shape
        self.coeffs = np.zeros(cols+1) #initialize coeffs (often as all 0's)
        for i in range(num_iterations):
            self.coeffs = self.coeffs - alpha * self.gradient(X, y, self.coeffs)
            #print "Cost is:",self.cost(X,y,self.coeffs)


    def predict_gd(self, X):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)

        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's.
        '''
        #Calling the predict function to get the predicted values
        
        return self.predict(X,self.coeffs)
    
    
    
    def accuracy(self, X, y):
        
        y_pred = self.predict_gd(X)
        return sum([1 if y_pred_i == y_i else 0 for y_pred_i, y_i in zip(y_pred, y)]) / float(X.shape[0])
