from __future__ import division
import numpy as np
import operator

#DeepSingh: Need to put comments on code

class NaiveBayes(object):   
    def __init__(self, alpha=1):     
        self.prior = {} 
        self.per_feature_per_label = {}       
        self.feature_sum_per_label = {}     
        self.likelihood = {}   
        self.posterior = {}
        self.alpha = alpha     
        self.p = None 
        
    def compute_prior(self, y):
        for k in y: #DeepSingh : Can use counter for creating dictionary
            if k in self.prior:
                self.prior[k] += 1
            else:
                self.prior[k] = 1

        for k,v in self.prior.iteritems():
            self.prior[k] = v/len(y)
    
    def compute_likelihood(self, X, y):   
	#DeepSingh: Some loops can be reduced
        for k,v in zip(y,X):
            if k in self.per_feature_per_label:
                self.per_feature_per_label[k] += v
            else:
                self.per_feature_per_label[k] = v
            
            if k in self.feature_sum_per_label:
                self.feature_sum_per_label[k] += sum(v)
            else:
                self.feature_sum_per_label[k] = sum(v)
           
        for k in self.feature_sum_per_label.iterkeys():
            self.likelihood[k] = (self.per_feature_per_label[k] + self.alpha)/(self.feature_sum_per_label[k] + (self.alpha*self.p))

    
    def fit(self, X, y):
        self.p = X.shape[1]
        self.compute_prior(y)
        self.compute_likelihood(X, y)
    
    def predict(self, X):
        predicted_vals = []
        row_value = {}
        for row in X:
            for k, v in self.prior.iteritems():
                row_value[k] = np.log(v) + sum(row * np.log(self.likelihood[k]))
            predicted_vals.append(max(row_value.iteritems(), key=operator.itemgetter(1))[0])
        return predicted_vals
  
    
    def score(self, X, y):
        return sum(self.predict(X) == y)/len(y)

