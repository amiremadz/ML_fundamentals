import numpy as np
import collections
from collections import defaultdict
import operator


class NaiveBayes(object):

    def __init__(self, alpha=1):
        self.prior = {}
        self.per_feature_per_label = defaultdict(float)
        self.feature_sum_per_label = defaultdict(float)
        self.likelihood = defaultdict(float)
        self.posterior = defaultdict(float)
        self.alpha = alpha
        self.p = None

    def compute_prior(self, y):
        #Counting the unique labels
        #label,count=np.unique(y,return_counts=True)
        #self.prior=dict(zip(label,count))
        self.prior=collections.Counter(y)


        

    def compute_likelihood(self, X, y):
        #Combining the label and feature and iterating to calculate the frequency and sum of feature per label.
         for label,feature in zip(y,X):
                self.per_feature_per_label[label]  +=feature
                self.feature_sum_per_label[label] +=sum(feature)
                #self.feature_sum_per_label.setdefault(label,[]).append(sum(feature))
         
         for label,val in self.per_feature_per_label.iteritems():
                self.likelihood[label] += np.divide(val+np.asarray(self.alpha),self.feature_sum_per_label[label]+np.asarray(self.alpha*self.p))

    def fit(self, X, y):
        self.p = X.shape[1]
        self.compute_prior(y)
        self.compute_likelihood(X, y)

    def predict(self, X):
        #List to store the predicted values
        result=[]
        for row in X:
            for label in self.prior.iterkeys():
                self.posterior[label]=row.dot(np.log(self.likelihood[label]))+np.log(self.prior[label])
            result.append(sorted(self.posterior.items(), key=operator.itemgetter(1))[1][0])
        return result

    def score(self, X, y):
        y_pred = self.predict(X)
        return sum(y_pred == y)/float(len(y))
