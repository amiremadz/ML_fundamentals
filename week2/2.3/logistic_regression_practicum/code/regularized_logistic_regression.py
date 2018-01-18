import numpy as np

def featureMatrixNew(X):
    
    '''This method adds a column to the feature matrix
    INPUT : Feature Matrix X
    OUTPUT: new Feature Matrix with column added
    '''
    X_new=np.insert(X, 0, 1, axis=1)
    return X_new
    

def hypothesis(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted percentages (floats between 0 and 1) 
    for the given data with the given coefficients.
    '''
    X_new=np.insert(X, 0, 1, axis=1)
    c=X_new.dot(coeffs)
    h=1./(1+np.exp(-c))
    return h

def predict(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted values (0 or 1) for the given data with 
    the given coefficients.
    '''
    
    #Using the logic that if beta*x>0 then y=1 else y=0 , interpretation from the graph
    
    p=[]  
    X_new=np.insert(X, 0, 1, axis=1)
    c=X_new.dot(coeffs)
    for i in range(0,len(c)):
        if c[i]>=0:
            p.append(1)
        else:
             p.append(0)
    return np.asarray(p)

def cost_function(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the value of the cost function for the data with the 
    given coefficients.
    '''
    #Cost function calculation using vectorized form
    
    h = hypothesis(X,coeffs)
    l=  y.dot(np.log(h))+ (1-y).dot(np.log(1-h))
    return -l/len(y)

def cost_regularized(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the value of the cost function with regularization 
    for the data with the given coefficients.
    '''
    lam=1.0
    h = hypothesis(X,coeffs)
    print "hypothesis",h
    l=  -(y.dot(np.log(h))+ (1-y).dot(np.log(1-h)))+ (lam*coeffs.T.dot(coeffs))/2
    print "cost", -l/len(y)
    return -l/len(y)

  

def cost_gradient(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the cost function at the given value 
    for the coeffs. 

    Return an array of the same size as the coeffs array.
    '''
    
    h = hypothesis(X,coeffs)
    print "hypothesis",h
    X_new=np.insert(X, 0, 1, axis=1)
    lg=np.asarray(h-y).dot(X_new)
    print "cost func",lg
    return lg

def gradient_regularized(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the cost function with regularization 
    at the given value for the coeffs. 

    Return an array of the same size as the coeffs array.
    '''
    
    #Assigning lambda as 1
    
    lam=1.0
    beta=np.copy(coeffs)
    beta[0]=0
    h = hypothesis(X,coeffs)
    
    X_new=np.insert(X, 0, 1, axis=1)
    grad=np.asarray(h-y).dot(X_new)+lam*beta
    
    return grad/len(y)
    

