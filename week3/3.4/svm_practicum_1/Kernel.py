import numpy.linalg as la
import numpy as np

class Kernel(object):
    """Implements a list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x,y,sigma):
            c=(x-y)**2
            return np.exp(sigma*c)
        return f
            
        

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x,y,dimension,offset):
            return (np.inner(x, y)+offset)**dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
         def f(x,y,dimension):
            return (np.inner(x, y)+1)**dimension
	 return f
            

    @staticmethod
    def homogenous_polynomial(dimension):
        def f(x,y,dimension):
            return np.inner(x, y)**dimension
        return f
            

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x,y,kappa,c):
            return np.tanh(kappa*np.inner(x,y)+c)
        return f
            
