import scipy.stats as scs
import numpy as np

class Gamma:
    '''
    Gamma distribution
    Parameters:
        - a : alpha
        - b : beta
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def pdf(self, x):
        '''
        Density function
        '''
        return scs.gamma.pdf(x, a = self.a, scale = 1/self.b)
    
    def log_pdf(self, x):
        '''
        Log-density function
        '''
        return scs.gamma.logpdf(x, a = self.a, scale = 1/self.b)

    def log_pdf_deriv(self, x):
        '''
        Derivative of the log-density
        '''
        return -x*self.b + (self.a-1)/x

class Normal:
    '''
    Normal distribution
    Parameters:
        - m : mean
        - s : standard deviation
    '''
    def __init__(self, mu, sigma):
        self.m = mu
        self.s = sigma
    
    def pdf(self, x):
        '''
        Density function
        '''
        return scs.norm.pdf(x, loc=self.m, scale=self.s)
    
    def log_pdf(self, x):
        '''
        Log-density function
        '''
        return scs.norm.logpdf(x, loc=self.m, scale=self.s)  

    def log_pdf_deriv(self,x):
        '''
        Derivative of the log-density
        '''
        return -0.5*((x-self.m)/self.s)**2 