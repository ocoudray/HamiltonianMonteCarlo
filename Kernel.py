import numpy as np

class Gaussian_kernel:
    '''
    Gaussian transition for Metropolis-Hastings algorithm
    Parameters:
        - s : standard deviation
        - accept : percentage of acceptation during MCMC
    '''
    def __init__(self, std):
        self.s = std
        self.accept = 0
    def simul(self, value, distribution):
        new_value = value + self.s * np.random.randn()
        new_pdf = distribution.pdf(new_value)
        pdf = distribution.pdf(value)
        alpha = new_pdf/pdf
        if np.isnan(alpha):
            return value
        u = np.random.rand()
        if u <= alpha:
            self.accept += 1
            return new_value
        else:
            return value
