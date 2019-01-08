class MCMC:
    '''
    Markov Chain Monte Carlo algorithm
    Parameters:
        - distribution : distribution to be simulated
        - integrator : in case of HMC, integrator used
        - kernel : in case of MH, Markov kernel used
        - params (dict) : additionnal parameters
            'init' : starting value for Markov Chain
        - value : current value (Markov Chain)
        - saved_values : list recording values explored by Markov chain
        - accept : percentage of acceptation
    '''
    def __init__(self, distribution, integrator, kernel, params):
        self.distribution = distribution
        self.integrator = integrator
        self.kernel = kernel
        self.params = params
        try:
            self.init = params['init']
        except:
            self.init = 0
        self.value = self.init
        self.saved_values = [self.value]
        self.accept = 0
    

class MH(MCMC):
    '''
    Metropolis-Hastings algorithm
    Does not need any integrator but needs a Markov kernel to be specified
    '''
    def __init__(self, distribution, kernel, params):
        super().__init__(distribution = distribution, integrator = None, kernel = kernel, params = params)
    def iterate(self):
        '''
        One iteration of MH algorithm and return next value
        '''
        return self.kernel.simul(self.value, self.distribution)
    def run(self, n_iteration):
        '''
        Perform n iterations and update saved_values and accept
        '''
        self.kernel.accept = 0
        for k in range(n_iteration):
            self.value = self.iterate()
            self.saved_values.append(self.value)
        self.accept = self.kernel.accept/n_iteration

class HMC(MCMC):
    def __init__(self, distribution, integrator, params):
        super().__init__(distribution = distribution, integrator = integrator, kernel = None, params = params)
    def iterate(self):
        '''
        One iteration of HMC algorithm and return next value
        '''
        return self.integrator.simul(self.value, self.distribution)
    def run(self, n_iteration):
        '''
        Perform n iterations and update saved_values and accept
        '''
        self.integrator.accept = 0
        for k in range(n_iteration):
            self.value = self.iterate()
            self.saved_values.append(self.value)
        self.accept = self.integrator.accept/n_iteration

class RHMC(MCMC):
    def __init__(self, distribution, integrator, params):
        super().__init__(distribution = distribution, integrator = integrator, kernel = None, params = params)
    def iterate(self):
        '''
        One iteration of HMC algorithm and return next value
        '''
        return self.integrator.simul_rhmc(self.value, self.distribution)
    def run(self, n_iteration):
        '''
        Perform n iterations and update saved_values and accept
        '''
        self.integrator.accept = 0
        for k in range(n_iteration):
            self.value = self.iterate()
            self.saved_values.append(self.value)
        self.accept = self.integrator.accept/n_iteration
        