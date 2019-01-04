import numpy as np

class Verlet:
    '''
    Velocity Verlet Integrator
    Estimate solution for differential equation : q'' + a(q) = 0
    Parameters:
        - acc : function q -> a(q)
        - step : step parameter in the integration
        - n_steps : number of iterations to perform
        - accept : percentage of acceptation while performing HMC
        - saved_p, saved_q : lists recording values of p and q visited
    '''
    def __init__(self, distribution, h = 0.001, n = 100):
        self.acc = distribution.log_pdf_deriv
        self.step = h
        self.n_steps = n
        self.accept = 0
        self.saved_p = []
        self.saved_q = []

    def iter_from(self, q0, p0, h):
        '''
        Perform one iteration for the integrator starting with initial conditions q0, p0
        (p = q')
        '''
        p_half = p0 - 0.5*h*self.acc(q0)
        q1 = q0 + p_half*h
        a1 = -self.acc(q1)
        p1 = p_half + a1*h*0.5
        return q1,p1
    
    def iterate(self, q, p, h, n):
        '''
        Perform n iterations starting from initial conditions (q,p)
        '''
        for k in range(n):
            q,p = self.iter_from(q,p,h)
        return q,p
    
    def simul(self, value, distribution):
        p0 = np.random.randn()
        new_value, p = self.iterate(value, p0, self.step, self.n_steps)
        try:
            alpha = -(1/2*p**2 - 1/2*p0**2)+(distribution.log_pdf(new_value)-distribution.log_pdf(value))
        except:
            alpha = 0
        u = np.log(np.random.rand())
        if np.isnan(alpha):
            self.saved_p.append(-p0)
            self.saved_q.append(value)
            return value
        elif u <= alpha:
            self.accept += 1
            self.saved_p.append(p)
            self.saved_q.append(new_value)
            return new_value
        else:
            self.saved_p.append(-p0)
            self.saved_q.append(value)
            return value