#Specific choices for the approximation architecture and the exogenous information processes
from math import floor, pi, sin, sqrt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.svm import SVR
from sklearn import linear_model
import numpy as np
import GPy


#uses scikit-learn
class Gaussian_Process_Regression:
    '''Performs GPR (used as approximation architecture)'''
    def __init__(self):
        length_scale = 1
        self.kernel =  RBF(length_scale)*(1 / sqrt(2 * pi * (length_scale**2)))
        self.gp = GaussianProcessRegressor(kernel = self.kernel, alpha = 1e-2, n_restarts_optimizer=6, normalize_y = True)

    def fit(self, training_data, target_values):
        '''Returns the prediction function of the GPR'''
        self.gp.fit(training_data, target_values)
        return self

    def predict(self, test_point):
        return self.gp.predict(test_point)

    #TODO: DELETE? (implemented in gpr.py (direct modification of sklearn files))
    def jacobian(self, test_point):
        '''Returns the jacobian of the mean of the predictive distribution wrt to the decisions of the testpoint'''
        return self.gp.jacobian_RBF(test_point)[4:]

#uses GPy
class Gaussian_Process_Regression_GPy:
    def __init__(self):
        length_scale = 1.
        self.kernel = GPy.kern.RBF(input_dim=3, variance=1./sqrt(2*pi*length_scale**2), lengthscale = length_scale) #TODO: richtiger Kernel?

    #    self.kernel = GPy.kern.RBF(input_dim=3, variance=1., lengthscale = length_scale)

    def fit(self, training_data, target_values):
        self.m = GPy.models.GPRegression(training_data, target_values, self.kernel, normalizer = True)
        self.m.optimize_restarts(num_restarts = 5)
        print("Kernel: ", self.kernel)
        print("Mean of target values:", np.mean(target_values))

    def predict(self, test_point):
        return self.m.predict(test_point)[0]

#uses scikitlearn
class Linear_Regression:
    def __init__(self):
        self.regr = linear_model.LinearRegression()

    def fit(self, training_data, target_values):
        self.regr.fit(training_data, target_values)

    def predict(self, testpoint):
        return self.regr.predict(testpoint)[0]

#TODO: choice of error penalty C?
#TODO: scale data, since not scale invariant
class Support_Vector_Regression:
    '''Performs SVR (used as approximation architecture)'''
    def __init__(self):
        self.svr = SVR(kernel='rbf')

    def fit(self, training_data, target_values):
        self.svr.fit(training_data, target_values)

    def predict(self, testpoint):
        return self.svr.predict(testpoint)[0]

class Stochastic_Gradient_Descent:
    '''Performs SVR (used as approximation architecture)'''
    def __init__(self):
        self.sgd = linear_model.SGDRegressor()

    def fit(self, training_data, target_values):
        self.sgd.fit(training_data, target_values)

    def predict(self, testpoint):
        return self.sgd.predict(testpoint)[0]



# VF = 0
class Foo:
    def __init__(self):
        pass
    def fit(self, training_data, target_values):
        pass
    def predict(self, test_point):
        return 0

class Foo_Preprocessor:
    '''Test Preprocessor, doesn't transform anything.'''
    def __init__(self):
        pass
    def fit_transform(self, X):
        return X
    def transform(self, X):
        return X

class Pseudonormal_Distribution:
    '''Sample from PN(mu, sigma^2, x_min, x_max, x_stepsize), as described in paper (Does Anything Work)'''
    def __init__(self, mu, sigma, x_min, x_max, x_stepsize):
        self.mean = mu
        self.variance = sigma**2
        self.x_min = x_min
        self.x_max = x_max
        self.x_stepsize = x_stepsize
        self.support = [x for x in np.arange(self.x_min, self.x_max + self.x_stepsize, self.x_stepsize)]

        constant = 1. / sqrt(2. * pi * self.variance)
        normal_densities = [constant * np.exp(- (x - self.mean)**2 / (2 * self.variance)) for x in self.support]
        sum_scale = np.sum(normal_densities)
        self.probabilities = [normal_densities[i] / sum_scale for i in range(len(self.support))]

    def sample(self):
        return np.random.choice(self.support, p = self.probabilities)

class Uniform_Distribution:
    '''Sample from discretized U(-1, 1)'''
    def __init__(self, E_stepsize):
        self.elements = np.arange(-1, 1 + E_stepsize, E_stepsize)

    def sample(self):
        return np.random.choice(self.elements)

class Sinusoidal_Price:
    def __init__(self, model, eps_dstr):
        self.P_min = model.P_min
        self.P_max = model.P_max
        self.t_max = model.t_max
        self.eps_dstr = eps_dstr

    def sample(self, t, previous):
        '''Given an error distributed according to distribution eps_dstr, sample from the sinusoidal price process at time t.'''
        return min( max( 40 - 10 * sin(( 5 * pi * t) / ( 2 * self.t_max)) + self.eps_dstr.sample(), self.P_min), self.P_max)

class MC_Jumps_Price:
    '''Markov Chain model for the price process, either with or without additional jumps)'''
    def __init__(self, model, eps_dstr, with_jumps = 0, jump_dstr = None, p = 0.031):
        self.P_min = model.P_min
        self.P_max = model.P_max
        self.eps_dstr = eps_dstr
        self.with_jumps = with_jumps

        self.jump_dstr = jump_dstr if with_jumps else eps_dstr
        self.p = p

    def sample(self, t, previous):
        if t == 0:
            return self.P_min
        else:
            bool_jump = 1 if (self.with_jumps and (np.random.random_sample() <=self.p)) else 0
            return min( max(previous + self.eps_dstr.sample() + bool_jump * self.jump_dstr.sample(), self.P_min), self.P_max)

class MC_Energy:
    '''Markov Chain model for the wind energy process with uniformly or pseudornormally distributed errors'''
    def __init__(self, model, eps_dstr):
        self.E_min = model.E_min
        self.E_max = model.E_max
        self.E_stepsize = model.E_stepsize
        self.eps_dstr = eps_dstr

    def sample(self, t, previous):
        if t == 0:
            return np.random.choice(np.arange(self.E_min, self.E_max + self.E_stepsize, self.E_stepsize))
        else:
            return min( max(previous + self.eps_dstr.sample(), self.E_min), self.E_max)

class Demand:
    def __init__(self, model):
        self.t_max = model.t_max

    def sample(self, t):
        return max(0, 3 - 4 * sin((2 * pi * t/ self.t_max)))
