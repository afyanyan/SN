import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
from astropy.time import Time
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from astropy.timeseries import TimeSeries
from numpy import random
from sklearn.gaussian_process.kernels import ConstantKernel as C, WhiteKernel, Matern, DotProduct, RBF
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from scipy.optimize import curve_fit
from scipy.optimize import minimize



# Define Exponential Composite Kernel: K = exp(gamma * Z - alpha * |Z - beta|**(1 + delta))
class ExpCompositeKernel(Kernel):
    
    # Init method
    def __init__(
        self,
        alpha=0.1,
        alpha_bounds=(1e-6, 1e2),
        beta=0.1,
        beta_bounds=(1e-6, 1e2),
        gamma=0.1,
        gamma_bounds=(1e-6, 1e2),
        delta=1.0e-3,
        delta_bounds=(1e-6, 1e2),
    ):
        self.alpha = alpha
        self.alpha_bounds = alpha_bounds
        self.beta = beta
        self.beta_bounds = beta_bounds
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds
        self.delta = delta
        self.delta_bounds = delta_bounds

    # Hyperparameters definitions
    @property
    def hyperparameter_alpha(self):
        return Hyperparameter('alpha', 'numeric', self.alpha_bounds)

    @property
    def hyperparameter_beta(self):
        return Hyperparameter('beta', 'numeric', self.beta_bounds)

    @property
    def hyperparameter_gamma(self):
        return Hyperparameter('gamma', 'numeric', self.gamma_bounds)

    @property
    def hyperparameter_delta(self):
        return Hyperparameter('delta', 'numeric', self.delta_bounds)
    
    # Kernel is non-stationary
    def is_stationary(self):
        return False

    # Diagonal of kernel
    def diag(self, X):
        # Compute inner product
        Z = np.einsum('ij,ij->i', X, X)
        diff = np.abs(Z - self.beta)
        A = diff ** (1.0 + self.delta)
        return np.exp(self.gamma * Z - self.alpha * A)

    # Full kernel
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)

        # Compute inner product
        if Y is None:
            Z = np.inner(X, X)
        else:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluated when Y is None.')
            Z = np.inner(X, Y)

        # Regularization to avoid log(0)
        eps = 1e-100
        diff = np.abs(Z - self.beta) + eps
        logdiff = np.log(diff)
        A = diff ** (1.0 + self.delta)

        # Kernel function
        K = np.exp(self.gamma * Z - self.alpha * A)

        if not eval_gradient:
            return K

        # Gradients w.r.t. log-parameters: dK/d(ln(p)) = p * dK/d(p)
        alpha_grad = -self.alpha * A * K
        beta_grad = self.beta * self.alpha * (1.0 + self.delta) * np.sign(Z - self.beta) * (diff ** self.delta) * K
        gamma_grad = self.gamma * Z * K
        delta_grad = -self.delta * self.alpha * A * logdiff * K

        # Stack gradients: shape (n, n, n_hyperparameters)
        K_gradient = np.stack((alpha_grad, beta_grad, gamma_grad, delta_grad), axis=2)

        return K, K_gradient

# Define Quartic Kernel: K = p0 + p1 * Z + p2 * Z **2 + p3 * Z**3 + p4 * Z**4
class QuarticKernel(Kernel):

    # Init method
    def __init__(
        self,
        p0=1.0,
        p0_bounds=(1e-6, 1e2),
        p1=1.0e-3,
        p1_bounds=(1e-6, 1e2),
        p2=1.0e-3,
        p2_bounds=(1e-6, 1e2),
        p3=1.0e-3,
        p3_bounds=(1e-6, 1e2),
        p4=1.0,
        p4_bounds=(1e-6, 1e2),
    ):
        self.p0 = p0
        self.p0_bounds = p0_bounds
        self.p1 = p1
        self.p1_bounds = p1_bounds
        self.p2 = p2
        self.p2_bounds = p2_bounds
        self.p3 = p3
        self.p3_bounds = p3_bounds
        self.p4 = p4
        self.p4_bounds = p4_bounds

    # Hyperparameter definitions
    @property
    def hyperparameter_p0(self):
        return Hyperparameter('p0', 'numeric', self.p0_bounds)

    @property
    def hyperparameter_p1(self):
        return Hyperparameter('p1', 'numeric', self.p1_bounds)

    @property
    def hyperparameter_p2(self):
        return Hyperparameter('p2', 'numeric', self.p2_bounds)

    @property
    def hyperparameter_p3(self):
        return Hyperparameter('p3', 'numeric', self.p3_bounds)

    @property
    def hyperparameter_p4(self):
        return Hyperparameter('p4', 'numeric', self.p4_bounds)

    # Kernel is non-stationary
    def is_stationary(self):
        return False

    # Diagonal of kernel
    def diag(self, X):
        # Compute inner product
        Z = np.einsum('ij,ij->i', X, X)
        return self.p0 + self.p1 * Z + self.p2 * Z**2 + self.p3 * Z**3 + self.p4 * Z**4

    # Full kernel
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)

        # Compute inner product
        if Y is None:
            Z = np.inner(X, X)
        else:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluated when Y is None.')
            Z = np.inner(X, Y)

        # Kernel function
        K = self.p0 + self.p1 * Z + self.p2 * Z**2 + self.p3 * Z**3 + self.p4 * Z**4

        if not eval_gradient:
            return K

        # Gradients w.r.t. log-parameters dK/d(ln(p)) = p * dK/d(p)
        p0_grad = self.p0 * np.ones_like(K)
        p1_grad = self.p1 * Z
        p2_grad = self.p2 * Z**2
        p3_grad = self.p3 * Z**3
        p4_grad = self.p4 * Z**4

        # Stack gradients: shape (n, n, n_hyperparameters)
        K_gradient = np.stack((p0_grad, p1_grad, p2_grad, p3_grad, p4_grad), axis=2)
        return K, K_gradient
   

def custom_optimizer(obj_func, initial_theta, bounds):
    # Safe objective function to handle NaN or inf values
    def safe_obj(theta):
        out = obj_func(theta)
        # Extract scalar value [sklearn may return (value, gradient) or arrays: coerce to scalar]
        if isinstance(out, tuple) or isinstance(out, list):
            val = out[0]
        else:
            val = out
        val =float(np.asarray(val).ravel()[0])

        # Handle NaN or inf values
        try:
            if np.isnan(val) or np.isinf(val):
                val = 1e100
        except:
            pass

        return val   

    # Wrapper to run the minimization
    def run_minimize(x0):
        res = minimize(safe_obj, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 300})
        return res

    # Generate random starting point within bounds
    bds = np.array(bounds)
    x0 = np.random.uniform(bds[:,0], bds[:,1])
    res = run_minimize(x0)
    return (res.x, float(res.fun))

# Function to summarize kernel hyperparameters
def summarize_kernel(kernel):
    print(f'Kernel type: {kernel.__class__.__name__}')
    print('Optimized hyperparameters:')
    for hyper, theta_val in zip(kernel.hyperparameters, kernel.theta):
        print(f'  {hyper.name:10s}: {np.exp(theta_val):.6g}', f' (bounds: {hyper.bounds})')

# GPR fitting function
def GPR_fit(X,y):
    global gpr, Xp

    # Reshape input data
    Xr = X.reshape(-1, 1)
    Xp = np.linspace(X.min()-10, X.max()+10, 600)
    Xpr = Xp.reshape(-1,1)

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(Xr)
    Xrs = scaler.transform(Xr)
    Xprs = scaler.transform(Xpr)

    # Define custom kernel
    kernel = (
        QuarticKernel(p0=1,p1=1e-3,p2=1e-3,p3=1e-3,p4=1,
                      p0_bounds=(1e-6, 1e2), p1_bounds=(1e-6, 1e2), p2_bounds=(1e-6, 1e2), 
                      p3_bounds=(1e-6, 1e2), p4_bounds=(1e-6, 1e2))
        * ExpCompositeKernel(alpha=0.1,beta=0.1,gamma=0.1,delta=1e-4,
                             alpha_bounds=(1e-6, 1e2), beta_bounds=(1e-6, 1e2),
                             gamma_bounds=(1e-6, 1e2), delta_bounds=(1e-6, 1e2))
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e2))   
    )

    # Initialize the GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                               normalize_y=True,
                               n_restarts_optimizer=10,
                               optimizer=custom_optimizer)#,optimizer=None)
    # Fit the model
    gpr.fit(Xrs, y.reshape(-1,1))

    # Make predictions
    yp, ystd = gpr.predict(Xprs, return_std=True)

    # Summarize the optimized kernel
    summarize_kernel(gpr.kernel_)

    return Xp, yp, ystd