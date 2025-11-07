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

#def light_curve(x,a,b,c,d,e,f,g,h):
#    return e**2*(1+c**2*(x/80)**h*np.tanh(f*(x/80)**a)*np.exp(-b*(x/80)**g)*np.exp(-d*(x/80)))
#def light_curve(x,a,b,c):
#    return c*np.tanh(a*x)*np.exp(-b*x**2)


#param, param_cov = curve_fit(light_curve, T, B)


#print('Parameters:', param)
#print('Covariance:', param_cov)


# Plot original and fitted data
#Tfit = np.linspace(0,100,101)
#plt.plot(T, B, '.', color='red', label="data")
#plt.plot(Tfit, light_curve(Tfit,*param), '-', color='blue', label="Fit")
#plt.ylim([0.94,1])
#plt.legend()
#plt.show()
lc = '../Quartic_Fit_OSW/o4_supernovae/SN_2024pxg/sn_2024pxg_atlas_o.dat'
bkg = '../Quartic_Fit_OSW/o4_BKG/SN_2024_pxg_background_tns.txt'

#bkg = '../Quartic_Fit_OSW/o4_BKG/SN_2024_phv_background_tns.txt'
#lc = '../Quartic_Fit_OSW/o4_supernovae/SN_2024phv/sn_2024phv_asas_g.dat'

#bkg = '../Quartic_Fit_OSW/o4_BKG/SN_2023_abdg_background_tns.txt'
#lc = '../Quartic_Fit_OSW/o4_supernovae/SN_2023abdg/sn_2023abdg_asas_g.dat'
#bkg = 'SN_2023_abdg_background_tns.txt'
#lc = 'sn_2023abdg_asas_g.dat'
df_bkg = pd.read_csv(bkg,header=None, names=['MJD','mag'])
df_lc = pd.read_csv(lc)


t = df_lc['MJD']
time = t - np.min(t)
mag = df_lc['mag']

#last_null_detection = np.random.choice(range(800,1000),1)
#selected_points = np.append(last_null_detection,np.random.choice(range(1300,len(time)), 100))
#selected_points = np.random.choice(range(1320,len(time)), 10)
#time_2 = time[selected_points]
#sap_flux_2 = sap_flux[selected_points]
#sap_flux_err_2 = sap_flux_err[selected_points]
#time_SBO=time[1320]
#time_2 -= time_SBO
#print(time_2)

X = np.array(time)
y = np.array(mag)
plt.plot(X, y,'.', color = 'red', label = 'Data Points') 
plt.xlabel('t [days from the discovery]')
plt.ylabel('SAP Flux Normalized (e-/s)')
plt.gca().invert_yaxis()
plt.legend()
plt.grid()
plt.savefig(lc.split('.')[0]+'.png')


class TanhKernel(Kernel):
      def __init__(self,
                  a=1,
                  a_bounds=(1e-5,1e5),
                  b=1,
                  b_bounds=(1e-5,1e5),
                  ):
         self.a = a
         self.a_bounds = a_bounds
         self.b = b
         self.b_bounds = b_bounds

      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("a", "numeric", self.a_bounds)
        
      @property
      def hyperparameter_shifting(self):
         return Hyperparameter("b", "numeric", self.b_bounds)

      def is_stationary(self):
         return False

      def diag(self, X):
         Z = np.einsum('ij,ij->i', X, X)
         K = np.tanh(self.a * Z + self.b)
         return K 

      def __call__(self, X, Y=None, eval_gradient=False):
         X = np.atleast_2d(X)
         if Y is None:
               Z = np.inner(X, X)
               K = np.tanh(self.a * Z + self.b)
         else:
               if eval_gradient:
                  raise ValueError(
                     "Gradient can only be evaluated when Y is None.")
               Z = np.inner(X, Y)    
               K = np.tanh(self.a * Z + self.b)

         if eval_gradient:
               # gradient with respect to ln(a)
            Z = np.inner(X, X)
            if not self.hyperparameter_shifting.fixed:
                  a_gradient = np.empty((K.shape[0], K.shape[1], 1))
                  a_gradient[...,0] = self.a * Z * (1 - K**2)
            else:
                a_gradient = np.empty((X.shape[0], X.shape[1], 0))

            # gradient with respect to ln(b)
            if not self.hyperparameter_scaling.fixed:
                b_gradient = np.empty((K.shape[0], K.shape[1], 1))
                b_gradient[...,0] = self.b * (1 - K**2)
            else:
                b_gradient = np.empty((K.shape[0], K.shape[1], 0))

            K_gradient = np.dstack((a_gradient, b_gradient))
                          
            return K, K_gradient

         else:
               return K


class ExpKernel(Kernel):
      def __init__(self,
                  a=1,
                  a_bounds=(1e-5,1e3),
                  b=1,
                  b_bounds=(1e-5,1e3),
                  ):
         self.a = a
         self.a_bounds = a_bounds
         self.b = b
         self.b_bounds = b_bounds

      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("a", "numeric", self.a_bounds)
        
      @property
      def hyperparameter_shifting(self):
         return Hyperparameter("b", "numeric", self.b_bounds)

      def is_stationary(self):
         return False

      def diag(self, X):
         Z = np.einsum('ij,ij->i', X, X)
         K = np.exp(self.a * Z + self.b)
         return K 

      def __call__(self, X, Y=None, eval_gradient=False):
         X = np.atleast_2d(X)
         if Y is None:
               Z = np.inner(X, X)
               K = np.exp(self.a * Z + self.b)
         else:
               if eval_gradient:
                  raise ValueError(
                     "Gradient can only be evaluated when Y is None.")
               Z = np.inner(X, Y)    
               K = np.exp(self.a * Z + self.b)

         if eval_gradient:
               # gradient with respect to ln(a)
            Z = np.inner(X, X)
            if not self.hyperparameter_scaling.fixed:
                  a_gradient = np.empty((K.shape[0], K.shape[1], 1))
                  a_gradient[...,0] = self.a * Z * K
            else:
                a_gradient = np.empty((X.shape[0], X.shape[1], 0))

            # gradient with respect to ln(b)
            if not self.hyperparameter_shifting.fixed:
                b_gradient = np.empty((K.shape[0], K.shape[1], 1))
                b_gradient[...,0] = self.b * K
            else:
                b_gradient = np.empty((K.shape[0], K.shape[1], 0))

            K_gradient = np.dstack((a_gradient, b_gradient))
                          
            return K, K_gradient

         else:
               return K


class Exp2Kernel(Kernel):
      def __init__(self,
                  a=1,
                  a_bounds=(1e-5,1e3),
                  b=1,
                  b_bounds=(1e-5,1e3),
                  ):
         self.a = a
         self.a_bounds = a_bounds
         self.b = b
         self.b_bounds = b_bounds

      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("a", "numeric", self.a_bounds)
        
      @property
      def hyperparameter_shifting(self):
         return Hyperparameter("b", "numeric", self.b_bounds)

      def is_stationary(self):
         return False

      def diag(self, X):
         Z = np.einsum('ij,ij->i', X, X)
         K = np.exp(self.a**2 * Z**2 + self.b)
         return K 

      def __call__(self, X, Y=None, eval_gradient=False):
         X = np.atleast_2d(X)
         if Y is None:
               Z = np.inner(X, X)
               K = np.exp(self.a**2 * Z**2 + self.b)
         else:
               if eval_gradient:
                  raise ValueError(
                     "Gradient can only be evaluated when Y is None.")
               Z = np.inner(X, Y)    
               K = np.exp(self.a**2 * Z**2 + self.b)

         if eval_gradient:
               # gradient with respect to ln(a)
            Z = np.inner(X, X)
            if not self.hyperparameter_scaling.fixed:
                  a_gradient = np.empty((K.shape[0], K.shape[1], 1))
                  a_gradient[...,0] = 2*self.a**2 * Z**2 * K
            else:
                a_gradient = np.empty((X.shape[0], X.shape[1], 0))

            # gradient with respect to ln(b)
            if not self.hyperparameter_shifting.fixed:
                b_gradient = np.empty((K.shape[0], K.shape[1], 1))
                b_gradient[...,0] = self.b * K
            else:
                b_gradient = np.empty((K.shape[0], K.shape[1], 0))

            K_gradient = np.dstack((a_gradient, b_gradient))
                          
            return K, K_gradient

         else:
               return K


class QuarticKernel(Kernel):
      def __init__(self,
                  a=1,
                  a_bounds=(1e-5, 1e5),
                  b=1,
                  b_bounds=(1e-5, 1e5),
                  c=1,
                  c_bounds=(1e-5, 1e5),
                  d=1,
                  d_bounds=(1e-5, 1e5),
                  e=1,
                  e_bounds=(1e-5, 1e5),
                    
                  ):
         self.a = a
         self.a_bounds = a_bounds
         self.b = b
         self.b_bounds = b_bounds
         self.c = c
         self.c_bounds = c_bounds
         self.d = d
         self.d_bounds = d_bounds
         self.e = e
         self.e_bounds = e_bounds

      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("a", "numeric", self.a_bounds)
        
      @property
      def hyperparameter_shifting(self):
         return Hyperparameter("b", "numeric", self.b_bounds)

      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("c", "numeric", self.c_bounds)
        
      @property
      def hyperparameter_shifting(self):
         return Hyperparameter("d", "numeric", self.d_bounds)

      @property
      def hyperparameter_shifting(self):
         return Hyperparameter("e", "numeric", self.d_bounds)

      def is_stationary(self):
         return False

      def diag(self, X):
         Z = np.einsum('ij,ij->i', X, X)
         K = self.a +self.b * Z + self.c * Z**2 + self.d * Z**3 +self.e * Z**4
         return K 

      def __call__(self, X, Y=None, eval_gradient=False):
         X = np.atleast_2d(X)
         if Y is None:
               Z = np.inner(X, X)
               K = self.a +self.b * Z + self.c * Z**2 + self.d * Z**3 +self.e * Z**4
         else:
               if eval_gradient:
                  raise ValueError(
                     "Gradient can only be evaluated when Y is None.")
               Z = np.inner(X, Y)    
               K = self.a + self.b * Z + self.c * Z**2 + self.d * Z**3 +self.e * Z**4

         if eval_gradient:
               # gradient with respect to ln(a)
            Z = np.inner(X, X)
            if not self.hyperparameter_shifting.fixed:
                  a_gradient = np.empty((K.shape[0], K.shape[1], 1))
                  a_gradient[...,0] = self.a
            else:
                a_gradient = np.empty((X.shape[0], X.shape[1], 0))

            # gradient with respect to ln(b)
            if not self.hyperparameter_scaling.fixed:
                b_gradient = np.empty((K.shape[0], K.shape[1], 1))
                b_gradient[...,0] = self.b * Z
            else:
                b_gradient = np.empty((K.shape[0], K.shape[1], 0))

             # gradient with respect to ln(b)
            if not self.hyperparameter_scaling.fixed:
                c_gradient = np.empty((K.shape[0], K.shape[1], 1))
                c_gradient[...,0] = self.c * Z**2
            else:
                c_gradient = np.empty((K.shape[0], K.shape[1], 0))

             # gradient with respect to ln(c)
            if not self.hyperparameter_scaling.fixed:
                d_gradient = np.empty((K.shape[0], K.shape[1], 1))
                d_gradient[...,0] = self.d * Z**3
            else:
                d_gradient = np.empty((K.shape[0], K.shape[1], 0))

             # gradient with respect to ln(b)
            if not self.hyperparameter_scaling.fixed:
                e_gradient = np.empty((K.shape[0], K.shape[1], 1))
                e_gradient[...,0] = self.e * Z**4
            else:
                e_gradient = np.empty((K.shape[0], K.shape[1], 0))

            K_gradient = np.dstack((a_gradient, b_gradient, c_gradient, d_gradient, e_gradient))
                          
            return K, K_gradient

         else:
               return K    


class PolyKernel(Kernel):
      def __init__(self,
                  p0=1,
                  p0_bounds=(1e-5, 1e5),
                  p1=1,
                  p1_bounds=(1e-5, 1e5),
                  p2=1,
                  p2_bounds=(1e-5, 1e5),
                  p3=1,
                  p3_bounds=(1e-5, 1e5),
                  p4=1,
                  p4_bounds=(1e-5, 1e5),
#                  p5=1,
#                  p5_bounds=(1e-5, 1e5),
#                  p6=1,
#                  p6_bounds=(1e-5, 1e5)                  
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
#         self.p5 = p5
#         self.p5_bounds = p5_bounds
#         self.p6 = p6
#         self.p6_bounds = p6_bounds

      @property
      def hyperparameter_shifting(self):
         return Hyperparameter("p0", "numeric", self.p0_bounds)
        
      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("p1", "numeric", self.p1_bounds)

      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("p2", "numeric", self.p2_bounds)
        
      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("p3", "numeric", self.p3_bounds)

      @property
      def hyperparameter_scaling(self):
         return Hyperparameter("p4", "numeric", self.p4_bounds)

#      @property
#      def hyperparameter_shifting(self):
#         return Hyperparameter("p5", "numeric", self.p5_bounds)

#      @property
#      def hyperparameter_shifting(self):
#         return Hyperparameter("p6", "numeric", self.p6_bounds)

      def is_stationary(self):
         return False

      def diag(self, X):
         Z = np.einsum('ij,ij->i', X, X)
         K = self.p0 +self.p1 * Z + self.p2 * Z**2 + self.p3 * Z**3 +self.p4 * Z**4 #+ self.p5 * Z**5 #+ self.p6 * Z**6
         return K 

      def __call__(self, X, Y=None, eval_gradient=False):
         X = np.atleast_2d(X)
         if Y is None:
               Z = np.inner(X, X)
               K = self.p0 +self.p1 * Z + self.p2 * Z**2 + self.p3 * Z**3 +self.p4 * Z**4 #+ self.p5 * Z**5 #+ self.p6 * Z**6
         else:
               if eval_gradient:
                  raise ValueError(
                     "Gradient can only be evaluated when Y is None.")
               Z = np.inner(X, Y)    
               K = self.p0 +self.p1 * Z + self.p2 * Z**2 + self.p3 * Z**3 +self.p4 * Z**4 #+ self.p5 * Z**5 #+ self.p6 * Z**6

         if eval_gradient:

            Z = np.inner(X, X)
            if not self.hyperparameter_shifting.fixed:
                p0_gradient = np.empty((K.shape[0], K.shape[1], 1))
                p0_gradient[...,0] = self.p0
            else:
                p0_gradient = np.empty((X.shape[0], X.shape[1], 0))

            if not self.hyperparameter_scaling.fixed:
                p1_gradient = np.empty((K.shape[0], K.shape[1], 1))
                p1_gradient[...,0] = self.p1 * Z
            else:
                p1_gradient = np.empty((K.shape[0], K.shape[1], 0))

            if not self.hyperparameter_scaling.fixed:
                p2_gradient = np.empty((K.shape[0], K.shape[1], 1))
                p2_gradient[...,0] = self.p2 * Z**2
            else:
                p2_gradient = np.empty((K.shape[0], K.shape[1], 0))

            if not self.hyperparameter_scaling.fixed:
                p3_gradient = np.empty((K.shape[0], K.shape[1], 1))
                p3_gradient[...,0] = self.p3 * Z**3
            else:
                p3_gradient = np.empty((K.shape[0], K.shape[1], 0))

            if not self.hyperparameter_scaling.fixed:
                p4_gradient = np.empty((K.shape[0], K.shape[1], 1))
                p4_gradient[...,0] = self.p4 * Z**4
            else:
                p4_gradient = np.empty((K.shape[0], K.shape[1], 0))

#            if not self.hyperparameter_scaling.fixed:
#                p5_gradient = np.empty((K.shape[0], K.shape[1], 1))
#                p5_gradient[...,0] = self.p5 * Z**5
#            else:
#                p5_gradient = np.empty((K.shape[0], K.shape[1], 0))

            #if not self.hyperparameter_scaling.fixed:
            #    p6_gradient = np.empty((K.shape[0], K.shape[1], 1))
            #    p6_gradient[...,0] = self.p6 * Z**6
            #else:
            #    p6_gradient = np.empty((K.shape[0], K.shape[1], 0))

            K_gradient = np.dstack((p0_gradient, p1_gradient, p2_gradient, p3_gradient, p4_gradient)) #, p5_gradient)) #, p6_gradient))
            return K, K_gradient

         else:
               return K    

def custom_optimizer(obj_func, initial_theta, bounds):
    def safe_obj(theta):
        out = obj_func(theta)
        # sklearn may return (value, gradient) or arrays; coerce to scalar value
        if isinstance(out, tuple) or isinstance(out, list):
            val = out[0]
        else:
            val = out
        return float(np.asarray(val).ravel()[0])

    def run_minimize(x0):
        res = minimize(safe_obj, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 300})
        return res

    bds = np.array(bounds)
    x0 = np.random.uniform(bds[:,0], bds[:,1])
    res = run_minimize(x0)
    return (res.x, float(res.fun))

def GPR_fit(X,y):
    global gpr, Xp
    Xr = X.reshape(-1, 1)
    Xp = np.linspace(X.min()-20, X.max()+10, 600)
    Xpr = Xp.reshape(-1,1)
    
    scaler = StandardScaler()
    scaler.fit(Xr)
    Xrs = scaler.transform(Xr)
    Xprs = scaler.transform(Xpr)

    kernel = (
        PolyKernel(p0_bounds=(0.00001, 100),p1_bounds=(0.00001, 100), \
                   p2_bounds=(0.00001, 100), p3_bounds=(0.00001, 100), \
                   p4_bounds=(0.00001, 100))
        #QuarticKernel(a_bounds=(0.00001, 100),b_bounds=(0.00001, 100), \
        #              c_bounds=(0.00001, 100), d_bounds=(0.00001, 100), \
        #              e_bounds=(0.00001, 100))
        #+ TanhKernel(a_bounds=(0.00001, 10), b_bounds=(0.00001, 10))
        * ExpKernel(a_bounds=(0.00001, 10), b_bounds=(0.00001, 10))
        #* Exp2Kernel(a_bounds=(0.00001, 100), b_bounds=(0.0001, 100))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e0))   
    )

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                               normalize_y=True,
                               n_restarts_optimizer=10,
                               optimizer=custom_optimizer)#,optimizer=None)
    gpr.fit(Xrs, y.reshape(-1,1))

    yp, ystd = gpr.predict(Xprs, return_std=True)

    print('Optimized parameters:', gpr.kernel_.get_params())

    return Xp, yp, ystd


Xp, yp, ystd = GPR_fit(X,y)


ave_BKG = np.array(df_bkg['mag'])
t_sbo = Xp[np.argmin(np.abs(yp - ave_BKG ))]
t_sbo_l = Xp[np.argmin(np.abs(yp-ystd - ave_BKG ))]
t_sbo_r = Xp[np.argmin(np.abs(yp+ystd - ave_BKG ))]
mag_sbo = yp[np.argmin(np.abs(yp - ave_BKG ))]

print ("The predicted shock break out time is between",t_sbo_l+np.min(t)," and ",t_sbo_r+np.min(t)," in MJD")

plt.figure(figsize=(8,5))
plt.plot(X, y, 'r.', label='Observed data')
plt.axhline(np.array(df_bkg['mag']), color='black', linestyle='-',label="BKG")
plt.plot(Xp, yp, color='blue', alpha=0.25)
plt.fill_between(Xp, yp-ystd, yp+ystd, alpha=0.1,color='b')
#plt.axhline(ave_BKG, color='black', linestyle='-',label="Average BKG") 
#plt.axvline(time_2[0], color='gray', linestyle='--',label="Discorvery date")
#plt.axvline(initial_time, color='cyan', linestyle='--',label="Initial date")
#plt.plot (t_sbo,ave_BKG, marker='*', color="yellow",markersize=15, label="t_SBO="+str(t_sbo.round(2)))
plt.xlabel('Days')
plt.ylabel('mag')
plt.ylim([12,22])
plt.gca().invert_yaxis()
plt.plot(t_sbo,ave_BKG,marker="*",markersize=15,label="t_SBO",color='y')
plt.grid()
plt.legend()
plt.savefig(lc.split('.')[0]+'-fit.png')



