# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:24:29 2021

@author: Lisa
"""

import numpy as np
from scipy.optimize import curve_fit

DEG_TO_RAD = np.pi/180.
RAD_TO_DEG = 180./np.pi
DEG_TO_ASEC = 3600
ASEC_TO_DEG = 1/3600
RAD_TO_ASEC = RAD_TO_DEG * DEG_TO_ASEC
ASEC_TO_RAD = 1 / RAD_TO_ASEC

#Converts the angle (in rad) from the interval [-pi, pi] to the interval [0, 2pi]
def convertAngle(angle):
  if (angle < 0):
    return 2*np.pi + angle 
  else:
    return angle

#Definition of a gaussian function. N: normalization, mu: mean of the gaussian,
#sigma: standard deviation of the gaussian
def gaussian(x, N, mu, sigma):
  return N * np.exp(- 0.5 * ((x - mu)/sigma)**2 )

def twoPeaks(x, N1, mu1, sigma1, N2, mu2, sigma2 ):
  return gaussian(x, N1, mu1, sigma1) + gaussian(x, N2, mu2, sigma2)

def threePeaks(x, N1, mu1, sigma1, N2, mu2, sigma2, N3, mu3, sigma3):
  return gaussian(x, N1, mu1, sigma1) + gaussian(x, N2, mu2, sigma2) + gaussian(x, N3, mu3, sigma3) 

#Modulation curve for the polarized data. a0: normalization, a1: modulation factor,
#a2: polarization angle. The curve has pi periodicity.
def fitSplinePol(phi, a0, a1, a2 ):
  return a0*(1 + a1*np.cos(2 * (phi - a2 + np.pi/2)))

#Modulation curve for the unpolarized data. a0: normalization, a1: modulation factor,
#a2: polarization angle. the curve has pi/2 periodicity.
def fitSplineUnpol(phi, a0, a1, a2 ):
  return a0*(1 + a1*np.cos(4 * (phi - a2 + np.pi/2)))

#Fits the data using a non-linear least square algorithm and saves the fit plot.
#The function takes the following input:
#   -fitfunc: fit model function
#   -xdata, ydata: x and y data per the fit
#   -yerrors: the errors on the y data points
#   -savename: the path at which the plot must be saved
def PolFitOLS(fitfunc, xdata, ydata, yerrors):
    #yerrors = [np.sqrt(y) for y in ydata]
    
    #Starting points (init_vals) and bounds (bd) for the fit parameters
    #init_vals = [1000, 1, 1]
    init_vals = [np.max(ydata), 1, 1]
    bd = ([0,0,0],[np.inf, 1, np.pi])
    
    #Fits the data
    best_vals, covar = curve_fit(f = fitfunc, xdata = xdata, ydata = ydata, p0=init_vals, bounds = bd)
    #Evaluate the error on the fit parameters using the covariance matrix and the chi squared
    #of the fit
    error_vals = np.sqrt(np.diag(covar))
    
    #Plots data points, fits the curve and evaluates chi squared of the fit
    x_fit = np.arange(min(xdata), max(xdata), 0.1)
    y_fit = [fitfunc(phi,best_vals[0], best_vals[1], best_vals[2]) for phi in xdata]
    chi_squared = sum([((ydata[i] - y_fit[i])**2)/yerrors[i]**2 for i in range(len(y_fit))])
    y_fit = [fitfunc(phi,best_vals[0], best_vals[1], best_vals[2]) for phi in x_fit]
    
    return best_vals, error_vals, x_fit, y_fit, chi_squared

#Fits the data using a residual resampling bootstrap algorithm and saves the fit plot.
#The function takes the following input:
#   -fitfunc: fit model function
#   -xdata, ydata: x and y data per the fit
#   -yerrors: the errors on the y data points
#   -savename: the path at which the plot must be saved
#A residual bootstrap algorithm works in the following way:
#   1 - Data are fitted according to a regular least-squares algorithm and residuals
#       and predicted values are evaluated
#   2 - New samples of the data are defined by adding a random selected (with replacement) residual
#       to each expected value
#   3 - The new bootstrap samples are fitted to obtain a distribution of the fit parameters
#   4 - The distribution of the fit parameters can be used for statystical analysis 
#More on the techinique here: https://blogs.sas.com/content/iml/2018/10/29/bootstrap-regression-residual-resampling.html
# def PolFitResidualResampling(fitfunc, xdata, ydata, yerrors, savename):
#     #Defines Starting points (init_vals) and bounds (bd) for the fit parameters and
#     #makes a first NLS fit of the data
#     init_vals = [1000, 1, 1]
#     bd = ([0,0,0],[np.inf, 1,np.pi])  
#     best_vals, covar = curve_fit(f = fitfunc, xdata = xdata, ydata = ydata, p0=init_vals, bounds = bd)

#     #Evaluate the predicted value of the fit and the residual
#     ypredicted = [fitfunc(phi,best_vals[0], best_vals[1], best_vals[2] ) for phi in xdata ]
#     yresidual = [ypredicted[i] - ydata[i] for i in range(len(ydata))]
    
#     #Number of bootstrap samples
#     bootnum = 100
#     bootsample = []
#     #Array containing the fit results of the bootstrap sample.
#     bootNorm = []
#     bootQ = []
#     bootPhi = []

#     for i in range(bootnum):
#         bootsample.append([])
#         #Create the bootstrap sample by adding a randomly selected residual to the predicted values
#         for j in range(len(xdata)):
#             bootsample[i].append(ypredicted[j] + yresidual[rd.randint(0,len(yresidual)-1)])
#         #Fits the bootstrap sample
#         res, cov = curve_fit(f = fitfunc, xdata = xdata, ydata = bootsample[i], p0=init_vals, bounds = bd)
#         bootNorm.append(res[0])
#         bootQ.append(res[1])
#         bootPhi.append(res[2])
    
#     #Evaluates the best fit parameters and the errors by evaluateding the mean and
#     #the standard deviation of the distribution
#     best_vals = [np.mean(bootNorm), np.mean(bootQ), np.mean(bootPhi)]
#     error_vals = [np.std(bootNorm), np.std(bootQ), np.std(bootPhi)]
    
#     #Plots data points, fits the curve and evaluates chi squared of the fit
#     y_fit = [fitfunc(phi,best_vals[0], best_vals[1], best_vals[2]) for phi in xdata]
#     chi_squared = sum([((ydata[i] - y_fit[i])**2)/yerrors[i]**2 for i in range(len(y_fit))])
#     plt.figure()
#     plt.title('Reduced Chi-Squared: {0:.2f}'.format(chi_squared/(len(xdata) - 3)))
#     plt.scatter(xdata, ydata, color = 'b')
#     plt.errorbar(xdata, ydata, yerr = yerrors, linestyle = "None", color = 'b')
#     plt.plot(xdata,y_fit, color = '#000000')
#     plt.savefig(savename)
    
#     plt.close()
    
#     return best_vals, error_vals
    
# def PolFitCaseResampling(xdata, ydata, savename):
#     yerrors = [np.sqrt(y) for y in ydata]
#     init_vals = [1000, 1, 1]
#     bd = ([0,0,0],[np.inf, 1,np.pi])
        
#     best_vals, covar = curve_fit(f = fitSplinePol, xdata = xdata, ydata = ydata, p0=init_vals, bounds = bd)

#     ypredicted = [fitSplinePol(phi,best_vals[0], best_vals[1], best_vals[2] ) for phi in xdata ]
#     yresidual = [ypredicted[i] - ydata[i] for i in range(len(ydata))]
    
#     plt.figure()
#     plt.scatter(ypredicted, yresidual)
    
#     plt.figure()
#     plt.hist(yresidual, bins = 4)
    
    
#     nBoot = 30000
#     bootsample = []
#     bootNorm = []
#     bootQ = []
#     bootPhi = []

#     for i in range(nBoot):
#         xSamples = []
#         ySamples = []
#         for j in range(len(xdata)):
#             ind = rd.randint(0, len(xdata) - 1) 
#             xSamples.append(xdata[ind])
#             ySamples.append(ydata[ind])
            
#         res, cov = curve_fit(f = fitSplinePol, xdata = xSamples, ydata = ySamples, p0=init_vals, bounds = bd)
    
#         bootNorm.append(res[0])
#         bootQ.append(res[1])
#         bootPhi.append(res[2])
        
#     plt.figure()
#     plt.plot(bootNorm)
    
#     plt.figure()
#     plt.plot(bootQ)
    
#     plt.figure()
#     plt.plot(bootPhi)
    
#     plt.figure()
#     plt.scatter(bootQ, bootPhi)

#     best_vals = [np.mean(bootNorm), np.mean(bootQ), np.mean(bootPhi)]
#     error_vals = [np.std(bootNorm), np.std(bootQ), np.std(bootPhi)]

#     y_fit = [fitSplinePol(phi,best_vals[0], best_vals[1], best_vals[2]) for phi in xdata]
#     #fit_error_one_sigma = np.sqrt(np.diag(covar))
  
#     chi_squared = sum([((ydata[i] - y_fit[i])**2)/y_fit[i] for i in range(len(y_fit))])
    
#     print(chi_squared)
#     print(chi_squared/(len(y_fit)-3))
  
#     plt.figure()
#     #plt.scatter(angle_fit, corrected_val)
#     #plt.plot(angle_fit,y_fit)
#     plt.scatter(xdata, ydata, color = 'b')
#     plt.errorbar(xdata, ydata, yerr = yerrors, linestyle = "None", color = 'b')
#     plt.plot(xdata,y_fit, color = '#000000')
#     plt.savefig(savename)
    
#     return best_vals, error_vals  