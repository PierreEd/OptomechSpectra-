# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:14:08 2021

@author: jacqu
"""
import numpy as np 
from .constants import *

# - - - - - - - - - - - - - - - - - - - - - - - -

def Xi(m , omega , omega_m , Q):
    gamma_m = omega_m/Q
    return 1 / (m*(omega_m**2 - omega**2 - 1j*gamma_m*omega))

# - - - - - - - - - - - - - - - - - - - - - - -

def transfer(mat , transfo):
    
    transfer_mat = transfo.dot(mat).dot(np.transpose(np.conjugate(transfo)))
    
    return transfer_mat 

# - - - - - - - - - - - - - - - - - - - - - - - -

def rot( theta ):
    
    return np.array([[np.cos(theta) , -np.sin(theta)],[np.sin(theta) , np.cos(theta)]])

# - - - - - - - - - - - - - - - - - - - - - - - -

'''
def variance(cov, quadrature_angle = 0, dB = True):
    cov_rot = transfer(cov , rot(quadrature_angle))
    var = np.real(cov_rot[0, 0])
    if dB:
        return  10 * np.log10(var)
    else:
        return var

# - - - - - - - - - - - - - - - - - - - - - - - -

def Sxx( cov , omega , theta , intensity_input):
    
        return lambda_carrier / (256 * finesse**2 * intensity_input) * variance(cov , theta, False) * (1 + (omega/omega_m)**2) 
''' 