# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:17:22 2025

@author: xiaoz
"""
import matplotlib.pyplot as plt
#import scipy.io as sio
import numpy as np
from rubber_band_filter import rubber_band_filter 
#from rubber_band_filter_gcv import rubber_band_filter_gcv 
# # fake data
np.random.seed(0) 
dt = 0.01
t = np.arange(0, 10 + dt, dt)[:, None]   
V = 2 * t + np.random.randn(*t.shape) * 0.2 + np.sin(2 * np.pi * 1.5 * t + 3 * np.pi / 8)
fc = 2

Vf, info = rubber_band_filter(t, V, fc,'tol', 1e-7,'niter',80)  
print(info['iterations_used'], info['dv'])
#print(info1['iterations_used'], info1['dv'])

plt.plot(t, V, label='Original')
plt.plot(t, Vf, label='Filtered')
#plt.plot(ts, Vf1, label='GCV')
plt.legend(); plt.show()