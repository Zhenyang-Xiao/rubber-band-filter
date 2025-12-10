# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:17:22 2025

@author: xiaoz
"""
import matplotlib.pyplot as plt
import scipy.io as sio
#import numpy as np
from rubber_band_filter import rubber_band_filter 
#from rubber_band_filter_gcv import rubber_band_filter_gcv 
# # fake data
# np.random.seed(0) 
# ts = np.linspace(0, 1, 5000)
# Vs = np.cos(2*np.pi*20*ts) + 0.2*np.random.randn(ts.size)
# fc = 30.0  # Hz
# real data
tdn = sio.loadmat("example data/TDS_data.mat")["tdn"]   # shape (N,2)
ts, Vs = tdn[:,0].astype(float), tdn[:,1].astype(float)
fc = 4.5  # THz

Vf, info = rubber_band_filter(ts, Vs, fc,'tol', 1e-11,'niter',50)  # default behavior
#Vf1, info1 = rubber_band_filter_gcv(ts, Vs, fc,'tol', 1e-11,'niter',50)  # default behavior
print(info['iterations_used'], info['dv'])
#print(info1['iterations_used'], info1['dv'])

plt.plot(ts, Vs, label='Original')
plt.plot(ts, Vf, label='Filtered')
#plt.plot(ts, Vf1, label='GCV')
plt.legend(); plt.show()