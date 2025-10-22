# Rubber band filter: optimal padding without edge artifacts.

The rubber band filter enables band-limited filtering without introducing detrimental edge artifacts.

This technique applies an optimal padding scheme during the filtering process and uses a Conjugate Gradient method to solve for the optimal filtered signal. More details can be found at https://doi.org/10.1364/OL.574053.

##### Calls:

###### matlab version:

&nbsp;	Vf = Rubber\_Band\_Filter(ts, Vs, freq)

&nbsp;	\[Vf,info] = Rubber\_Band\_Filter(ts, Vs, freq)

&nbsp;	Vf = Rubber\_Band\_Filter(ts, Vs, freq, 'exitmode','fixed')

&nbsp;	Vf = Rubber\_Band\_Filter(ts, Vs, freq, 'exitmode','thresh','tol',1e-9)

&nbsp;	Vf = Rubber\_Band\_Filter(ts, Vs, freq, 'Niter',10)

###### python version:

&nbsp;       Vf, info               = rubber\_band\_filter(ts, Vs, freq)

&nbsp;       Vf, info               = rubber\_band\_filter(ts, Vs, freq, 'exitmode','fixed')

&nbsp;       Vf, info               = rubber\_band\_filter(ts, Vs, freq, 'exitmode','thresh','tol',1e-9)

&nbsp;       Vf, info               = rubber\_band\_filter(ts, Vs, freq, 'Niter', 10)

##### Input for the function:

###### Required inputs:

&nbsp;	ts: time axis of the signal to be processed.

&nbsp;	Vs: signal to be processed.

&nbsp;	fs: cut off frequency

###### Optional inputs:

&nbsp;	exitmode: fixed or threshold, fixed means exit the CG loop after all iterations, threshold means exit the CG loop once the tolerance is reached. 

&nbsp;	tol     : none for fixed mode or tolerance for threshold mode

&nbsp;	npad    : number of extra padding regions (default = 1)

&nbsp;	niter   : number of CG iterations (default = 30)

&nbsp;	ws      : weights for valid region (default = 1 except 0 where Vs is NaN), only necessary when the rubber band filter can not give a good result, can be a weight depended on the SNR to provide better filtering results.

##### Output:

&nbsp;	Vf   : filtered signal

&nbsp;	info : useful information including parameters used, convergence status, exit mode, and more.

By default, the padded region uses 0.03 \* tukeywindow(0.2, length(Vs)). The function does not provide an optional input to modify this, but it can be adjusted manually (line 76–77 in MATLAB, or line 103 in Python).

This code is a clean and convenient version for basic use. It can be modified to add features such as using GCV for optimal padding-weight selection, 2D filtering, and bandpass filtering. Examples can be found at https://doi.org/10.6084/m9.figshare.30290425.

