# Rubber band filter: optimal padding without edge artifacts.

The rubber band filter enables band-limited filtering without introducing detrimental edge artifacts.

This technique applies an optimal padding scheme during the filtering process and uses a Conjugate Gradient method to solve for the optimal filtered signal. More details can be found at https://doi.org/10.1364/OL.574053.

##### Calls:

###### matlab version:

 	Vf = Rubber\_Band\_Filter(ts, Vs, freq)

 	\[Vf,info] = Rubber\_Band\_Filter(ts, Vs, freq)

 	Vf = Rubber\_Band\_Filter(ts, Vs, freq, 'exitmode','fixed')

 	Vf = Rubber\_Band\_Filter(ts, Vs, freq, 'exitmode','thresh','tol',1e-9)

 	Vf = Rubber\_Band\_Filter(ts, Vs, freq, 'Niter',10)

###### python version:

        Vf, info               = rubber\_band\_filter(ts, Vs, freq)

        Vf, info               = rubber\_band\_filter(ts, Vs, freq, 'exitmode','fixed')

        Vf, info               = rubber\_band\_filter(ts, Vs, freq, 'exitmode','thresh','tol',1e-9)

        Vf, info               = rubber\_band\_filter(ts, Vs, freq, 'Niter', 10)

##### Input for the function:

###### Required inputs:

 	ts: time axis of the signal to be processed.

 	Vs: signal to be processed.

 	fs: cut-off frequency.

###### Optional inputs:

 	exitmode: fixed or threshold, ‘fixed’ exits the CG loop after all iterations; ‘threshold’ exits once the tolerance is reached.

 	tol     : not used in fixed mode; tolerance value in threshold mode.

 	npad    : number of extra padding regions (default = 1).

 	niter   : number of CG iterations (default = 30).

 	ws      : weights for valid region (default = 1 except 0 where Vs is NaN), only necessary when the default rubber band filter does not give a good result, can be a weight depending on the SNR to provide better filtering results.

##### Output:

 	Vf   : filtered signal.

 	info : useful information including parameters used, convergence status, exit mode, and more.

By default, the padded region uses 0.03 \* tukeywindow(0.2, length(Vs)). The function does not provide an optional input to modify this, but it can be adjusted manually (line 76–77 in MATLAB, or line 103 in Python).

This code is a clean and convenient version for basic use. It can be modified to add features such as using GCV for optimal padding-weight selection, 2D filtering, and bandpass filtering. Examples can be found at https://doi.org/10.6084/m9.figshare.30290425.

