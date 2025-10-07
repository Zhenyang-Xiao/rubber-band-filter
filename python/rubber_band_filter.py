# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:14:09 2025

@author: xiaoz
"""

import numpy as np
import warnings

class ThresholdNotReachedWarning(UserWarning):
    """CG hit Niter before meeting tol in 'thresh' mode."""
    pass

def rubber_band_filter(ts, Vs, freq, *args):
    """
    Rubber_Band_Filter
        Low-pass filtering without edge artifacts via optimal padding
        solved by Conjugate Gradient (CG).

    Examples:
        Vf, info               = rubber_band_filter(ts, Vs, freq)
        Vf, info               = rubber_band_filter(ts, Vs, freq, 'exitmode','fixed')
        Vf, info               = rubber_band_filter(ts, Vs, freq, 'exitmode','thresh','tol',1e-9)
        Vf, info               = rubber_band_filter(ts, Vs, freq, 'Niter', 10)

    Required:
        ts    : 1D array of times (monotonic, ~uniform)
        Vs    : 1D array of signal (same length as ts)
        freq  : cutoff frequency (Hz)
        
    Optional name–value parameters (MATLAB-style):
        'exitmode' : 'fixed' or 'thresh' (default 'thresh')
        'tol'      : tolerance for 'thresh' (default 1e-7; ignored for 'fixed')
        'npad'     : padding count (default 1; right-side mirror)
        'niter'    : CG iterations (default 30)
        'ws'       : weights vector for valid region (default ones; 0 where Vs is NaN)

    Returns
    -------
    Vf : filtered signal (same length as Vs)
    info : dict with diagnostics (iterations, dv, err, Vf_pad, etc.)
    """
    ts = np.asarray(ts).reshape(-1)
    Vs = np.asarray(Vs).reshape(-1)
    assert ts.size == Vs.size, "ts and Vs must have the same length"

    # ---- defaults ----
    exitMode = 'thresh'
    tol      = 1e-7
    Npad     = 1
    Niter    = 30
    ws       = np.ones_like(Vs, dtype=float)

    i = 0
    while i < len(args):
        key = args[i]
        if not isinstance(key, (str, bytes)):
            raise ValueError(f"Unexpected non-string option at position {i+1}: {key!r}")
        key_norm = key.strip().lower()
       # ensure a value exists
        if i + 1 >= len(args):
            raise ValueError(f'Option "{key}" is missing a value.')
        val = args[i + 1]
        if key_norm == 'exitmode':
            s = str(val).strip().lower()
            if s in ('fixed', 'fix', 'full'):
                exitMode = 'fixed'
            elif s in ('thresh', 'threshold', 'resid', 'residual'):
                exitMode = 'thresh'
            else:
                raise ValueError(f'Unknown exitmode "{val}". Use "fixed" or "thresh".')
        elif key_norm == 'tol':
                tol = float(val)

        elif key_norm == 'npad':
                Npad = int(val)

        elif key_norm == 'niter':
                Niter = int(val)
        elif key_norm == 'ws':
                ws = np.asarray(val, float).reshape(-1)
                if ws.shape != Vs.shape:
                    raise ValueError("ws must have the same length as Vs")
        else:
                raise ValueError(f'Unknown option "{key}".')
        i += 2

    # ---- handle NaNs: deweight and linearly interpolate with extrapolation ----
    if ws.shape != Vs.shape:
        raise ValueError("ws must match Vs length if provided.")

    valid = ~np.isnan(Vs)
    ws = ws * valid.astype(float)
    if not np.all(valid):
        Vs = _interp_linear_extrap(ts, ts[valid], Vs[valid])

    # ---- inline padding ----
    Vso = Vs.copy()
    if Npad == 0:
        wk = ws
    elif Npad == 1:
        wsh = 0.03 * _tukeywin(ws.size, 0.2)
        Vs  = np.concatenate([Vso, Vso[::-1]])
        wk  = np.concatenate([ws,  wsh])
    else:
        hn  = np.hanning(2*Vso.size)[:Vso.size]
        Vs  = np.concatenate([Vso, (Vso*hn)[::-1], np.zeros(Vso.size*(Npad-2)), (Vso[::-1])*hn])
        wk  = np.concatenate([ws, np.zeros(ws.size*Npad)])

    # ---- frequency grid & passband (odd length) ----
    dt  = float(np.mean(np.diff(ts)))
    N   = Vs.size
    fss = np.fft.fftshift(np.fft.fftfreq(N, d=dt))  # centered frequencies
    zi_arr = np.where(fss == 0)[0]
    if zi_arr.size == 0:
        raise RuntimeError("Zero frequency not found in grid.")
    zi = int(zi_arr[0])

    g = np.where(np.abs(fss) <= freq)[0]
    xtra = 0.0
    epsf = np.finfo(float).eps * max(1.0, abs(freq))
    while g.size % 2 == 0:
        xtra += epsf
        g = np.where(np.abs(fss) <= (freq + xtra))[0]

    # ---- precompute transforms ----
    W = np.fft.fftshift(np.fft.fft(wk.astype(np.complex128)))
    P = np.fft.fftshift(np.fft.fft(Vs.astype(np.complex128)))


    # ---- conv over filtered region ----
    def convf(xi):
        xi = np.asarray(xi)
        if xi.size == g.size:
            zix = int(np.where(fss[g] == 0)[0][0])
        else:
            zix = int(np.where(fss == 0)[0][0])

        nconv = xi.size + W.size - 1
        co_full = np.fft.ifft(np.fft.fft(xi, nconv) * np.fft.fft(W, nconv))

        ko = zi + zix  # zero-based alignment index in the convolution result
        half = (g.size - 1) // 2
        start = ko - half
        stop  = ko + half + 1
        return co_full[start:stop]

    # ---- CG ----
    b = convf(P)
    x0 = P[g].copy(); x = x0            # keep intent
    np.random.seed(0); x0 = np.zeros_like(P[g]); x = x0
    r = b - convf(x)
    p = r.copy()
    tiny   = np.finfo(float).tiny
    bnorm2 = float(np.real(np.vdot(b, b)))


    dv = np.zeros(Niter, dtype=float)   # normalized residual^2
    for ii in range(Niter):
        Ap = convf(p)
        denom = np.vdot(p, Ap)
        if np.abs(denom) <= tiny:
            denom = tiny + 0j
        ak = np.real(np.vdot(r, r) / denom).item()   # exact same semantics as float(...) w/o warning
        xo = x.copy()
        x  = x + ak * p
        ro = r.copy()
        r  = r - ak * Ap
        den_ro = np.vdot(ro, ro)
        if np.abs(den_ro) <= tiny:
            den_ro = tiny + 0j
        bk = np.real(np.vdot(r, r) / den_ro).item()  # same here
        # safeguard
        if np.sum(wk * np.abs(_ToV(x, fss, g) - Vs)**2) > np.sum(wk * np.abs(_ToV(xo, fss, g) - Vs)**2):
            bk = 0.0
        p = r + bk * p

        dv[ii] = np.real(np.vdot(r, r)) / (bnorm2 if bnorm2 > tiny else tiny)
        if exitMode.lower() == 'thresh' and dv[ii] <= tol:
            dv = dv[:ii+1]
            break
    used_iter = min(ii+1, Niter)

    # ---- outputs ----
    Vf_pad = _ToV(x, fss, g)
    if np.isrealobj(Vs):
        Vf_pad = np.real(Vf_pad)
    Vf = Vf_pad[:Vso.size]
    final_dv  = float(dv[-1]) if len(dv) else float("nan")
    converged = True

    if exitMode == 'thresh':
        if used_iter == Niter and final_dv > tol:
            converged = False
            warnings.warn(
                (f"Rubber_Band_Filter:ThresholdNotReached | "
                 f"CG hit Niter={Niter} before meeting tol={tol:.3g} "
                 f"(final dv={final_dv:.3g}). Consider increasing Niter "
                 f"or relaxing tol."),
            ThresholdNotReachedWarning,
            stacklevel=2
        )

    info = {
        'dt': dt,
        'N_original': Vso.size,
        'N_padded': Vs.size,
        'cutoff_used': float(freq + xtra),
        'passband_len': g.size,
        'iterations_requested': Niter,
        'iterations_used': used_iter,
        'exitMode': exitMode,
        'tol': tol,
        'dv': dv,
        'final_dv': final_dv,      
        "converged": converged, 
        'err': float(np.sum(wk * np.abs(_ToV(x, fss, g) - Vs)**2)),
        'Vf_pad': Vf_pad
    }
    return Vf, info


# ---------------- helpers ----------------

def _ToV(xin, fss, g):
    Vspec = np.zeros_like(fss, dtype=complex)
    Vspec[g] = xin
    return np.fft.ifft(np.fft.ifftshift(Vspec))

def _is_exitmode(x):
    if isinstance(x, (str, bytes)):
        s = str(x).strip().lower()
        # accept abbreviations/synonyms
        fixed = ('fix', 'fixed', 'full', 'niter', 'maxiter')
        thresh = ('thr', 'thresh', 'threshold', 'tol', 'tolerance', 'resid', 'residual')
        return s in fixed or s in thresh
    return False

def _canon_exitmode(x):
    s = str(x).strip().lower()
    if s.startswith(('fix', 'full', 'niter', 'maxiter')):
        return 'fixed'
    return 'thresh'

def _tukeywin(M, alpha=0.5):
    # SciPy-free Tukey window
    if M <= 0:
        return np.array([])
    if alpha <= 0:
        return np.ones(M)
    if alpha >= 1:
        return np.hanning(M)
    n = np.arange(M, dtype=float)
    w = np.ones(M, dtype=float)
    edge = int(np.floor(alpha*(M-1)/2.0))
    # first taper
    if edge > 0:
        w[:edge+1] = 0.5*(1 + np.cos(np.pi*(2*(n[:edge+1])/(alpha*(M-1)) - 1)))
    # last taper
    if edge > 0:
        w[-(edge+1):] = 0.5*(1 + np.cos(np.pi*(2*(1 - n[-(edge+1):]/(M-1))/alpha - 1)))
    return w

def _interp_linear_extrap(x, xp, fp):
    # piecewise linear with linear extrapolation at ends
    x = np.asarray(x, float); xp = np.asarray(xp, float); fp = np.asarray(fp, float)
    order = np.argsort(xp)
    xp = xp[order]; fp = fp[order]
    y = np.interp(x, xp, fp)  # inside range
    left_mask  = x < xp[0]
    right_mask = x > xp[-1]
    if np.any(left_mask):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[left_mask] = fp[0] + slope*(x[left_mask] - xp[0])
    if np.any(right_mask):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[right_mask] = fp[-1] + slope*(x[right_mask] - xp[-1])
    return y

