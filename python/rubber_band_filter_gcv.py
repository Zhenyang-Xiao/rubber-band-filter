# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:57:35 2025

@author: xiaoz
"""

import numpy as np
import warnings


class ThresholdNotReachedWarning(UserWarning):
    """CG hit Niter before meeting tol in 'thresh' mode."""
    pass


def rubber_band_filter_gcv(ts, Vs, freq, *args):
    """
    Rubber_Band_Filter_GCV (Python)
    Low-pass filtering without edge artifacts; pad-weight shape (alpha, r) is
    chosen by GCV. Solved by Conjugate Gradient (CG).

    Examples
    --------
    Vf, info = rubber_band_filter_gcv(ts, Vs, fc)
    Vf, info = rubber_band_filter_gcv(ts, Vs, fc, 'exitmode','fixed')
    Vf, info = rubber_band_filter_gcv(ts, Vs, fc, 'exitmode','thresh','tol',1e-9)
    Vf, info = rubber_band_filter_gcv(ts, Vs, fc, 'Niter',50, 'Npad',1)

    Name–value options (case-insensitive)
    -------------------------------------
    'exitmode' : 'thresh' (default) or 'fixed'
    'tol'      : tolerance for 'thresh' (default 1e-7)
    'Npad'     : padding count (default 1; right-side mirror)
    'Niter'    : CG iterations per solve (default 30)
    'ws'       : weights on valid region (default ones; 0 where Vs is NaN)
    """
    # --- required inputs ---
    ts = np.asarray(ts).reshape(-1)
    Vs = np.asarray(Vs).reshape(-1)
    if ts.size != Vs.size:
        raise ValueError("ts and Vs must have the same length")

    # --- defaults ---
    exitMode = 'thresh'
    tol      = 1e-7
    Npad     = 1
    Niter    = 30
    ws       = np.ones_like(Vs, dtype=float)

    # --- MATLAB-style name–value scan (key at i, value at i+1) ---
    i = 0
    while i < len(args):
        key = args[i]
        if not isinstance(key, (str, bytes)):
            raise ValueError(f"Unexpected non-string option at position {i+1}: {key!r}")
        if i + 1 >= len(args):
            raise ValueError(f'Option "{key}" is missing a value.')
        val = args[i + 1]
        k = key.strip().lower()

        if k == 'exitmode':
            s = str(val).strip().lower()
            if s in ('fixed', 'fix', 'full'):
                exitMode = 'fixed'
            elif s in ('thresh', 'threshold', 'resid', 'residual'):
                exitMode = 'thresh'
            else:
                raise ValueError(f'Unknown exitmode "{val}". Use "fixed" or "thresh".')
        elif k == 'tol':
            tol = float(val)
        elif k == 'npad':
            Npad = int(val)
        elif k == 'niter':
            Niter = int(val)
        elif k == 'ws':
            ws = np.asarray(val, float).reshape(-1)
            if ws.shape != Vs.shape:
                raise ValueError("ws must have the same length as Vs")
        else:
            raise ValueError(f'Unknown option "{key}".')
        i += 2

    if Niter < 1:
        raise ValueError("Niter must be >= 1")
    if Npad < 0:
        raise ValueError("Npad must be >= 0")

    # -------- NaN handling: deweight & interpolate over Vs --------
    valid = ~np.isnan(Vs)
    ws = ws * valid.astype(float)
    if not np.all(valid):
        Vs = _interp_linear_extrap(ts, ts[valid], Vs[valid])

    # -------- padding (signal only; weights decided per solve) --------
    Vso = Vs.copy()
    if Npad == 0:
        Vs_pad = Vso
    elif Npad == 1:
        Vs_pad = np.concatenate([Vso, Vso[::-1]])
    else:
        hn = np.hanning(2 * Vso.size)[:Vso.size]
        Vs_pad = np.concatenate([Vso, (Vso * hn)[::-1], np.zeros(Vso.size * (Npad - 2)), (Vso[::-1]) * hn])
    Vs = Vs_pad  # use padded signal below

    # -------- frequency grid & odd-length passband --------
    dt  = float(np.mean(np.diff(ts)))
    N   = Vs.size
    fss = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    zidx = np.where(fss == 0)[0]
    if zidx.size == 0:
        raise RuntimeError("Zero frequency not found in grid.")
    zi = int(zidx[0])

    g = np.where(np.abs(fss) <= float(freq))[0]
    xtra = 0.0
    epsf = np.finfo(float).eps * max(1.0, abs(float(freq)))
    while g.size % 2 == 0:
        xtra += epsf
        g = np.where(np.abs(fss) <= (float(freq) + xtra))[0]

    # precompute spectrum of the (current) padded signal
    P = np.fft.fftshift(np.fft.fft(Vs.astype(np.complex128)))

    # -------- pad-weight shape parameters (to be chosen by GCV) --------
    alpha0 = 0.03   # baseline scale
    r0     = 0.20   # baseline Tukey ratio
    shape_len = ws.size

    def make_shape(r):
        r = max(0.0, min(1.0, float(r)))
        return _tukeywin(shape_len, r)

    # -------- local solver: CG with current P, (alpha, r) -> Vf_full/Vf_obs --------
    def local_solve(alpha, r):
        shape = make_shape(r)
        if Npad == 0:
            wk = ws
        elif Npad == 1:
            wk = np.concatenate([ws, alpha * shape])
        else:
            wk = np.concatenate([ws, np.zeros(ws.size * Npad)])

        W = np.fft.fftshift(np.fft.fft(wk.astype(np.complex128)))

        def convf(xi):
            xi = np.asarray(xi)
            if xi.size == g.size:
                zix = int(np.where(fss[g] == 0)[0][0])
            else:
                zix = int(np.where(fss == 0)[0][0])
            nconv = xi.size + W.size - 1
            co_full = np.fft.ifft(np.fft.fft(xi, nconv) * np.fft.fft(W, nconv))
            half = (g.size - 1) // 2
            ko = zi + zix
            return co_full[ko - half : ko + half + 1]

        # CG solve (same method as base filter)
        b = convf(P)
        x = np.zeros_like(P[g])
        rcg = b - convf(x)
        p = rcg.copy()
        tiny   = np.finfo(float).tiny
        bnorm2 = float(np.real(np.vdot(b, b)))

        dv = np.zeros(Niter, float)
        for ii in range(Niter):
            Ap = convf(p)
            denom = np.vdot(p, Ap)
            if np.abs(denom) <= tiny:
                denom = tiny + 0j
            ak = (np.vdot(rcg, rcg) / denom).real.item()
            xo = x.copy()
            x  = x + ak * p
            ro = rcg.copy()
            rcg = rcg - ak * Ap
            den_ro = np.vdot(ro, ro)
            if np.abs(den_ro) <= tiny:
                den_ro = tiny + 0j
            bk = (np.vdot(rcg, rcg) / den_ro).real.item()

            # monotonicity safeguard in time domain
            if np.sum(wk * np.abs(_ToV(x, fss, g) - Vs)**2) > np.sum(wk * np.abs(_ToV(xo, fss, g) - Vs)**2):
                bk = 0.0
            p = rcg + bk * p

            dv[ii] = float(np.real(np.vdot(rcg, rcg)) / (bnorm2 if bnorm2 > tiny else tiny))
            if exitMode == 'thresh' and dv[ii] <= tol:
                dv = dv[:ii+1]
                break
        used_iter = min(ii + 1, Niter)

        Vf_full = _ToV(x, fss, g)
        if np.isrealobj(Vs):
            Vf_full = np.real(Vf_full)
        Vf_obs = Vf_full[:Vso.size]
        return Vf_full, Vf_obs, wk, dv, used_iter

    # -------- GCV: coarse sweep --------
    rng = np.random.default_rng(0)  # reproducible Hutchinson probes
    y_obs = Vso.astype(float).reshape(-1)
    nobs  = y_obs.size

    alphas0 = np.logspace(np.log10(alpha0/10.0), np.log10(alpha0*10.0), 15)
    rs0     = np.linspace(max(0.0, r0-0.5), min(1.0, r0+0.5), 15)
    mprobe0 = 7

    gcv_best = np.inf
    alpha_star, r_star = alpha0, r0

    for r_try in rs0:
        for a_try in alphas0:
            _, yhat_obs, _, _, _ = local_solve(a_try, r_try)
            res = y_obs - np.asarray(yhat_obs, float).reshape(-1)
            num = float(np.sum(ws * np.abs(res)**2))

            df = 0.0
            for _ in range(mprobe0):
                z = np.sign(rng.standard_normal(nobs)).astype(float)
                if Npad == 0:
                    Zs_pad = z
                elif Npad == 1:
                    Zs_pad = np.concatenate([z, z[::-1]])
                else:
                    hn = np.hanning(2 * nobs)[:nobs]
                    Zs_pad = np.concatenate([z, (z * hn)[::-1], np.zeros(nobs * (Npad - 2)), (z[::-1]) * hn])
                P_backup = P
                P = np.fft.fftshift(np.fft.fft(Zs_pad.astype(np.complex128)))
                _, zhat_obs, _, _, _ = local_solve(a_try, r_try)  # S_theta z
                P = P_backup
                df += float(np.real(np.vdot(z, np.asarray(zhat_obs).reshape(-1))))
            df /= mprobe0

            denom = max((nobs - df)**2, np.finfo(float).eps)
            gcv = num / denom
            if gcv < gcv_best:
                gcv_best = gcv
                alpha_star, r_star = float(a_try), float(r_try)

    # -------- refine window --------
    on_alpha_edge = (alpha_star <= alphas0.min()+np.finfo(float).eps) or (alpha_star >= alphas0.max()-np.finfo(float).eps)
    on_r_edge     = (r_star     <= rs0.min()+np.finfo(float).eps)     or (r_star     >= rs0.max()-np.finfo(float).eps)
    if on_alpha_edge:
        alpha_lo, alpha_hi = alpha_star/6.0, alpha_star*6.0
    else:
        alpha_lo, alpha_hi = alpha_star/3.0, alpha_star*3.0
    r_halfspan = 0.30 if on_r_edge else 0.15
    r_lo = max(0.0, r_star - r_halfspan)
    r_hi = min(1.0, r_star + r_halfspan)

    alphas1 = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), 17)
    rs1     = np.linspace(r_lo, r_hi, 17)
    mprobe1 = 10

    gcv_best2 = gcv_best
    alpha_star2, r_star2 = alpha_star, r_star

    for r_try in rs1:
        for a_try in alphas1:
            _, yhat_obs, _, _, _ = local_solve(a_try, r_try)
            res = y_obs - np.asarray(yhat_obs, float).reshape(-1)
            num = float(np.sum(ws * np.abs(res)**2))

            df = 0.0
            for _ in range(mprobe1):
                z = np.sign(rng.standard_normal(nobs)).astype(float)
                if Npad == 0:
                    Zs_pad = z
                elif Npad == 1:
                    Zs_pad = np.concatenate([z, z[::-1]])
                else:
                    hn = np.hanning(2 * nobs)[:nobs]
                    Zs_pad = np.concatenate([z, (z * hn)[::-1], np.zeros(nobs * (Npad - 2)), (z[::-1]) * hn])
                P_backup = P
                P = np.fft.fftshift(np.fft.fft(Zs_pad.astype(np.complex128)))
                _, zhat_obs, _, _, _ = local_solve(a_try, r_try)
                P = P_backup
                df += float(np.real(np.vdot(z, np.asarray(zhat_obs).reshape(-1))))
            df /= mprobe1

            denom = max((nobs - df)**2, np.finfo(float).eps)
            gcv = num / denom
            if gcv < gcv_best2:
                gcv_best2 = gcv
                alpha_star2, r_star2 = float(a_try), float(r_try)

    alpha_star, r_star = alpha_star2, r_star2

    # -------- final solve with (alpha*, r*) --------
    Vf_full, Vf_obs, wk_final, dv, used_iter = local_solve(alpha_star, r_star)

    final_dv  = float(dv[-1]) if len(dv) else float("nan")
    converged = True
    if exitMode == 'thresh' and used_iter == Niter and final_dv > tol:
        converged = False
        warnings.warn(
            (f"Rubber_Band_Filter:ThresholdNotReached | "
             f"CG hit Niter={Niter} before meeting tol={tol:.3g} "
             f"(final dv={final_dv:.3g}). Consider increasing Niter or relaxing tol."),
            ThresholdNotReachedWarning,
            stacklevel=2
        )

    # -------- outputs --------
    Vf = np.real(Vf_obs) if np.isrealobj(Vso) else Vf_obs
    err = float(np.sum(wk_final * np.abs(Vf_full - Vs)**2))

    info = dict(
        alpha=alpha_star,
        r=r_star,
        dt=dt,
        N_original=Vso.size,
        N_padded=Vs.size,
        cutoff_used=float(freq + xtra),
        passband_len=g.size,
        iterations_requested=Niter,
        iterations_used=used_iter,
        exitMode=exitMode,
        tol=tol,
        dv=dv,
        final_dv=final_dv,
        converged=converged,
        err=err,
        Vf_pad=Vf_full,
        gcv_min=gcv_best2
    )
    return Vf, info


# ---------------- helpers ----------------

def _ToV(xin, fss, g):
    Vspec = np.zeros_like(fss, dtype=complex)
    Vspec[g] = xin
    return np.fft.ifft(np.fft.ifftshift(Vspec))

def _tukeywin(M, alpha=0.5):
    if M <= 0:
        return np.array([])
    if alpha <= 0:
        return np.ones(M)
    if alpha >= 1:
        return np.hanning(M)
    n = np.arange(M, dtype=float)
    w = np.ones(M, dtype=float)
    edge = int(np.floor(alpha*(M-1)/2.0))
    if edge > 0:
        w[:edge+1] = 0.5*(1 + np.cos(np.pi*(2*(n[:edge+1])/(alpha*(M-1)) - 1)))
        w[-(edge+1):] = 0.5*(1 + np.cos(np.pi*(2*(1 - n[-(edge+1):]/(M-1))/alpha - 1)))
    return w

def _interp_linear_extrap(x, xp, fp):
    x = np.asarray(x, float); xp = np.asarray(xp, float); fp = np.asarray(fp, float)
    order = np.argsort(xp); xp = xp[order]; fp = fp[order]
    y = np.interp(x, xp, fp)
    left  = x < xp[0]; right = x > xp[-1]
    if np.any(left):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[left] = fp[0] + slope * (x[left] - xp[0])
    if np.any(right):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[right] = fp[-1] + slope * (x[right] - xp[-1])
    return y
