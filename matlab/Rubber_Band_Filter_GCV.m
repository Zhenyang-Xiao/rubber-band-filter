function [Vf, info] = Rubber_Band_Filter_GCV(ts, Vs, freq, varargin)
% Rubber_Band_Filter_GCV
% Low-pass filtering without edge artifacts. The pad weight shape
% (alpha scale and Tukey ratio r) is chosen by GCV. Fitting is solved
% by Conjugate Gradient (CG) with the same exit policy as the base filter.
%
% Examples:
%   Vf          = Rubber_Band_Filter_GCV(ts, Vs, freq);
%   [Vf, info]  = Rubber_Band_Filter_GCV(ts, Vs, freq, 'exitmode','fixed');
%   Vf          = Rubber_Band_Filter_GCV(ts, Vs, freq, 'exitmode','thresh','tol',1e-9);
%   Vf          = Rubber_Band_Filter_GCV(ts, Vs, freq, 'Niter',50,'Npad',1);
%
% Required:
%   ts   : time vector (monotonic, ~uniform)
%   Vs   : signal vector (same length as ts)
%   freq : cutoff frequency (Hz)
%
% Optional name–value parameters:
%   'exitmode' : 'thresh' (default) or 'fixed'
%   'tol'      : tolerance for 'thresh' (default 1e-7)
%   'Npad'     : padding count (default 1; right-side mirror)
%   'Niter'    : max CG iterations per solve (default 30)
%   'ws'       : weights on valid region (default ones; 0 where Vs is NaN)

    % ------------ inputs & defaults ------------
    ts = ts(:); Vs = Vs(:);
    assert(numel(ts) == numel(Vs), 'ts and Vs must have same length.');

    exitMode = 'thresh';
    tol      = 1e-7;
    Npad     = 1;
    Niter    = 30;
    ws       = ones(size(Vs));

    % case-by-case varargin scan (key at i, value at i+1)
    for i1 = 1:length(varargin)
        key = varargin{i1};
        if ischar(key) || (isstring(key) && isscalar(key))
            key = lower(strtrim(char(key)));
            if i1+1 > length(varargin), error('Option "%s" missing value.', key); end
            val = varargin{i1+1};
            switch key
                case 'exitmode', exitMode = canonical_exitmode(val);
                case 'tol',      tol      = val;
                case 'npad',     Npad     = val;
                case 'niter',    Niter    = val;
                case 'ws',       ws       = val;
            end
        end
    end
    if isempty(ws), ws = ones(size(Vs)); else, ws = ws(:); end

    % ------------ deweight NaNs & interpolate ------------
    ws = ws .* (1 - isnan(Vs));
    if any(isnan(Vs))
        Vs(isnan(Vs)) = interp1(ts(~isnan(Vs)), Vs(~isnan(Vs)), ts(isnan(Vs)), 'linear', 'extrap');
    end

    % ------------ pad the signal only (weights built per solve) ------------
    Vso = Vs;
    if Npad == 0
        Vs = Vso;
    elseif Npad == 1
        Vs = [Vso; flipud(Vso)];
    else
        hn = hanning(2*length(Vso)); hn = hn(1:length(Vso));
        Vs = [Vso; flipud(Vso.*hn); zeros(length(Vso)*(Npad-2),1); flipud(Vso).*hn];
    end

    % ------------ frequency grid & passband ------------
    dt = mean(diff(ts));
    N  = length(Vs);
    fss = fftshift((1/(N*dt))*(0:N-1)');
    zi = find(fss == 0, 1, 'first');
    if ~isempty(zi), fss(1:zi-1) = fss(1:zi-1) - 1/dt; else, error('Zero frequency not found.'); end

    g    = find(abs(fss) <= freq);
    xtra = 0;
    while mod(length(g),2)==0
        xtra = xtra + eps(freq);
        g = find(abs(fss) <= freq + xtra);
    end

    % precompute spectrum of padded signal (updated temporarily for Hutchinson probes)
    P = fftshift(fft(Vs));

    % ------------ parametric pad-weight shape ------------
    alpha0 = 0.03;    % baseline scale
    r0     = 0.20;    % baseline Tukey ratio
    shape_len  = length(ws);
    make_shape = @(r) tukeywin(shape_len, max(0,min(1,r)));

    % ------------ local solver (CG with same exit rules) ------------
    function [Vf_full, Vf_obs, wk_out, dv, used_iter] = local_solve(alpha, r)
        shape = make_shape(r);
        if Npad == 0
            wk = ws;
        elseif Npad == 1
            wk = [ws; alpha*shape];
        else
            % keep observed weights; no weights on extra pads (consistent with your code)
            wk = [ws; zeros(length(ws)*Npad,1)];
        end

        W = fftshift(fft(wk));
        function co = convf(xi)
            if length(xi)==length(g), zix = find(fss(g)==0, 1, 'first');
            else,                     zix = find(fss==0,    1, 'first'); end
            co_full = ifft( fft(xi, length(xi)+length(W)-1) .* fft(W, length(xi)+length(W)-1) );
            ko = zi - 1 + zix;
            co = co_full( ko - (length(g)-1)/2 : ko + (length(g)-1)/2 );
        end

        b   = convf(P);
        x   = zeros(size(P(g)));    % start at zero (no rng seeding)
        rCG = b - convf(x);
        p   = rCG;
        bnorm2 = real(b' * b);

        dv = zeros(1, Niter);
        for ii = 1:Niter
            Ap = convf(p);
            ak = (rCG' * rCG) / (p' * Ap);
            xo = x;
            x  = x + ak * p;
            ro = rCG;
            rCG = rCG - ak * Ap;
            bk = (rCG' * rCG) / (ro' * ro);

            % monotonicity safeguard in time-domain
            if sum(wk .* abs((ToV(x) - Vs)).^2) > sum(wk .* abs((ToV(xo) - Vs)).^2)
                bk = 0;
            end
            p = rCG + bk * p;

            dv(ii) = real(rCG' * rCG) / bnorm2;
            if strcmpi(exitMode,'thresh') && dv(ii) <= tol
                dv = dv(1:ii);
                break;
            end
        end
        used_iter = min(ii, Niter);

        Vf_full = ToV(x);
        if isreal(Vs), Vf_full = real(Vf_full); end
        Vf_obs  = Vf_full(1:length(Vso));
        wk_out  = wk;
    end

    % ------------ GCV: coarse sweep ------------
    y_obs  = Vso(:);
    nobs   = numel(y_obs);
    rng(0);    % reproducible Hutchinson probes

    alphas0 = logspace(log10(alpha0/10), log10(alpha0*10), 15);
    rs0     = linspace(max(0,r0-0.5), min(1,r0+0.5), 15);
    mprobe0 = 7;

    gcv_best = inf; alpha_star = alpha0; r_star = r0;
    for ir = 1:numel(rs0)
        r_try = rs0(ir);
        for ia = 1:numel(alphas0)
            a_try = alphas0(ia);

            % numerator: weighted residual on observed segment
            [~, yhat_obs] = local_solve(a_try, r_try);
            res = y_obs - yhat_obs;
            num = sum( (ws(:) .* abs(res)).^2 );

            % denominator: (nobs - df)^2 via Hutchinson
            df = 0;
            for k = 1:mprobe0
                z = sign(randn(nobs,1));
                if Npad==0
                    Zs_pad = z;
                elseif Npad==1
                    Zs_pad = [z; flipud(z)];
                else
                    hn = hanning(2*nobs); hn = hn(1:nobs);
                    Zs_pad = [z; flipud(z.*hn); zeros(nobs*(Npad-2),1); flipud(z).*hn];
                end
                P_backup = P; P = fftshift(fft(Zs_pad));
                [~, zhat_obs] = local_solve(a_try, r_try);
                P = P_backup;
                df = df + (z.' * zhat_obs);
            end
            df = df / mprobe0;

            gcv = num / max((nobs - df)^2, eps);
            if gcv < gcv_best
                gcv_best = gcv; alpha_star = a_try; r_star = r_try;
            end
        end
    end

    % ------------ refine window ------------
    on_alpha_edge = (alpha_star <= min(alphas0)+eps) || (alpha_star >= max(alphas0)-eps);
    on_r_edge     = (r_star     <= min(rs0)+eps)     || (r_star     >= max(rs0)-eps);
    if on_alpha_edge, alpha_lo = alpha_star/6; alpha_hi = alpha_star*6;
    else,             alpha_lo = alpha_star/3; alpha_hi = alpha_star*3; end
    r_halfspan = on_r_edge * 0.30 + (~on_r_edge) * 0.15;
    r_lo = max(0, r_star - r_halfspan);
    r_hi = min(1, r_star + r_halfspan);

    alphas1 = logspace(log10(alpha_lo), log10(alpha_hi), 17);
    rs1     = linspace(r_lo, r_hi, 17);
    mprobe1 = 10;

    gcv_best2 = gcv_best; alpha_star2 = alpha_star; r_star2 = r_star;
    for ir = 1:numel(rs1)
        r_try = rs1(ir);
        for ia = 1:numel(alphas1)
            a_try = alphas1(ia);

            [~, yhat_obs] = local_solve(a_try, r_try);
            res = y_obs - yhat_obs;
            num = sum( (ws(:) .* abs(res)).^2 );

            df = 0;
            for k = 1:mprobe1
                z = sign(randn(nobs,1));
                if Npad==0
                    Zs_pad = z;
                elseif Npad==1
                    Zs_pad = [z; flipud(z)];
                else
                    hn = hanning(2*nobs); hn = hn(1:nobs);
                    Zs_pad = [z; flipud(z.*hn); zeros(nobs*(Npad-2),1); flipud(z).*hn];
                end
                P_backup = P; P = fftshift(fft(Zs_pad));
                [~, zhat_obs] = local_solve(a_try, r_try);
                P = P_backup;
                df = df + (z.' * zhat_obs);
            end
            df = df / mprobe1;

            gcv = num / max((nobs - df)^2, eps);
            if gcv < gcv_best2
                gcv_best2 = gcv; alpha_star2 = a_try; r_star2 = r_try;
            end
        end
    end

    alpha_star = alpha_star2; r_star = r_star2;

    % ------------ final solve at (alpha*, r*) ------------
    [Vf_full, Vf_obs, wk_final, dv, used_iter] = local_solve(alpha_star, r_star);

    % threshold-mode warning (match base function behavior)
    final_dv = dv(end);
    converged = true;
    if strcmpi(exitMode,'thresh') && used_iter == Niter && final_dv > tol
        converged = false;
        warning('Rubber_Band_Filter:ThresholdNotReached', ...
            'CG hit Niter=%d before meeting tol=%.3g (final dv=%.3g). Consider increasing Niter or relaxing tol.', ...
            Niter, tol, final_dv);
    end

    % ------------ outputs ------------
    Vf      = Vf_obs;            % cropped to original length
    Vf_pad  = Vf_full;           %#ok<NASGU>  (in info)
    err     = sum(wk_final .* abs((Vf_full - Vs)).^2);

    if nargout > 1
        info = struct( ...
            'alpha', alpha_star, ...
            'r', r_star, ...
            'dt', dt, ...
            'N_original', length(Vso), ...
            'N_padded',   length(Vs), ...
            'cutoff_used', freq + xtra, ...
            'passband_len', length(g), ...
            'iterations_requested', Niter, ...
            'iterations_used', used_iter, ...
            'exitMode', exitMode, ...
            'tol', tol, ...
            'dv', dv, ...
            'final_dv', final_dv, ...
            'converged', converged, ...
            'err', err, ...
            'Vf_pad', Vf_full, ...
            'gcv_min', gcv_best2 ...
        );
    end

    % ---------- nested ----------
    function Vout = ToV(xin)
        VoutSpec = zeros(size(fss)); VoutSpec(g) = xin;
        Vout = ifft(ifftshift(VoutSpec));
    end
end

% -------- helpers ----------
function s = canonical_exitmode(x)
    s = lower(strtrim(string(x)));
    if any(s == ["fixed","niter","maxiter","full","fix"])
        s = "fixed";
    else
        s = "thresh";
    end
    s = char(s);
end
