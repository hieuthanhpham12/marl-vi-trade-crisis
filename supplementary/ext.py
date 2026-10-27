"""Supplementary experiments built ON TOP OF the authors' original ICTA2026 code
(sr_plr_marl_v8_mc_mr.py, imported as E). Nothing in the original schedulers,
operators or projection is changed; only the runner is generalised (extra shocks,
observation noise, missing data, drift, exact projection, reference equilibrium).
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'icta2026'))
import sr_plr_marl_v8_mc_mr as E

LAM = 1.0

# ------------------------------------------------------------------ networks
def make_single(m, n):
    return E.TradeNetwork(E.NetCfg(m=m, n=n))

def make_mcmr(m, n, K, L):
    return E.MultiCommodityRouteNetwork(E.MCMRCfg(m=m, n=n, K=K, L=L))

def is_mc(net):
    return isinstance(net, E.MultiCommodityRouteNetwork)

def x0_of(net, rng):
    if is_mc(net):
        c = net.cfg
        x = rng.uniform(c.init_low, c.init_high, size=(c.m, c.n, c.L, c.K))
    else:
        c = net.cfg
        x = rng.uniform(c.init_low, c.init_high, size=(c.m, c.n))
    return net.project(x)

def apply_shock(net, kind, mag):
    if kind == 'price':
        if is_mc(net): net.price_mult *= mag
        else: net.price_mult *= mag
    elif kind == 'tau' and is_mc(net):
        net.tau = net.tau * mag
    elif kind == 'cut':
        if is_mc(net): net.Q = net.Q * mag
        else:
            for i in range(net.cfg.m): net.supply_cut[i] = 1.0 - mag
    else:
        raise ValueError(kind)

# ---------------------------------------------------- exact Euclidean projection (MC/MR)
def _pcap(z, u, cap):
    y = np.clip(z, 0, u)
    if y.sum() <= cap + 1e-14: return y
    lo, hi = 0.0, float(z.max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.clip(z - mid, 0, u).sum() > cap: lo = mid
        else: hi = mid
    return np.clip(z - hi, 0, u)

def _Psup(X, net):
    m, n, L, K = X.shape; Y = np.empty_like(X)
    for i in range(m):
        for k in range(K):
            Y[i, :, :, k] = _pcap(X[i, :, :, k].ravel(), net.u[i, :, :, k].ravel(), net.Q[i, k]).reshape(n, L)
    return Y

def _Pdem(X, net):
    m, n, L, K = X.shape; Y = np.empty_like(X)
    for j in range(n):
        for k in range(K):
            Y[:, j, :, k] = _pcap(X[:, j, :, k].ravel(), net.u[:, j, :, k].ravel(), net.D_cap[j, k]).reshape(m, L)
    return Y

def proj_exact(net, x, iters=300, tol=1e-9):
    X = x.copy(); p = np.zeros_like(X); q = np.zeros_like(X)
    for _ in range(iters):
        a = _Psup(X + p, net); p = X + p - a
        Xn = _Pdem(a + q, net); q = a + q - Xn
        if np.abs(Xn - X).max() < tol: X = Xn; break
        X = Xn
    return X

def G_exact(net, x, lam=LAM):
    return float(np.linalg.norm(x - proj_exact(net, x - lam * net.compute_F(x)))) / lam

# ------------------------------------------------ reference equilibrium (MC/MR)
def potential(net, x):
    c = net.cfg; X = x.reshape(c.m, c.n, c.L, c.K)
    q = X.sum(axis=(1, 2)); d = X.sum(axis=(0, 2))
    f = (net.a * q + 0.5 * net.b * q ** 2).sum() + (net.tau * X + 0.5 * net.beta * X ** 2).sum()
    d0 = (net.p - c.P_min) / net.s
    Phi = np.where(d <= d0, net.p * d - 0.5 * net.s * d ** 2,
                   net.p * d0 - 0.5 * net.s * d0 ** 2 + c.P_min * (d - d0))
    return f - (net.price_mult * Phi).sum()

def solve_xstar(net):
    c = net.cfg; shp = (c.m, c.n, c.L, c.K); d = int(np.prod(shp))
    idx = np.arange(d).reshape(shp); rows, rhs = [], []
    for i in range(c.m):
        for k in range(c.K):
            r = np.zeros(d); r[idx[i, :, :, k].ravel()] = 1; rows.append(r); rhs.append(net.Q[i, k])
    for j in range(c.n):
        for k in range(c.K):
            r = np.zeros(d); r[idx[:, j, :, k].ravel()] = 1; rows.append(r); rhs.append(net.D_cap[j, k])
    A = np.array(rows); b = np.array(rhs)
    Fv = lambda z: net.compute_F(z.reshape(shp)).ravel()
    res = minimize(lambda z: potential(net, z), np.ones(d), jac=Fv, bounds=[(0, float(v)) for v in net.u.ravel()],
                   constraints=[dict(type='ineq', fun=lambda z: b - A @ z, jac=lambda z: -A)],
                   method='SLSQP', options=dict(maxiter=600, ftol=1e-14))
    return res.x.reshape(shp)

# --------------------------------------------------------------------- runner
def run_ext(sched, net, seed, N=1200, shocks=((300, 'price', 2.0),), sigma=0.01, obs_sigma=0.0,
            miss=0.0, drift=None, exact_proj=False, xstar=None):
    """Same timing convention as the original runners: log at t, then inject the shock scheduled
    at t, then update with the post-shock operator."""
    rng = np.random.default_rng(seed)
    x = x0_of(net, rng)
    D0 = net.compute_D(x); sched.init(D0)
    rule = getattr(sched, 'update_rule', 'plr')
    P = (lambda y: proj_exact(net, y)) if exact_proj else net.project
    sh = {s[0]: s for s in shocks}
    base_pm = np.copy(net.price_mult)
    Dmax_eval = max(D0, 1e-12)
    T = N
    out = dict(D=np.zeros(T), G=np.zeros(T), rho=np.zeros(T), rho_eval=np.zeros(T), flag=np.zeros(T, bool),
               alpha=np.zeros(T), err=np.full(T, np.nan))
    for t in range(T):
        Dtrue = net.compute_D(x)
        Fc = net.compute_F(x)
        if miss > 0:
            M = rng.random(Fc.shape) > miss
            Dobs = float(np.abs(Fc)[M].mean()) if M.any() else Dtrue
        else:
            M = None; Dobs = Dtrue
        if obs_sigma > 0:
            Dobs = float(np.mean(np.abs(Fc + obs_sigma * rng.normal(size=Fc.shape))))
        G = net.compute_G_lambda(x, lam=LAM)
        alpha, diag = sched.step(t, Dobs)
        Dmax_eval = max(Dmax_eval, Dtrue)
        out['D'][t] = Dtrue; out['G'][t] = G; out['rho'][t] = diag['rho']
        out['rho_eval'][t] = max(0.0, 1 - Dtrue / Dmax_eval)
        out['flag'][t] = bool(diag.get('shock_detected', False)); out['alpha'][t] = alpha
        if xstar is not None:
            out['err'][t] = np.linalg.norm(x - xstar) / max(np.linalg.norm(xstar), 1e-12)
        if t in sh:
            apply_shock(net, sh[t][1], sh[t][2]); base_pm = np.copy(net.price_mult)
        if drift is not None and t >= drift['t0']:
            net.price_mult = base_pm * (1 + drift['amp'] * np.sin(2 * np.pi * (t - drift['t0']) / drift['period']))
        Fx = net.compute_F(x)
        if M is not None: Fx = Fx * M
        if rule == 'eg':
            y = P(x - alpha * Fx + alpha * rng.normal(0, sigma, size=x.shape))
            Fy = net.compute_F(y)
            if M is not None: Fy = Fy * M
            x = P(x - alpha * Fy + alpha * rng.normal(0, sigma, size=x.shape))
        else:
            x = P(x - alpha * Fx + alpha * rng.normal(0, sigma, size=x.shape))
    out['x'] = x
    out['G_final'] = net.compute_G_lambda(x, lam=LAM)
    out['D_final'] = net.compute_D(x)
    return out

# ----------------------------------------------------------------- metrics
def g50(G, ts, t_end, window=400, peak_win=20, frac=0.5):
    """D_max-independent criterion of the accepted paper: G_peak = max G over the first 20 observable
    post-shock episodes, recovery = first t > t_peak with G <= 0.5 G_peak. Returns (success, RT)."""
    seg = G[ts + 1: ts + 1 + peak_win]
    gp = seg.max(); tp = ts + 1 + int(np.argmax(seg))
    end = min(t_end, ts + window + 1)
    for t in range(tp + 1, end):
        if G[t] <= frac * gp: return True, t - ts
    return False, window

def rho_rec(rho, ts, t_end, window=400, thr=0.5):
    for t in range(ts + 1, min(t_end, ts + window + 1)):
        if rho[t] >= thr: return True, t - ts
    return False, window

def schedulers(T2=100.0, sigma=0.01, **srkw):
    plr = E.PLRCfg(sigma=sigma, T2=T2); sr = E.SRCfg(sigma=sigma, T2=T2, **srkw)
    M = E._v8_main_methods(plr, sr)
    M.update({'SR-WarmOnly': lambda: E.SRPLRWarmRestartOnly(sr), 'SR-ClockOnly': lambda: E.SRPLRClockOnly(sr),
              'SR-Lock50': lambda: E.SRPLRLock(sr, lock_len=50)})
    return M
