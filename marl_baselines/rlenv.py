"""Multi-agent environment wrapped around the ICTA2026 trade-network operator (icta2026 code via supplementary/ext.py).
Agents = supply nodes i (m agents); agent i controls the flows x[i,...] it ships (all markets/routes/commodities).
Observation of agent i : own flows and own components of the (noisy) operator F, normalised.
Global observation (critic): all flows and all F components.
Action: a in [-1,1]^{d_i};  x <- P_X(x + s*a).      Team reward: r = -D(x_next)  (economic mismatch).
Operator evaluations: one F evaluation per step (same as PG/PLR family)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'supplementary'))
import numpy as np
import ext

def mk(spec):
    return ext.make_single(spec[1], spec[2]) if spec[0] == 'single' else ext.make_mcmr(*spec[1:])

class Env:
    def __init__(self, spec, seed, sigma=0.01, s=0.5, H=50, shock_aware=False):
        self.spec = spec; self.net = mk(spec); self.rng = np.random.default_rng(seed)
        self.sigma = sigma; self.s = s; self.H = H; self.shock_aware = shock_aware
        self.pm0 = np.array(self.net.price_mult, dtype=float)
        x0 = ext.x0_of(self.net, np.random.default_rng(12345))
        self.xs = 10.0
        self.fs = float(np.abs(self.net.compute_F(x0)).mean())
        self.m = x0.shape[0]; self.da = x0[0].size; self.shape = x0.shape
        self.od = 2 * self.da; self.gd = 2 * x0.size
        self.t = 0
    def reset(self, x0=None, pm=None):
        self.net.price_mult = self.pm0 * 1.0
        if self.shock_aware and pm is None:
            pm = self.rng.uniform(1.0, 3.0)
        if pm is not None: self.net.price_mult = self.pm0 * pm
        self.x = ext.x0_of(self.net, self.rng) if x0 is None else x0
        self.t = 0
        return self.obs()
    def obs(self):
        F = self.net.compute_F(self.x) + self.sigma * self.rng.normal(size=self.x.shape)
        xn = np.clip(self.x / self.xs, -5, 5); Fn = np.clip(F / self.fs, -10, 10)
        loc = np.concatenate([xn.reshape(self.m, -1), Fn.reshape(self.m, -1)], axis=1)   # (m, 2*da)
        glob = np.concatenate([xn.ravel(), Fn.ravel()])
        return loc.astype(np.float32), glob.astype(np.float32)
    def step(self, a):
        """a: (m, da) in [-1,1]"""
        a = np.clip(a, -1, 1).reshape(self.shape)
        self.x = self.net.project(self.x + self.s * a)
        self.t += 1
        r = -float(self.net.compute_D(self.x))
        return self.obs(), r, (self.t >= self.H)
    def G(self): return float(self.net.compute_G_lambda(self.x, lam=1.0))
    def D(self): return float(self.net.compute_D(self.x))
