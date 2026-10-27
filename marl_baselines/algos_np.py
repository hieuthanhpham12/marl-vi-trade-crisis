"""MAPPO and MADDPG implemented from scratch in numpy (small MLPs, manual backprop, Adam).
MAPPO : shared Gaussian actor (agent-id one-hot appended), centralised value function, PPO-clip, GAE.
MADDPG: separate deterministic actor and centralised critic Q_i(s, a_1..a_m) per agent, replay buffer, target nets, Gaussian exploration."""
import numpy as np
from npnn import MLP, Adam

class MAPPO:
    name = 'MAPPO'
    def __init__(self, m, od, gd, da, seed, lr=3e-4, roll=400, epochs=5, gamma=0.95, lam=0.95, clip=0.2, log_std0=-1.0):
        self.rng = np.random.default_rng(seed); self.m, self.od, self.gd, self.da = m, od, gd, da
        self.actor = MLP(od + m, da, self.rng, out_tanh=True); self.critic = MLP(gd, 1, self.rng, out_scale=1.0)
        self.log_std = np.full(da, float(log_std0))
        self.aopt = Adam(self.actor.P + [self.log_std], lr); self.copt = Adam(self.critic.P, lr)
        self.eye = np.eye(m, dtype=np.float32); self.roll, self.epochs, self.gamma, self.lam, self.clip = roll, epochs, gamma, lam, clip
        self.buf = []
    def _in(self, loc):            # (B,m,od) -> (B*m, od+m)
        B = loc.shape[0]; return np.concatenate([loc, np.broadcast_to(self.eye, (B, self.m, self.m))], -1).reshape(B * self.m, -1)
    def act(self, loc, glob, explore=True):
        mu = self.actor.forward(self._in(loc[None]))                  # (m,da)
        if not explore: return mu.astype(np.float32), None
        std = np.exp(self.log_std); a = mu + std * self.rng.normal(size=mu.shape)
        lp = (-0.5 * ((a - mu) / std) ** 2 - self.log_std - 0.9189385).sum(-1)
        return a.astype(np.float32), lp
    def observe(self, loc, glob, a, lp, r, nglob, cut): self.buf.append((loc, glob, a, lp, r, nglob, cut))
    def ready(self): return len(self.buf) >= self.roll
    def update(self):
        B = self.buf; T = len(B); m = self.m
        loc = np.stack([b[0] for b in B]); glob = np.stack([b[1] for b in B]); a = np.stack([b[2] for b in B]); olp = np.stack([b[3] for b in B])
        r = np.array([b[4] for b in B]); ng = np.stack([b[5] for b in B]); cut = np.array([b[6] for b in B])
        v = self.critic.forward(glob)[:, 0]; vn = self.critic.forward(ng)[:, 0]
        delta = r + self.gamma * vn - v; adv = np.zeros(T); g = 0.0
        for t in range(T - 1, -1, -1): g = delta[t] + self.gamma * self.lam * (1 - cut[t]) * g; adv[t] = g
        ret = adv + v; advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        inp = self._in(loc).reshape(T, m, -1); std = np.exp(self.log_std)
        for _ in range(self.epochs):
            perm = self.rng.permutation(T)
            for mb in np.array_split(perm, 4):
                nb = len(mb)
                mu = self.actor.forward(inp[mb].reshape(nb * m, -1)).reshape(nb, m, -1)
                am = a[mb]; z = (am - mu) / std
                lp = (-0.5 * z ** 2 - self.log_std - 0.9189385).sum(-1); ratio = np.exp(lp - olp[mb]); A = advn[mb][:, None]
                active = ((A >= 0) & (ratio < 1 + self.clip)) | ((A < 0) & (ratio > 1 - self.clip))
                dlp = -(A * ratio) * active / (nb * m)                          # d loss / d lp   (nb,m)
                dmu = dlp[..., None] * (z / std); dls = (dlp[..., None] * (z ** 2 - 1)).sum((0, 1))
                self.actor.forward(inp[mb].reshape(nb * m, -1)); self.actor.backward(dmu.reshape(nb * m, -1))
                self.aopt.step(self.actor.G + [dls])
                V = self.critic.forward(glob[mb])[:, 0]; self.critic.backward(((V - ret[mb]) / nb)[:, None]); self.copt.step(self.critic.G)
        self.buf = []

class MADDPG:
    name = 'MADDPG'
    def __init__(self, m, od, gd, da, seed, lr=1e-3, gamma=0.95, tau=0.01, batch=128, warm=2000, every=2, expl=0.2, expl_min=0.05, expl_decay=0.9999, cap=100000):
        self.rng = np.random.default_rng(seed); self.m, self.od, self.gd, self.da = m, od, gd, da
        self.actors = [MLP(od, da, self.rng, out_tanh=True) for _ in range(m)]; self.critics = [MLP(gd + m * da, 1, self.rng, out_scale=1.0) for _ in range(m)]
        self.tactors = [x.copy() for x in self.actors]; self.tcritics = [x.copy() for x in self.critics]
        self.aopt = [Adam(x.P, lr) for x in self.actors]; self.copt = [Adam(x.P, lr) for x in self.critics]
        self.gamma, self.tau, self.batch, self.warm, self.every = gamma, tau, batch, warm, every
        self.expl, self.expl_min, self.expl_decay = expl, expl_min, expl_decay
        self.cap = cap; self.n = 0; self.steps = 0
        self.L = np.zeros((cap, m, od), np.float32); self.G = np.zeros((cap, gd), np.float32); self.A = np.zeros((cap, m, da), np.float32)
        self.R = np.zeros(cap, np.float32); self.NL = np.zeros((cap, m, od), np.float32); self.NG = np.zeros((cap, gd), np.float32)
    def act(self, loc, glob, explore=True):
        a = np.stack([self.actors[i].forward(loc[i:i + 1])[0] for i in range(self.m)])
        if explore: a = np.clip(a + self.expl * self.rng.normal(size=a.shape), -1, 1)
        return a.astype(np.float32), None
    def observe(self, loc, glob, a, lp, r, nglob, cut, nloc=None):
        k = self.n % self.cap
        self.L[k] = loc; self.G[k] = glob; self.A[k] = np.clip(a, -1, 1); self.R[k] = r; self.NG[k] = nglob; self.NL[k] = nloc
        self.n += 1; self.steps += 1; self.expl = max(self.expl_min, self.expl * self.expl_decay)
    def ready(self): return self.n >= self.warm and self.steps % self.every == 0
    def update(self):
        N = min(self.n, self.cap); idx = self.rng.integers(0, N, self.batch); b = self.batch; m = self.m; da = self.da
        L, G, A, R, NL, NG = self.L[idx], self.G[idx], self.A[idx], self.R[idx], self.NL[idx], self.NG[idx]
        NA = np.stack([self.tactors[i].forward(NL[:, i]) for i in range(m)], 1)
        nin = np.concatenate([NG, NA.reshape(b, -1)], -1); cin = np.concatenate([G, A.reshape(b, -1)], -1)
        for i in range(m):
            y = R + self.gamma * self.tcritics[i].forward(nin)[:, 0]
            q = self.critics[i].forward(cin)[:, 0]; self.critics[i].backward((2 * (q - y) / b)[:, None]); self.copt[i].step(self.critics[i].G)
        for i in range(m):
            acts = [self.actors[i].forward(L[:, i]) if j == i else A[:, j] for j in range(m)]
            ain = np.concatenate([G, np.stack(acts, 1).reshape(b, -1)], -1)
            self.critics[i].forward(ain); dx = self.critics[i].backward(np.full((b, 1), -1.0 / b))
            dact = dx[:, self.gd + i * da: self.gd + (i + 1) * da]
            self.actors[i].forward(L[:, i]); self.actors[i].backward(dact); self.aopt[i].step(self.actors[i].G)
        for net, tnet in list(zip(self.actors, self.tactors)) + list(zip(self.critics, self.tcritics)):
            for p, tp in zip(net.P, tnet.P): tp *= (1 - self.tau); tp += self.tau * p
