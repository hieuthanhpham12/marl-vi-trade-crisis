import numpy as np
class MLP:
    """in -> h -> h -> out, tanh hidden, optional tanh output; manual backprop."""
    def __init__(self, i, o, rng, h=64, out_tanh=False, out_scale=0.1):
        s = lambda a, b, sc=1.0: rng.normal(0, sc / np.sqrt(a), size=(a, b))
        self.P = [s(i, h), np.zeros(h), s(h, h), np.zeros(h), s(h, o, out_scale), np.zeros(o)]
        self.out_tanh = out_tanh; self.G = [np.zeros_like(p) for p in self.P]
    def forward(self, x):
        W1, b1, W2, b2, W3, b3 = self.P
        self.x = x; self.h1 = np.tanh(x @ W1 + b1); self.h2 = np.tanh(self.h1 @ W2 + b2); y = self.h2 @ W3 + b3
        if self.out_tanh: y = np.tanh(y)
        self.y = y; return y
    def backward(self, dy):
        W1, b1, W2, b2, W3, b3 = self.P
        if self.out_tanh: dy = dy * (1 - self.y ** 2)
        self.G[4] = self.h2.T @ dy; self.G[5] = dy.sum(0)
        dz2 = (dy @ W3.T) * (1 - self.h2 ** 2); self.G[2] = self.h1.T @ dz2; self.G[3] = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * (1 - self.h1 ** 2); self.G[0] = self.x.T @ dz1; self.G[1] = dz1.sum(0)
        return dz1 @ W1.T
    def copy(self):
        import copy; return copy.deepcopy(self)
class Adam:
    def __init__(self, params, lr, clip=1.0, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params; self.lr = lr; self.clip = clip; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0
    def step(self, grads):
        if self.clip:
            n = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if n > self.clip: grads = [g * (self.clip / (n + 1e-12)) for g in grads]
        self.t += 1
        for k, (p, g) in enumerate(zip(self.p, grads)):
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g; self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
if __name__ == '__main__':      # finite-difference gradient check
    rng = np.random.default_rng(0); net = MLP(5, 3, rng, out_tanh=True, out_scale=1.0); x = rng.normal(size=(4, 5)); w = rng.normal(size=(4, 3))
    f = lambda: float((net.forward(x) * w).sum())
    net.forward(x); dx = net.backward(w.copy()); err = 0
    for k, p in enumerate(net.P):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p.shape); old = p[idx]; p[idx] = old + 1e-6; fp = f(); p[idx] = old - 1e-6; fm = f(); p[idx] = old
            err = max(err, abs((fp - fm) / 2e-6 - net.G[k][idx]) / (1e-8 + abs(net.G[k][idx])))
    xi = (1, 2); old = x[xi]; x[xi] = old + 1e-6; fp = f(); x[xi] = old - 1e-6; fm = f(); x[xi] = old
    print('param grad rel err', err, ' input grad err', abs((fp - fm) / 2e-6 - dx[xi]))
