"""Fast smoke tests (a few seconds). Run:  python tests/test_smoke.py   (or: pytest -q)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("icta2026", "supplementary", "marl_baselines"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import numpy as np


def test_original_code_reproduces_icta_2x2():
    """Table 1 of the ICTA2026 paper: on the 2x2 network the price shock x2 at episode 300 gives
    RT = 240 for SR-PLR-MARL (G50 criterion) and PLR-original does not recover within 400 episodes."""
    import ext
    M = ext.schedulers()
    out = {}
    for name in ("PLR-original", "SR-PLR-MARL"):
        net = ext.make_single(2, 2)
        r = ext.run_ext(M[name](), net, 0, N=800, shocks=((300, "price", 2.0),), sigma=0.01)
        out[name] = ext.g50(r["G"], 300, 800)
    assert out["SR-PLR-MARL"][0] and out["SR-PLR-MARL"][1] == 240.0, out
    assert not out["PLR-original"][0], out


def test_operator_is_strongly_monotone_on_samples():
    import ext
    net = ext.make_single(3, 3)
    rng = np.random.default_rng(0)
    worst = np.inf
    for _ in range(50):
        x = ext.x0_of(net, rng); y = ext.x0_of(net, rng)
        d = (x - y).ravel(); df = (net.compute_F(x) - net.compute_F(y)).ravel()
        worst = min(worst, float(d @ df) / float(d @ d))
    assert worst > 0.02, worst            # mu >= beta_min ~ 0.03 (Proposition in the report)


def test_numpy_mlp_gradients():
    from npnn import MLP
    rng = np.random.default_rng(0)
    net = MLP(5, 3, rng, out_tanh=True, out_scale=1.0)
    x = rng.normal(size=(4, 5)); w = rng.normal(size=(4, 3))
    f = lambda: float((net.forward(x) * w).sum())
    net.forward(x); net.backward(w.copy())
    err = 0.0
    for k, p in enumerate(net.P):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p.shape); old = p[idx]
            p[idx] = old + 1e-6; fp = f(); p[idx] = old - 1e-6; fm = f(); p[idx] = old
            err = max(err, abs((fp - fm) / 2e-6 - net.G[k][idx]) / (1e-8 + abs(net.G[k][idx])))
    assert err < 1e-4, err


def test_marl_baselines_run():
    from rlrun import train, deploy
    hp = dict(lr=1e-3, s=0.5)
    for algo in ("MAPPO", "MADDPG"):
        ag, env = train(algo, ("single", 2, 2), 0, hp, "none", 600)     # tiny budget, only checks the plumbing
        res = deploy(ag, algo, ("single", 2, 2), 0, hp, "frozen", 450)
        assert np.isfinite(res["G_final"]), res


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK ", name)
