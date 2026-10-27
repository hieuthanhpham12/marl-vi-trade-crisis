import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'supplementary'))
import numpy as np, json, time, copy

import ext
from rlenv import Env
from algos_np import MAPPO, MADDPG

NSTEPS = {'single': 800, 'mcmr': 1200}
def make_agent(algo, env, seed, hp):
    if algo == 'MAPPO': return MAPPO(env.m, env.od, env.gd, env.da, seed, lr=hp['lr'], roll=400, epochs=5)
    return MADDPG(env.m, env.od, env.gd, env.da, seed, lr=hp['lr'])

def train(algo, spec, seed, hp, regime, budget):
    env = Env(spec, seed, s=hp['s'], shock_aware=(regime == 'aware'), H=50)
    ag = make_agent(algo, env, seed, hp)
    loc, glob = env.reset(); steps = 0; curve = []
    while steps < budget:
        a, lp = ag.act(loc, glob, True)
        (nloc, nglob), r, done = env.step(a)
        if algo == 'MAPPO': ag.observe(loc, glob, a, lp, r, nglob, float(done))
        else: ag.observe(loc, glob, a, lp, r, nglob, float(done), nloc=nloc)
        steps += 1
        if ag.ready(): ag.update()
        if done: loc, glob = env.reset()
        else: loc, glob = nloc, nglob
    return ag, env

def deploy(ag, algo, spec, seed, hp, mode, N, t_shock=300, mag=2.0, sigma=0.01, traj=False):
    env = Env(spec, seed + 1000, s=hp['s'], H=10 ** 9, sigma=sigma)
    loc, glob = env.reset()
    online = (mode == 'online')
    if online:
        if algo == 'MAPPO': ag.roll = 100; ag.epochs = 3
        else: ag.expl = ag.expl_min
    G = np.zeros(N); D = np.zeros(N)
    for t in range(N):
        G[t] = env.G(); D[t] = env.D()
        if t == t_shock:
            ext.apply_shock(env.net, 'price', mag); loc, glob = env.obs()
        a, lp = ag.act(loc, glob, explore=online)
        (nloc, nglob), r, done = env.step(a)
        if online:
            if algo == 'MAPPO': ag.observe(loc, glob, a, lp, r, nglob, 0.0)
            else: ag.observe(loc, glob, a, lp, r, nglob, 0.0, nloc=nloc)
            if ag.ready(): ag.update()
        loc, glob = nloc, nglob
    ok, rt = ext.g50(G, t_shock, N)
    # also a threshold-free robustness check: first t after shock with G <= 1.0 and G <= 0.5
    def first(th):
        idx = np.where(G[t_shock + 1:t_shock + 401] <= th)[0]
        return int(idx[0]) + 1 if len(idx) else 400
    out = dict(g50_ok=bool(ok), g50_rt=float(rt), G_pre=float(G[t_shock - 1]), G_peak=float(G[t_shock + 1:t_shock + 21].max()),
                G_final=float(G[-1]), D_final=float(D[-1]), t_G1=first(1.0), t_G05=first(0.5), G_min_post=float(G[t_shock + 1:].min()))
    if traj: out['G'] = G.tolist()
    return out

def job(spec_):
    algo, net, seed, hp, regime, budget, modes = spec_['algo'], tuple(spec_['net']), spec_['seed'], spec_['hp'], spec_['regime'], spec_['budget'], spec_['modes']
    t0 = time.time(); ag, env = train(algo, net, seed, hp, regime, budget); ttr = time.time() - t0
    N = NSTEPS[net[0]]; out = dict(spec=spec_, train_s=ttr)
    for md in modes:
        a2 = copy.deepcopy(ag) if md == 'online' else ag
        out[md] = deploy(a2, algo, net, seed, hp, md, N)
    return out

if __name__ == '__main__':
    t = time.time()
    r = job(dict(algo=sys.argv[1], net=['single', 3, 3], seed=0, hp=dict(lr=float(sys.argv[2]), s=float(sys.argv[3])), regime=sys.argv[4], budget=int(sys.argv[5]), modes=['frozen', 'online']))
    print(json.dumps(r, indent=1), time.time() - t)
