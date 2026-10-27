import ext, numpy as np, json, sys, time, os
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
from multiprocessing import Pool

def mk(net):
    return ext.make_single(net[1], net[2]) if net[0]=='single' else ext.make_mcmr(*net[1:])

def seed_metrics(r, shocks, N, window=400, drift=None):
    d={}
    st=[s[0] for s in shocks]
    gs=[];rs=[];re=[]
    for k,ts in enumerate(st):
        end=st[k+1] if k+1<len(st) else N
        gs.append(ext.g50(r['G'],ts,end,window=min(window,end-ts-1)))
        rs.append(ext.rho_rec(r['rho'],ts,end,window=min(window,end-ts-1)))
        re.append(ext.rho_rec(r['rho_eval'],ts,end,window=min(window,end-ts-1)))
    d['g50']=gs; d['rho_s']=rs; d['rho_e']=re
    fl=np.where(r['flag'])[0]
    good=set()
    for ts in st: good.update(range(ts+1,ts+4))
    d['n_det']=int(sum(1 for t in fl if t in good)); d['n_false']=int(sum(1 for t in fl if t not in good))
    d['G_final']=r['G_final']; d['D_final']=r['D_final']
    if drift is not None:
        t0=drift['t0']; d['G_drift']=float(r['G'][t0:].mean()); d['G_late']=float(r['G'][N*2//3:].mean())
    return d

def job(spec):
    net0=spec['net']; M=ext.schedulers(T2=spec.get('T2',100.0),sigma=spec.get('sigma',0.01),**spec.get('srkw',{}))
    shocks=[tuple(s) for s in spec['shocks']]; N=spec['N']
    res=[]
    xs=None
    if spec.get('xstar'):
        n2=mk(net0)
        for (t,k,m) in shocks: ext.apply_shock(n2,k,m)
        xs=ext.solve_xstar(n2)
    for s in range(spec['seeds']):
        net=mk(net0)
        r=ext.run_ext(M[spec['method']](),net,s,N=N,shocks=shocks,sigma=spec.get('sigma',0.01),obs_sigma=spec.get('obs_sigma',0.0),
                      miss=spec.get('miss',0.0),drift=spec.get('drift'),exact_proj=spec.get('exact',False),xstar=xs)
        d=seed_metrics(r,shocks,N,drift=spec.get('drift'))
        if spec.get('xstar'):
            d['relerr']=float(np.linalg.norm(r['x']-xs)/np.linalg.norm(xs))
            n3=mk(net0)
            for (t,k,m) in shocks: ext.apply_shock(n3,k,m)
            d['G_impl']=n3.compute_G_lambda(r['x']); d['G_ex']=ext.G_exact(n3,r['x']); d['D_star']=float(n3.compute_D(xs))
            d['D_end']=float(n3.compute_D(r['x']))
        res.append(d)
    return dict(spec=spec,res=res)

def run_jobs(specs,fname,procs=2):
    os.makedirs(RESULTS, exist_ok=True); fname=os.path.join(RESULTS, fname)
    t0=time.time(); out=[]
    with Pool(procs) as p:
        for i,o in enumerate(p.imap(job,specs)):
            out.append(o)
            if i%10==0: print(fname,i+1,'/',len(specs),round(time.time()-t0),'s',flush=True)
    json.dump(out,open(fname,'w'))
    print(fname,'done',round(time.time()-t0),'s',flush=True)
