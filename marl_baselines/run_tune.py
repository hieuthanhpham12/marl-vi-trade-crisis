import json, time, itertools, os
HERE=os.path.dirname(os.path.abspath(__file__))
from multiprocessing import Pool
from rlrun import job
B=100000
specs=[]
for algo,lrs in [('MAPPO',[3e-4,1e-3]),('MADDPG',[5e-4,1e-3])]:
    for lr in lrs:
        for s in [0.5,1.0]:
            for seed in [100,101]:
                specs.append(dict(algo=algo,net=['single',3,3],seed=seed,hp=dict(lr=lr,s=s),regime='none',budget=B,modes=['frozen']))
specs.sort(key=lambda d: d['algo']!='MADDPG')   # longest first
t=time.time(); out=[]
with Pool(2) as p:
    for i,o in enumerate(p.imap(job,specs)):
        out.append(o); print(i+1,len(specs),round(time.time()-t),flush=True)
json.dump(out,open(os.path.join(HERE,'results','res_tune.json'),'w'))
print('done',flush=True)
