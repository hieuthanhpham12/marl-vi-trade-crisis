import json, time, sys, os
HERE=os.path.dirname(os.path.abspath(__file__))
from multiprocessing import Pool
from rlrun import job
regime=sys.argv[1]; B=int(sys.argv[2]) if len(sys.argv)>2 else 100000
hp=json.load(open(os.path.join(HERE,'best_hp.json')))
nets=[['single',2,2],['single',3,3],['mcmr',3,3,1,3]]
specs=[]
for net in nets:
    for algo in ['MADDPG','MAPPO']:
        for seed in range(10):
            specs.append(dict(algo=algo,net=net,seed=seed,hp=hp[algo],regime=regime,budget=B,modes=['frozen','online']))
specs.sort(key=lambda d: d['algo']!='MADDPG')
t=time.time(); out=[]
with Pool(2) as p:
    for i,o in enumerate(p.imap(job,specs)):
        out.append(o); print(i+1,len(specs),round(time.time()-t),flush=True)
        if (i+1)%10==0: json.dump(out,open(os.path.join(HERE,'results',f'res_main_{regime}.json'),'w'))
json.dump(out,open(os.path.join(HERE,'results',f'res_main_{regime}.json'),'w')); print('done',flush=True)
