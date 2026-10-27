import json,sys,numpy as np
from summ import agg
f=sys.argv[1]; keys=sys.argv[2].split(',')
for o in json.load(open(f)):
    sp=o['spec']; a=agg(o)
    tag=''.join(f' {k}={sp[k]}' for k in('sigma','obs_sigma','miss','drift','exact','srkw') if sp.get(k) not in (None,0.0,False))
    print(sp['exp'],sp['net'][1:],sp['method'],sp['seeds'],tag,' | '+' '.join(f'{k}={a[k]:.3g}' for k in keys if k in a))
