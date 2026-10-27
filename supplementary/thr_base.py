import ext, numpy as np, json, os
from multiprocessing import Pool
NETS=[('single',2,2),('single',3,3),('mcmr',3,3,1,3)]
METHS=['PLR-original','SR-PLR-MARL','Fixed-LR','PG-fixed','EG']
def first(G,ts,th):
    idx=np.where(G[ts+1:ts+401]<=th)[0]
    return (int(idx[0])+1,True) if len(idx) else (400,False)
def job(a):
    n,m=a; M=ext.schedulers(); N=800 if n[0]=='single' else 1200; out=[]
    for s in range(30):
        net=ext.make_single(n[1],n[2]) if n[0]=='single' else ext.make_mcmr(*n[1:])
        r=ext.run_ext(M[m](),net,s,N=N,shocks=((300,'price',2.0),),sigma=0.01)
        out.append(dict(t1=first(r['G'],300,1.0),t05=first(r['G'],300,0.5),Gpre=float(r['G'][299])))
    return (n,m,out)
if __name__=='__main__':
    with Pool(2) as p: R=p.map(job,[(n,m) for n in NETS for m in METHS])
    json.dump([[list(n),m,o] for n,m,o in R],open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','thr_base.json'),'w'))
    for n,m,o in R: print(n,m,'t<=1: %.1f ok %.2f | t<=0.5: %.1f ok %.2f | Gpre %.3f'%(np.mean([x['t1'][0] for x in o]),np.mean([x['t1'][1] for x in o]),np.mean([x['t05'][0] for x in o]),np.mean([x['t05'][1] for x in o]),np.mean([x['Gpre'] for x in o])))
