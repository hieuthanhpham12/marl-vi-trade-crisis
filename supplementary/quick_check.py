import ext, numpy as np, time
M=ext.schedulers()
t0=time.time()
for (m,n) in [(2,2),(3,3)]:
    for name in ['PLR-original','SR-PLR-MARL','Fixed-LR','PG-fixed','EG']:
        rts=[];ok=[]
        for s in range(30):
            net=ext.make_single(m,n)
            r=ext.run_ext(M[name](),net,s,N=800,shocks=((300,'price',2.0),))
            a,b=ext.g50(r['G'],300,800); ok.append(a); rts.append(b)
        print(m,n,name,'succ',np.mean(ok),'RT',round(np.mean(rts),1),'+-',round(np.std(rts,ddof=1),2),round(time.time()-t0),'s',flush=True)
