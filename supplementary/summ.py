import json,numpy as np,sys
def agg(o):
    r=o['res']; sp=o['spec']; K=len(sp['shocks'])
    out={}
    for key in ['g50','rho_s','rho_e']:
        if K==0: continue
        # mean over shocks of per-seed; also last shock
        out[key+'_succ']=float(np.mean([all(x[0] for x in d[key]) for d in r]))
        out[key+'_rt']=float(np.mean([np.mean([x[1] for x in d[key]]) for d in r]))
        out[key+'_succL']=float(np.mean([d[key][-1][0] for d in r]))
        out[key+'_rtL']=float(np.mean([d[key][-1][1] for d in r]))
        out[key+'_rtL_sd']=float(np.std([d[key][-1][1] for d in r],ddof=1))
    out['G']=float(np.mean([d['G_final'] for d in r])); out['Gsd']=float(np.std([d['G_final'] for d in r],ddof=1))
    out['D']=float(np.mean([d['D_final'] for d in r]))
    out['det']=float(np.mean([d['n_det'] for d in r])); out['false']=float(np.mean([d['n_false'] for d in r]))
    for k in ['G_drift','G_late','relerr','G_impl','G_ex','D_star','D_end']:
        if k in r[0]: out[k]=float(np.mean([d[k] for d in r]))
    return out
if __name__=='__main__':
    for o in json.load(open(sys.argv[1])):
        sp=o['spec']; a=agg(o)
        tag={k:v for k,v in sp.items() if k in('sigma','obs_sigma','miss','drift','exact','srkw') and v not in (None,0.0,False)}
        print(sp['net'],sp['method'],tag,{k:round(v,3) for k,v in a.items()})
