import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'icta2026'))
import sr_plr_marl_v8_mc_mr as e, numpy as np, time, pandas as pd, sys
plr=e.PLRCfg(sigma=0.01,T2=100.0); sr=e.SRCfg(sigma=0.01,T2=100.0)
M=e._v8_main_methods(plr,sr)
cfg=e.MCMRCfg(m=4,n=4,K=3,L=2); shock=e.MCMRShockCfg(episode=300,price_multiplier=2.0,all_commodities=True)
t0=time.time()
for name in ['PLR-original','SR-PLR-MARL','Fixed-LR','PG-fixed']:
    rows=[]
    for s in range(3):
        df=e.run_mc_mr_single(M[name](),cfg,shock,plr,1200,s)
        rows.append(e.v8_metrics(df,300))
    d=pd.DataFrame(rows); print(name,d[['success','RT_all','Final_G']].mean().round(3).to_dict(), round(time.time()-t0),'s',flush=True)
