from drv import *
import sys
which=sys.argv[1]
S1=('single',3,3); M96=('mcmr',4,4,3,2)
nm=lambda n:800 if n[0]=='single' else 1200
meths5=['PLR-original','SR-PLR-MARL','Fixed-LR','PG-fixed','EG']
specs=[]
if which=='B1':
    for n in [S1,M96]:
        for sg in [0.01,0.05,0.1,0.3,1.0]:
            for m in meths5: specs.append(dict(exp='B1',net=n,method=m,seeds=30,N=nm(n),shocks=[[300,'price',2.0]],sigma=sg))
if which=='B2':
    for n in [S1,M96]:
        for og in [0.01,0.1,0.5,1.0,2.0]:
            for m in ['PLR-original','SR-PLR-MARL']: specs.append(dict(exp='B2',net=n,method=m,seeds=30,N=nm(n),shocks=[[300,'price',2.0]],obs_sigma=og))
if which=='C':
    for n in [S1,M96]:
        for pi in [0.1,0.3,0.5,0.7]:
            for m in meths5: specs.append(dict(exp='C',net=n,method=m,seeds=30,N=nm(n),shocks=[[300,'price',2.0]],miss=pi))
if which=='D':
    for m in meths5: specs.append(dict(exp='D1',net=S1,method=m,seeds=30,N=1000,shocks=[[300,'price',2.0],[600,'price',1.2]]))
    for m in meths5: specs.append(dict(exp='D2',net=M96,method=m,seeds=30,N=1300,shocks=[[300,'price',2.0],[600,'tau',2.0],[900,'cut',0.6]]))
    for c in [0,10,50,100]: specs.append(dict(exp='D3',net=M96,method='SR-PLR-MARL',seeds=30,N=900,shocks=[[300,'price',1.5],[320,'price',1.5]],srkw=dict(cooldown=c)))
    for m in ['PLR-original','Fixed-LR','PG-fixed','EG']: specs.append(dict(exp='D3',net=M96,method=m,seeds=30,N=900,shocks=[[300,'price',1.5],[320,'price',1.5]]))
if which=='E':
    for amp in [0.1,0.3]:
        for per in [100,400]:
            for m in meths5: specs.append(dict(exp='E',net=M96,method=m,seeds=30,N=1500,shocks=[],drift=dict(t0=300,amp=amp,period=per)))
if which=='F':
    for n in [('single',2,2),S1,('mcmr',3,3,1,3),M96,('mcmr',5,5,3,2)]:
        for m in ['PLR-original','SR-WarmOnly','SR-ClockOnly','SR-PLR-MARL','SR-Lock50']:
            specs.append(dict(exp='F',net=n,method=m,seeds=30,N=nm(n),shocks=[[300,'price',2.0]]))
if which=='G':
    for eta in [0.5,0.3,0.2]:
        for m in meths5:
            specs.append(dict(exp='G',net=M96,method=m,seeds=30,N=1200,shocks=[[300,'cut',eta]],xstar=True,exact=False))
        for m in meths5:
            specs.append(dict(exp='G',net=M96,method=m,seeds=8,N=1200,shocks=[[300,'cut',eta]],xstar=True,exact=True))
run_jobs(specs,f'res_{which}.json')
