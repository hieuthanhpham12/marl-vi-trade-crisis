from drv import *
nets=[('single',2,2),('single',3,3),('mcmr',3,3,1,3),('mcmr',3,3,3,1),('mcmr',4,4,3,2),('mcmr',5,5,3,2)]
meths=['PLR-original','SR-PLR-MARL','Fixed-LR','Poly-decay','PG-fixed','EG']
specs=[dict(exp='A',net=n,method=m,seeds=30,N=800 if n[0]=='single' else 1200,shocks=[[300,'price',2.0]]) for n in nets for m in meths]
run_jobs(specs,'res_A.json')
