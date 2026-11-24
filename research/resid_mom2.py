import sys; sys.path.insert(0,"research")
exec(open("research/resid_mom_cv.py").read().split('F=ew(Zall)')[0].replace('HL=(3,10,25,1000)','HL=(1,2,3,5,10,25,1000)'))
F=ew(Zall)
# per-fold train-side features
P={}
def recent_info(d, idx, coef):
    cm=dict(zip(tm,coef)); h=d["hl"][d["hl"].i.isin(idx)&(d["hl"].bar_ix<=49)]
    w=remaining(h.bar_ix.to_numpy()); act=pd.Series((w>0).astype(float)).groupby(h.i.to_numpy()).sum().reindex(idx).fillna(0).to_numpy()
    return act
def mk(Fx, s, act, Zlast, kind):
    cols=[s]
    if kind in ("ew","ew+last","ew+inter","all"): cols += [Fx[:,k] for k in range(Fx.shape[1])]
    if kind in ("ew+last","all"): cols += [Zlast[:,k] for k in range(Zlast.shape[1])]
    if kind in ("ew+inter","all"): cols += [Fx[:,2]*(act>0), Fx[:,2]*act, s*np.abs(Fx[:,2])]
    return np.column_stack(cols)
kinds=["ew","ew+last","ew+inter","all"]
P={k:np.zeros(N) for k in kinds}; Pr={k:np.zeros(N) for k in kinds}
for a,b in folds:
    coef=fit([(ins,a,99),(ins,b,49),(oos,allo,49)],True,lam)
    Za=resid49(ins,a,coef); Fa=ew(Za); sa=signal(ins,a,coef); sb=sig[b]*gk[b]
    aa=recent_info(ins,a,coef); ab=recent_info(ins,b,coef)
    for k in kinds:
        Xa=mk(Fa,sa,aa,Za[:,-10:],k); Xb=mk(F[b],sb,ab,Zall[b][:,-10:],k)
        r=LinearRegression().fit(Xa,y[a]/gk[a]); P[k][b]=(r.predict(Xb)-r.intercept_)/gk[b]
        mu,sd=Xa.mean(0),Xa.std(0)+1e-12; rr=Ridge(alpha=50).fit((Xa-mu)/sd,y[a]/gk[a]); Pr[k][b]=(rr.predict((Xb-mu)/sd)-rr.intercept_)/gk[b]
np.save("research/pred_ew_ridge.npy", Pr["ew"])
for k in kinds:
    print(f"{k:10s} OLS 5fold={sh(P[k]):.3f} {[round(sh(P[k],b),2) for _,b in folds]}   ridge={sh(Pr[k]):.3f} {[round(sh(Pr[k],b),2) for _,b in folds]}")
