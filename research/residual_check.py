import sys; sys.path.insert(0,"research")
from features import *
_, hl, close, ret = load("ins")
C=close.to_numpy(); S=close.index; sidx={s:i for i,s in enumerate(S)}
R=np.diff(np.log(C),axis=1); R=R-R.mean(0)          # R[:,j]: bar j+1 return, market-adjusted
eff=template_effects(hl,close); vm=eff.groupby("verb").f.mean(); tm=eff.groupby("tmpl").f.mean()
t=hl.tmpl.map(tm).fillna(hl.verb.map(vm)).fillna(0).to_numpy()
i=hl.session.map(sidx).to_numpy(); b=hl.bar_ix.to_numpy()
D=np.zeros_like(R)                                   # expected news drift per bar
for k in range(0,11):
    wk=0.5 if k in (0,10) else 1.0; j=b+k-1; ok=(j>=0)&(j<99)
    np.add.at(D,(i[ok],j[ok]),wk*t[ok])
E=R-D
print("var of bar returns explained by news drift: %.1f%%"%(100*(1-E.var()/R.var())))
ac=lambda X,l: np.corrcoef(X[:,:-l].ravel(),X[:,l:].ravel())[0,1]
print("residual autocorr lag1..5:",[round(ac(E,l),4) for l in range(1,6)])
print("|resid| autocorr lag1 (vol clustering):",round(ac(np.abs(E),1),4))
# best possible linear predictor of the 10-bar residual from past 49 residual bars (OOS via CV)
from sklearn.linear_model import RidgeCV; from sklearn.model_selection import cross_val_predict
X=E[:,:49]; y=E[:,49:59].sum(1)
p=cross_val_predict(RidgeCV(alphas=np.logspace(-4,2,13)),X,y,cv=5)
print("CV corr of residual-path forecast with future 10-bar residual: %.3f"%np.corrcoef(p,y)[0,1])
