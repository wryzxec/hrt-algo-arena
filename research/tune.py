import sys; sys.path.insert(0,"research")
from features import *
from sklearn.model_selection import KFold
hl, close, ret = headline_table("ins"); _,hl_full,_,_ = load("ins")
C = close.to_numpy()[:, :50]; S = close.index; y = ret.to_numpy()
lr=np.diff(np.log(C),axis=1); vol=lr[:,-20:].std(1); m3=np.log(C[:,49]/C[:,46])
sh=lambda p: (p*y).mean()/(p*y).std()*np.sqrt(252)
kf=KFold(5,shuffle=True,random_state=0); folds=list(kf.split(S))
effs=[]
for tr,te in folds:
    eff=template_effects(hl_full,close,S[tr]); vmean=eff.groupby("verb").f.mean()
    g=eff.groupby("tmpl").f; tv=eff.groupby("tmpl").verb.first().map(vmean); effs.append(((g.mean()*g.size()+tv*30)/(g.size()+30), vmean))
def news(L, k0w=0.5):
    p=np.zeros(len(S))
    for (tr,te),(tm,vm) in zip(folds,effs):
        h=hl[hl.session.isin(S[te])]; b=h.bar_ix.to_numpy(); w=np.zeros(len(h))
        for k in range(L+1):
            wk = k0w if k in (0,L) else 1.0; w+=wk*((b+k>49)&(b+k<=59))
        t=h.tmpl.map(tm).fillna(h.verb.map(vm)).fillna(0)
        p[te]=(t*w).groupby(h.session).sum().reindex(S[te]).fillna(0).to_numpy()
    return p
for L in (8,9,10,11,12): print("L",L, round(sh(news(L)),3))
p=news(10)
for a in (0,0.5,1,2): print("size / vol^a a=",a, round(sh(p/vol**a),3))
for q in (0.9,0.95,0.99): c=np.quantile(np.abs(p),q); print("clip q",q, round(sh(np.clip(p,-c,c)),3))
for beta in (0, 0.05,0.1,0.2): print("news + beta*m3 beta=",beta, round(sh(p/p.std()+beta*m3/m3.std()),3))
print("per-fold sharpe:", [round(sh_f,2) for sh_f in [((p*y)[te].mean()/(p*y)[te].std()*np.sqrt(252)) for _,te in folds]])
h1=np.arange(len(S))<5000; print("first/second half of sessions:", round(((p*y)[h1].mean()/(p*y)[h1].std())*np.sqrt(252),2), round(((p*y)[~h1].mean()/(p*y)[~h1].std())*np.sqrt(252),2))
