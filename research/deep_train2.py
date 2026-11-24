"""Deep models with honest early stopping (inner 10% of TRAIN sessions), seed ensembles."""
import sys, time, json; sys.path.insert(0,"research")
import numpy as np, torch, torch.nn as nn
from deep_data import build, windows
exec(open("research/deep_train.py").read().split("def run(")[0].split("d = build()")[1].replace("def sharpe","def _sh"))  # reuse model classes
d = build()
def sharpe(p, y): pnl = p*y; return float(pnl.mean()/pnl.std()*np.sqrt(252))
SC = torch.tensor([2e-3, 3e-3, 2e-3, 1e-2])
class Struct(nn.Module):
    """pred = sum_headlines effect(template) * g(age), learned age curve; + small price head"""
    def __init__(s, n, emb=16):
        super().__init__(); s.emb = nn.Embedding(n, emb, padding_idx=0); s.eff = nn.Sequential(nn.Linear(emb, 32), nn.GELU(), nn.Linear(32, 1))
        s.age = nn.Parameter(torch.zeros(50)); s.price = nn.Linear(4*5, 1)
    def forward(s, xp, xt):
        e = s.eff(s.emb(xt)).squeeze(-1) * (xt > 0)          # (n,50,4)
        news = (e.sum(2) * torch.nn.functional.softplus(s.age)).sum(1)
        return news + s.price((xp[:, -5:]/SC).flatten(1)).squeeze(-1)

def run(name, cls, aug, epochs, seed, bs=512, lr=2e-3, wd=1e-2):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    trs = d["tr"].copy(); rng.shuffle(trs); inner, fit = trs[:700], trs[700:]
    ts = list(range(49, 90)) if aug else [49]
    xp, xt, y, v = windows(d, fit, ts); ixp, ixt, iy, iv = windows(d, inner, [49]); vxp, vxt, vy, vv = windows(d, d["va"], [49])
    ys = y.std(); m = cls(d["ntmpl"]); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    xp, xt, yt = map(torch.tensor, (xp, xt, y/ys)); n = len(yt)
    best = (-9, None, None)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for k in range(n // bs):
            j = perm[k*bs:(k+1)*bs]; loss = ((m(xp[j], xt[j]) - yt[j])**2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            pi = m(torch.tensor(ixp), torch.tensor(ixt)).numpy(); pv = m(torch.tensor(vxp), torch.tensor(vxt)).numpy()
        si = sharpe(pi/iv, iy)
        if si > best[0]: best = (si, ep, pv/vv)
    return best

if __name__ == "__main__":
    name = sys.argv[1]; aug = sys.argv[2] == "1"; seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cls = globals()[name]; t0 = time.time(); preds = []; eps = []
    vy = d["y"][d["va"]]
    for s in range(seeds):
        si, ep, pv = run(name, cls, aug, 6 if aug else 40, 42+s); preds.append(pv/pv.std()); eps.append(ep)
        print(json.dumps(dict(model=name, aug=aug, seed=42+s, inner_best=round(si,2), epoch=ep, sel_sharpe=round(sharpe(pv, vy),3))), flush=True)
    ens = np.mean(preds, 0); np.save(f"research/deep2_{name}_{int(aug)}.npy", ens)
    print(json.dumps(dict(model=name, aug=aug, ENSEMBLE_sel_sharpe=round(sharpe(ens, vy),3), sec=round(time.time()-t0))), flush=True)
