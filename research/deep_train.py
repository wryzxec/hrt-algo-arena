import sys, time, json; sys.path.insert(0,"research")
import numpy as np, torch, torch.nn as nn
from deep_data import build, windows
torch.manual_seed(42); np.random.seed(42)
dev = "cpu"
d = build()
def sharpe(p, y): pnl = p*y; return float(pnl.mean()/pnl.std()*np.sqrt(252))

class Base(nn.Module):
    def __init__(self, ntmpl, emb=16, hid=64):
        super().__init__()
        self.emb = nn.Embedding(ntmpl, emb, padding_idx=0)
        self.inp = nn.Linear(4 + emb + 8, hid)
        self.pos = nn.Parameter(torch.randn(50, 8)*0.1)
    def embed(self, xp, xt):
        e = self.emb(xt).sum(2)                                    # (n,50,emb) news at each bar
        xp = xp / torch.tensor([2e-3, 3e-3, 2e-3, 1e-2])           # rough unit scaling
        pos = self.pos.unsqueeze(0).expand(xp.shape[0], -1, -1)
        return torch.relu(self.inp(torch.cat([xp, e, pos], -1)))
class GRU(Base):
    def __init__(s, n, hid=64): super().__init__(n, hid=hid); s.rnn = nn.GRU(hid, hid, 2, batch_first=True, dropout=0.1); s.out = nn.Linear(hid, 1)
    def forward(s, xp, xt): h,_ = s.rnn(s.embed(xp, xt)); return s.out(h[:, -1]).squeeze(-1)
class CNN(Base):
    def __init__(s, n, hid=64):
        super().__init__(n, hid=hid)
        s.net = nn.Sequential(nn.Conv1d(hid,hid,5,padding=4,dilation=2), nn.GELU(), nn.Conv1d(hid,hid,5,padding=8,dilation=4), nn.GELU(), nn.Conv1d(hid,hid,3,padding=1))
        s.out = nn.Linear(hid*12, 1)
    def forward(s, xp, xt): h = s.net(s.embed(xp, xt).transpose(1,2)); return s.out(h[:, :, -12:].flatten(1)).squeeze(-1)
class TF(Base):
    def __init__(s, n, hid=64):
        super().__init__(n, hid=hid)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid, 4, 128, 0.1, batch_first=True), 3); s.out = nn.Linear(hid, 1)
    def forward(s, xp, xt): return s.out(s.enc(s.embed(xp, xt))[:, -1]).squeeze(-1)
class MLP(Base):
    """bag of (template x age) -- age-aware embedding sum, then MLP"""
    def __init__(s, n, hid=64):
        super().__init__(n, hid=hid); s.age = nn.Parameter(torch.zeros(50, 1)); s.mlp = nn.Sequential(nn.Linear(16*50+4*12, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, xp, xt):
        e = s.emb(xt).sum(2)
        return s.mlp(torch.cat([e.flatten(1), (xp[:, -12:]/torch.tensor([2e-3,3e-3,2e-3,1e-2])).flatten(1)], 1)).squeeze(-1)

def run(name, cls, aug, epochs=8, bs=512, lr=2e-3):
    t0 = time.time()
    ts = list(range(49, 90)) if aug else [49]
    xp, xt, y, v = windows(d, d["tr"], ts)
    vxp, vxt, vy, vv = windows(d, d["va"], [49])
    ys = y.std(); m = cls(d["ntmpl"]); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    xp, xt, yt = map(torch.tensor, (xp, xt, y/ys)); n = len(yt)
    steps = epochs * (n // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    best, hist = -9, []
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for k in range(n // bs):
            j = perm[k*bs:(k+1)*bs]; loss = ((m(xp[j], xt[j]) - yt[j])**2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        m.eval()
        with torch.no_grad(): p = m(torch.tensor(vxp), torch.tensor(vxt)).numpy()
        hist.append((round(sharpe(p, vy),3), round(sharpe(p/vv, vy),3)))
    res = dict(model=name, aug=aug, n_train=int(n), epochs=epochs, last=hist[-1], curve=hist, sec=round(time.time()-t0))
    print(json.dumps(res), flush=True)
    np.save(f"research/deep_pred_{name}_{int(aug)}.npy", p)
    return res

if __name__ == "__main__":
    which = sys.argv[1:] or ["MLP","CNN","GRU","TF"]
    for nm in which:
        for aug in (False, True):
            run(nm, globals()[nm], aug, epochs=8 if aug else 30)
