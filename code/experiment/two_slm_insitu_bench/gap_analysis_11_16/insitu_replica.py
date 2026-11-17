"""In-silico replica of the bench in-situ PPO (runs 5 / 6) inside the FITTED bench model, plus gradient oracles.

Run from code/experiment/two_slm_insitu_bench with the qpi env:
    python gap_analysis_11_16/insitu_replica.py <experiment> [...]
Starting point: the pfaff M2 checkpoint simulated with the fitted bench (gap_analysis_11_16/fitted_sets.txt), i.e. the
model of the bench that PPO run 6 faced. Decoder corrections only (keys frozen), on b x b LUNA-pixel blocks, as the bench agent
(insitu_policy.PolicyGradientAgent2SLM): mirrored samples, sigma 0.2, Adam lr 5e-3, K = 4 clipped PPO steps, advantage =
standardized rewards, 6 of the 100 a values per epoch (a_idx = (epoch + arange(6) * 16) % 100), reward = per-function min-max
NMSE (or RMSE) over those 6 a. Optional detector noise, calibrated so the 6-a NMSE reward of a fixed mask has the bench's
std (0.0024).
Experiments:
  ppo_nmse / ppo_nmse_nonoise / ppo_rmse / ppo_nmse_25a (every state on all 25 eval a) / ppo_nmse_b16
  oracle_b32 / oracle_b16 / oracle_b1 (exact gradient of the 25-a NMSE, Adam lr 0.02) / oracle_b32_rmse
Every 5 epochs (steps): NMSE and mean RMSE on the 25 eval a (cfg.a[::4]), noise-free.  Writes replica_<exp>.csv.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

sys.path.insert(0, os.path.join(os.getcwd(), 'gap_analysis_11_16'))
import sensitivity as S
torch.set_grad_enabled(True)        # sensitivity.py switches autograd off globally (forward-only sweeps)

dev = S.dev
OUT = S.OUT
FIT = dict(kv.split('=', 1) for kv in open(os.path.join(OUT, 'fitted_sets.txt')).read().split())
A100 = torch.as_tensor(np.asarray(S.cfg.a), dtype=torch.float32, device=dev)
T100 = torch.from_numpy(np.load('targets.npy')).double().to(dev)                     # [100, 4096]
VAR = T100.var(0).clamp(min=4e-4)                                                      # full-grid variance (agent: cfg.target_var)
E25 = torch.arange(0, 100, 4, device=dev)


def load_model():
    ck = torch.load(S.CK[2], map_location='cpu', weights_only=False)
    tc = S.init_params(); S.apply_saved_config(tc, ck['config'])
    for k, v in FIT.items():
        cur = getattr(tc, k)
        setattr(tc, k, v if isinstance(cur, str) else type(cur)(v))
    tc.device = 'cuda'; S.recompute_derived(tc)
    m = S.TimeMultiplexedNFA(tc).to(dev); m.load_state_dict(ck['model'], strict=False); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def up(d, b, D):
    return d.repeat_interleave(b, -2).repeat_interleave(b, -1)[..., :D, :D] if b > 1 else d


def detectors(m, base, d, b, a, grad=False):
    """decoder = base + up(d) -> [len(a), 4096] frame-averaged detector intensities"""
    dec = base + up(d, b, base.shape[-1]).unsqueeze(1)                                 # [M, 1, D, D]
    with torch.set_grad_enabled(grad):
        return functional_call(m, {'decoder_phases': dec}, (a,)).mean(1).reshape(len(a), -1).double()


def perfunc(D, T):
    lo, hi = D.amin(-2, keepdim=True), D.amax(-2, keepdim=True)
    return T.amin(-2, keepdim=True) + (D - lo) / (hi - lo).clamp(min=1e-12) * (T.amax(-2, keepdim=True) - T.amin(-2, keepdim=True))


def nmse(D, T):           # D [..., A, Nf]
    return (((perfunc(D, T) - T) ** 2).mean(-2) / VAR).mean(-1)


def rmse(D, T):
    return ((perfunc(D, T) - T) ** 2).mean(-2).sqrt().mean(-1)


def evaluate(m, base, d, b):
    with torch.no_grad():
        D = torch.cat([detectors(m, base, d, b, A100[E25[i:i + 5]]) for i in range(0, 25, 5)])
    T = T100[E25]
    return float(nmse(D, T)), float(rmse(D, T))


def calibrate_noise(m, base, d, b, target_std=0.0024):
    """relative detector noise s (x mean detector level) that gives the 6-a NMSE reward a std of target_std"""
    idx = (torch.arange(6, device=dev) * 16) % 100
    with torch.no_grad():
        D = detectors(m, base, d, b, A100[idx]); T = T100[idx]
        lvl = D.mean()
        out = {}
        for s in (0.01, 0.02, 0.04, 0.08, 0.16):
            r = torch.stack([nmse(D + s * lvl * torch.randn_like(D), T) for _ in range(40)])
            out[s] = float(r.std())
    ss = np.array(list(out)); st = np.array([out[s] for s in ss])
    s_star = float(np.exp(np.interp(np.log(target_std), np.log(st), np.log(ss))))
    return s_star, out


def ppo(m, base, b=32, epochs=75, states=36, a_batch=6, sigma=0.2, lr=5e-3, K=4, eps=0.2, reward='nmse', noise=0.0,
        all25=False, log=None):
    Mf, Dd = base.shape[0], base.shape[-1]
    h = -(-Dd // b)
    d = torch.zeros(Mf, h, h, device=dev, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr, betas=(0.9, 0.999))
    rows = [(0,) + evaluate(m, base, d.detach(), b)]
    log(f'  ep  -1  NMSE {rows[0][1]:.4f}  RMSE {rows[0][2]:.4e}')
    for ep in range(epochs):
        idx = E25 if all25 else (ep + torch.arange(a_batch, device=dev) * (100 // a_batch)) % 100
        T = T100[idx]
        e = torch.randn((states // 2, Mf, h, h), device=dev) * sigma
        smp = (d.detach().unsqueeze(0) + torch.stack([e, -e], 1).flatten(0, 1)).detach()      # [S, M, h, w]
        with torch.no_grad():
            Ds = torch.stack([detectors(m, base, smp[s], b, A100[idx]) for s in range(states)])
            if noise > 0:
                Ds = Ds + noise * Ds.mean() * torch.randn_like(Ds)
            rew = -(nmse(Ds, T) if reward == 'nmse' else rmse(Ds, T))
        old_lp = torch.distributions.Normal(d.detach(), sigma).log_prob(smp)
        adv = ((rew - rew.mean()) / (rew.std() + 1e-8)).float().view(-1, 1, 1, 1)
        for _ in range(K):
            new_lp = torch.distributions.Normal(d, sigma).log_prob(smp)
            ratio = torch.exp(new_lp - old_lp)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            r = evaluate(m, base, d.detach(), b)
            rows.append((ep + 1,) + r)
            log(f'  ep {ep:3d}  NMSE {r[0]:.4f}  RMSE {r[1]:.4e}  rms corr {d.detach().pow(2).mean().sqrt():.3f} rad')
    return rows, d.detach()


def oracle(m, base, b=32, steps=150, lr=0.02, objective='nmse', log=None):
    """Adam on the exact gradient of the 25-a objective (two-pass: detector gradient, then chunked backprop)"""
    Mf, Dd = base.shape[0], base.shape[-1]
    h = -(-Dd // b)
    d = torch.zeros(Mf, h, h, device=dev, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)
    T = T100[E25]
    rows = [(0,) + evaluate(m, base, d.detach(), b)]
    log(f'  step  -1  NMSE {rows[0][1]:.4f}  RMSE {rows[0][2]:.4e}')
    for st in range(steps):
        with torch.no_grad():
            D = torch.cat([detectors(m, base, d, b, A100[E25[i:i + 5]]) for i in range(0, 25, 5)])
        D.requires_grad_(True)
        L = nmse(D, T) if objective == 'nmse' else rmse(D, T)
        g, = torch.autograd.grad(L, D)
        opt.zero_grad()
        for i in range(0, 25, 5):
            Di = detectors(m, base, d, b, A100[E25[i:i + 5]], grad=True)
            Di.backward(g[i:i + 5])
        opt.step()
        if (st + 1) % 5 == 0:
            r = evaluate(m, base, d.detach(), b)
            rows.append((st + 1,) + r)
            log(f'  step {st:3d}  NMSE {r[0]:.4f}  RMSE {r[1]:.4e}  rms corr {d.detach().pow(2).mean().sqrt():.3f} rad')
    return rows, d.detach()


def main():
    torch.manual_seed(0)
    m = load_model()
    base = m.decoder_phases.detach().clone()                                            # [M, 1, D, D]
    base_sq = base[:, 0]
    for exp in sys.argv[1:]:
        t0 = time.time()
        lines = []
        def log(s):
            print(f'[{exp}] {s}', flush=True); lines.append(s)
        if exp.startswith('ppo'):
            b = 16 if exp.endswith('b16') else 32
            noise = 0.0
            if 'nonoise' not in exp:
                noise, tab = calibrate_noise(m, base_sq.unsqueeze(1), torch.zeros(base.shape[0], -(-base.shape[-1] // b), -(-base.shape[-1] // b), device=dev), b)
                log(f'noise: relative detector std {noise:.4f} (6-a NMSE reward std vs s: {tab})')
            rows, d = ppo(m, base, b=b, reward='rmse' if 'rmse' in exp else 'nmse', noise=noise, all25=exp.endswith('25a'), log=log)
        else:
            b = 1 if 'b1' in exp.split('_')[1] and exp.split('_')[1] == 'b1' else (16 if 'b16' in exp else 32)
            rows, d = oracle(m, base, b=b, objective='rmse' if exp.endswith('rmse') else 'nmse',
                             steps=60 if b == 1 else 150, log=log)
        log(f'done in {time.time() - t0:.0f} s')
        np.savetxt(os.path.join(OUT, f'replica_{exp}.csv'), np.array(rows), delimiter=',', header='epoch_or_step,nmse25,rmse25',
                   comments='')
        open(os.path.join(OUT, f'replica_{exp}.log'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
