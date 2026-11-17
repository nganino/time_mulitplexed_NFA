"""Two-mask PPO policy for in-situ QPI mask fine-tuning.

Why policy gradient?
--------------------
Normal training (train.py) backprops through a differentiable diffraction
propagator: PyTorch computes d(loss)/d(mask) directly.  Here we can't do
that -- the "forward model" is the real lab.  A captured camera image is
just sensor counts; there is no chain rule that lets you ask "if I change
mask pixel (i,j) by +epsilon, how does the captured image change?".

Policy gradient sidesteps this by treating the masks as *Gaussian random
variables* rather than fixed parameters.  We hold a "policy mean" (u1, u2)
and add Gaussian noise (sigma) to draw S noisy variants.  Each variant gets
displayed, captured, and scored.  Variants that scored above average were
moves "uphill"; we push (u1, u2) toward them.  This gives an unbiased
estimate of the true gradient using only forward evaluations -- no
differentiation of the optics required.

Why PPO and not plain REINFORCE?
--------------------------------
Plain REINFORCE wastes data: each batch of camera captures yields exactly
one gradient step.  PPO reuses each batch for K=4 inner update steps with
an importance-ratio clip that keeps the new policy from drifting too far
from the sampling distribution.  Same camera time, ~4x the learning.

State of (u1, u2)
-----------------
At any moment, u1 and u2 ARE the fine-tuned masks.  Initially they're loaded
from the trained checkpoint; the in-situ loop only nudges them in radians
per step.  When training ends, save them and use them as drop-in replacements
for `phase_mask_1` / `phase_mask_2`.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


# =====================================================================
# Reward: -RMSE between detector readings and targets after one
# least-squares scale + offset per state.
# =====================================================================
def compute_reward(o_pred, target, cfg):
    """o_pred: [S, A, Nf] detector readings (S mask states, A a values);
    target: [A, Nf] targets on the same a values.  Fits f = s * I + b
    per state over all (a, k) -- the model's training readout, so camera
    gain / dark offset do not matter -- then returns -mean_k RMSE_k,
    the number test.py reports.  Returns [S], higher is better."""
    P = o_pred.double()
    T = target.to(P.device).double().unsqueeze(0)                 # [1, A, Nf]
    Pc = P - P.mean(dim=(1, 2), keepdim=True)
    Tc = T - T.mean(dim=(1, 2), keepdim=True)
    s = (Pc * Tc).sum(dim=(1, 2), keepdim=True) / ((Pc ** 2).sum(dim=(1, 2), keepdim=True) + 1e-12)
    f_hat = s * Pc + T.mean(dim=(1, 2), keepdim=True)
    rmse = torch.sqrt(((f_hat - T) ** 2).mean(dim=1)).mean(dim=1)  # [S]
    return (-rmse).float()


# =====================================================================
# Agent: holds u1, u2 and the optimizer; samples noisy variants; updates
# via clipped PPO.
# =====================================================================
class PolicyGradientAgent2SLM(nn.Module):
    """Two-mask Gaussian policy with PPO updates.

    The policy is parameterized by two normal distributions:
        u1_sampled ~ N(self.u1, self.sigma1)
        u2_sampled ~ N(self.u2, self.sigma2)

    self.u1, self.u2 are the *trainable* means -- the running fine-tuned
    masks.  self.sigma1, self.sigma2 are exploration std maps; non-trainable
    here (only the schedule changes them).  Pixels live in radians.

    Freezing a mask sets that mask's sigma to 0 and removes its u from the
    optimizer's parameter list, so it stays exactly at the trained value
    while the other mask is tuned.
    """
    def __init__(self, cfg, init_mask1, init_mask2,
                 freeze_pluto=False, freeze_luna=False):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.freeze_pluto = freeze_pluto
        self.freeze_luna  = freeze_luna

        # Trainable policy means.  Shape comes from the loaded checkpoint --
        # we don't impose a shape here, so if the trained masks are 1080x1920
        # the policy lives on the same grid.  (mask_bin > 1 not implemented;
        # plug it in here if you want a coarser policy.)
        self.u1 = nn.Parameter(init_mask1.clone().to(self.device),
                               requires_grad=not freeze_pluto)
        self.u2 = nn.Parameter(init_mask2.clone().to(self.device),
                               requires_grad=not freeze_luna)

        # Exploration std (not trained).  Frozen masks get sigma=0 so every
        # sample is exactly the mean -- effectively "show the trained mask".
        s1 = torch.full_like(self.u1.data, 0.0 if freeze_pluto else cfg.sigma)
        s2 = torch.full_like(self.u2.data, 0.0 if freeze_luna  else cfg.sigma)
        self.sigma1 = nn.Parameter(s1, requires_grad=False)
        self.sigma2 = nn.Parameter(s2, requires_grad=False)

        # Optimizer touches only un-frozen parameters.
        params = [p for p in [self.u1, self.u2] if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999))

        self.epoch = 0

    # ----------------------------------------------------------------
    # Sampling: draw S noisy mask pairs and record their log-probabilities.
    # ----------------------------------------------------------------
    def sample_states(self, n_states):
        """Draw n_states perturbed mask pairs from the current policy.

        Returns:
            u1_s, u2_s : [S, H, W]  (detached samples to display on the SLMs)
            logp1, logp2 : [S, H, W]  per-pixel log-probs under the OLD policy
                                      -- needed for the PPO importance ratio
        """
        # clamp(min=1e-6) prevents log_prob from exploding when sigma is 0.
        # For frozen masks we still construct a Normal with tiny sigma but
        # then overwrite the samples with the mean and zero out the log_prob,
        # so the frozen mask contributes nothing to the PPO update.
        s1 = self.sigma1.clamp(min=1e-6)
        s2 = self.sigma2.clamp(min=1e-6)

        dist1 = Normal(self.u1, s1)
        dist2 = Normal(self.u2, s2)
        u1_s = dist1.sample((n_states,))      # [S, H, W]
        u2_s = dist2.sample((n_states,))
        logp1 = dist1.log_prob(u1_s)
        logp2 = dist2.log_prob(u2_s)

        # Frozen mask: every sample is exactly the mean, log_prob doesn't
        # matter (it gets multiplied by 0 in the PPO loss).
        if self.freeze_pluto:
            u1_s = self.u1.unsqueeze(0).expand_as(u1_s).contiguous()
            logp1 = torch.zeros_like(logp1)
        if self.freeze_luna:
            u2_s = self.u2.unsqueeze(0).expand_as(u2_s).contiguous()
            logp2 = torch.zeros_like(logp2)

        return u1_s.detach(), u2_s.detach(), logp1, logp2

    # ----------------------------------------------------------------
    # PPO update.  Reuses the batch (states, old log-probs, rewards) for
    # K inner gradient steps with importance-ratio clipping.
    # ----------------------------------------------------------------
    def update_ppo(self, states1, states2, logp1_old, logp2_old, rewards):
        """Clipped PPO update.

        Args:
            states1, states2:   [S, H, W]  the noisy mask pairs we displayed
            logp1_old, logp2_old: [S, H, W]  log-probs under the policy at
                sampling time (BEFORE this update).  Detached.
            rewards: [S]  one scalar reward per sample.

        Mechanics:
          1. Normalize rewards -> "advantage" (zero mean, unit variance).
             This is the per-sample signal for "above or below average?".
          2. For K inner steps:
              a. Re-compute log_prob of the *same samples* under the CURRENT
                 (possibly-updated) policy -- gives new_logp.
              b. ratio = exp(new_logp - logp_old).  > 1 means current policy
                 likes that sample more than the sampling policy did.
              c. surrogate1 = ratio * advantage
                 surrogate2 = clip(ratio, 1-eps, 1+eps) * advantage
                 loss = -min(surr1, surr2).mean()
                 -- the min/clip prevents the update from moving so far that
                 the old samples become useless for estimating the gradient.
              d. backprop loss, optimizer.step()
          3. Return mean reward (for logging).
        """
        cfg = self.cfg
        rewards = rewards.detach()
        baseline = rewards.mean()
        std = rewards.std() + 1e-8
        # Advantage: (reward - mean)/std.  The /std normalization keeps the
        # gradient magnitude similar across calibration objects of varying
        # difficulty -- a hard digit doesn't get a tiny update just because
        # its raw rewards are smaller than an easy grating's.
        advantage = ((rewards - baseline) / std).detach()      # [S]

        logp1_old = logp1_old.detach()
        logp2_old = logp2_old.detach()
        adv = advantage.view(-1, 1, 1, 1)                      # [S, 1, 1, 1] (masks are [S, M, H, W])

        for _ in range(cfg.K):
            # Re-evaluate log-probs under the CURRENT u1, u2.  These change
            # every inner step because optimizer.step() moves u1, u2.
            s1 = self.sigma1.clamp(min=1e-6)
            s2 = self.sigma2.clamp(min=1e-6)
            dist1 = Normal(self.u1, s1)
            dist2 = Normal(self.u2, s2)
            new_logp1 = (dist1.log_prob(states1)
                         if not self.freeze_pluto
                         else torch.zeros_like(logp1_old))
            new_logp2 = (dist2.log_prob(states2)
                         if not self.freeze_luna
                         else torch.zeros_like(logp2_old))

            # One PPO loss per non-frozen mask; average them.  Per-mask
            # treatment lets each have its own importance ratio.
            loss_terms = []
            for new_lp, old_lp, frozen in [
                    (new_logp1, logp1_old, self.freeze_pluto),
                    (new_logp2, logp2_old, self.freeze_luna)]:
                if frozen:
                    continue
                ratio = torch.exp(new_lp - old_lp)             # [S, H, W]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.ppo_eps, 1 + cfg.ppo_eps) * adv
                # -min: we want LARGE rewards.  The min of surr1, surr2 is
                # the pessimistic estimate (per the PPO paper); negating
                # makes it a minimization target.
                loss_terms.append(-torch.min(surr1, surr2).mean())

            if not loss_terms:
                break   # both masks frozen -- nothing to update
            loss = sum(loss_terms) / len(loss_terms)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return float(rewards.mean().item())

    # ----------------------------------------------------------------
    # Sigma annealing.  Wider noise early (to find a rough direction),
    # narrower noise later (to refine).  Disabled by default.
    # ----------------------------------------------------------------
    def update_sigma(self, epoch):
        cfg = self.cfg
        if not cfg.sigma_anneal:
            return
        if epoch >= cfg.anneal_epochs:
            new_s = cfg.sigma_final
        else:
            # Exponential decay from cfg.sigma to cfg.sigma_final over
            # cfg.anneal_epochs epochs.
            k = np.log(cfg.sigma / cfg.sigma_final) / cfg.anneal_epochs
            new_s = cfg.sigma_final + (cfg.sigma - cfg.sigma_final) * np.exp(-k * epoch)
        if not self.freeze_pluto:
            self.sigma1.data.fill_(new_s)
        if not self.freeze_luna:
            self.sigma2.data.fill_(new_s)

    # ----------------------------------------------------------------
    # Save state.  Just the masks + sigma + optimizer state; you can
    # restart from here if a long run is interrupted.
    # ----------------------------------------------------------------
    def save(self, path):
        torch.save({
            'epoch':     self.epoch,
            'u1':        self.u1.detach().cpu(),
            'u2':        self.u2.detach().cpu(),
            'sigma1':    self.sigma1.detach().cpu(),
            'sigma2':    self.sigma2.detach().cpu(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
