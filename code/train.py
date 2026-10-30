'''
Time-Multiplexed NFA — training (Nonlinear Function Approximation)

Usage:
    python train.py
    python train.py --set freeze_slm=true
    python train.py --set M=5 lr_slm=5e-3
    python train.py --set Np=25 Nf=1024 num_layers=3 M=2   # sizes follow automatically
    python train.py --set device_profile=lab num_layers=0 decoder=true Nf=1024 Np=64 M=2 batch_size=16
    python train.py --set device_profile=lab ... slm_phase_bits=8   # 8-bit SLM phase
    python train.py --set encoding_mode=split Np=9 M=4      # 36 harmonics, 9 per frame

--set KEY=VALUE overrides any config.py value. Design values (layer_size,
phase_key_size, N_trainable_features) and the device-profile values are re-derived from their
formulas after the overrides are applied -- unless you --set them yourself, in
which case your value sticks. The derived values are printed (and written to
LOG.txt) at startup, tagged [set] or [derived]. See config.py "Derive unless you
set it" and README.md.

Runs are written to code/logs/<run_name>/ (or --log_dir): config.json, LOG.txt, model/ (best.pth + the last
epoch=*.pth checkpoints), tfboard/. Figures come from test.py.

Optimised: phase keys, diffractive layers and (E+D) decoder phases, each with its own Adam + cosine schedule, plus the
training-only readout scale / offset (train_readout affine, the default; discarded after training). Every val / test
number uses the min-max readout (loss.py). Checkpoints written before 2026-10-23 are rejected on load.
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch

from config import (init_params, recompute_derived, config_to_dict,
                    mark_explicit, describe_design, CHECKPOINT_READOUT, check_checkpoint_format)
from model import TimeMultiplexedNFA
from dataloader import get_function_approx_dataloaders
from loss import FunctionApproxLoss


# --------------------------------------------------------------------------- #

def _apply_overrides(config, overrides):
    '''Apply KEY=VALUE strings to config, preserving the original type.'''
    for kv in overrides:
        key, val_str = kv.split('=', 1)
        if not hasattr(config, key):
            raise ValueError(
                f"--set {key}={val_str}: '{key}' is not a config attribute "
            )
        current = getattr(config, key, None)
        if val_str == 'auto':                       # e.g. N_sim=auto (sized from the geometry)
            val = val_str
        elif isinstance(current, bool):
            val = val_str.lower() in ('true', '1', 'yes')
        elif isinstance(current, int):
            val = int(float(val_str))
            if val != float(val_str):               # e.g. 0.01 onto an int default would silently become 0
                raise ValueError(f'--set {key}={val_str}: {key} is an int (default {current!r}), got a fractional value')
        elif isinstance(current, float):
            val = float(val_str)
        elif current is None:
            # Parameters that default to None have no type to preserve. Coerce to
            # int, then float, and only then leave it a string.
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
        else:
            val = val_str
        setattr(config, key, val)
        mark_explicit(config, [key])   # never re-derived by recompute_derived()
        print(f'  Override: {key} = {val}')


# --------------------------------------------------------------------------- #

class TimeMultiplexedNFATrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch  = 0
        self.best_val_loss = float('inf')

        self.model     = self._init_model()
        self.criterion = FunctionApproxLoss(config).to(self.device)   # min-max readout (+ training-only affine)
        (self.opt_slm, self.opt_layers,
         self.sched_slm, self.sched_layers) = self._init_optimizers()
        # Training-only learned readout (train_readout 'affine'): its scale / offset use lr_slm and its cosine schedule
        self.opt_readout = self.sched_readout = None
        if list(self.criterion.parameters()):
            self.opt_readout = torch.optim.Adam(self.criterion.parameters(), lr=config.lr_slm, betas=(0.9, 0.999))
            self.sched_readout = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_readout, T_max=config.max_epoch, eta_min=config.lr_slm * config.lr_min_ratio)
        # Decoder (E+D, model.py): own Adam + cosine schedule, only when config.decoder is on
        self.opt_decoder = self.sched_decoder = None
        if self.model.decoder:
            self.opt_decoder = torch.optim.Adam([self.model.decoder_phases], lr=config.lr_decoder, betas=(0.9, 0.999))
            self.sched_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_decoder, T_max=config.max_epoch, eta_min=config.lr_decoder * config.lr_min_ratio)

        if config.ckpt_to_load is not None:
            print(f'Loading checkpoint: {config.ckpt_to_load}')
            self.load(config.ckpt_to_load)

        self.is_training = True

    # ---------------------------------------------------------------------- #

    def _init_model(self):
        model = TimeMultiplexedNFA(self.config)
        model.to(self.device)
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
        print(f'  Phase keys : {model.T} x {self.config.phase_key_size}^2'
              f' = {model.T * self.config.phase_key_size ** 2:,} params')
        print(f'  Layers     : {model.num_layers} x {self.config.layer_size}^2'
              f' = {model.num_layers * self.config.layer_size ** 2:,} params')
        if model.decoder:
            n_px = int(model.decoder_mask_px.sum().item())
            print(f'  Decoder    : {model.decoder_size}^2 SLM px, {model.n_decoders} x {n_px:,} trainable SLM px'
                  f' ({"one per key" if model.decoder_per_key else "shared by all keys"})')
        return model

    def _init_optimizers(self):
        freeze_slm = bool(getattr(self.config, 'freeze_slm', False))

        if freeze_slm:
            self.model.slm_phases.requires_grad_(False)
            opt_slm   = None
            sched_slm = None
            print('  Phase-key plane frozen.')
        else:
            opt_slm = torch.optim.Adam(
                [self.model.slm_phases], lr=self.config.lr_slm, betas=(0.9, 0.999)
            )
            sched_slm = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_slm, T_max=self.config.max_epoch, eta_min=self.config.lr_slm * self.config.lr_min_ratio
            )

        # Diffractive layers (an empty parameter at K = 0, so this optimizer is then a no-op).
        opt_layers = torch.optim.Adam(
            [self.model.layer_phases], lr=self.config.lr_layer, betas=(0.9, 0.999)
        )
        sched_layers = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_layers, T_max=self.config.max_epoch, eta_min=self.config.lr_layer * self.config.lr_min_ratio
        )

        return opt_slm, opt_layers, sched_slm, sched_layers

    # ---------------------------------------------------------------------- #

    def _set_mode(self, training):
        '''Toggle train()/eval() on both the model and the loss module (the loss module holds no
        training-mode-dependent state; the model's measurement noise is training-only).'''
        if training:
            self.model.train(); self.criterion.train()
        else:
            self.model.eval(); self.criterion.eval()
        self.is_training = training

    def train_step(self, a, target):
        if not self.is_training:
            self._set_mode(True)

        if self.opt_slm is not None:
            self.opt_slm.zero_grad(set_to_none=True)
        self.opt_layers.zero_grad(set_to_none=True)
        if self.opt_decoder is not None:
            self.opt_decoder.zero_grad(set_to_none=True)
        if self.opt_readout is not None:
            self.opt_readout.zero_grad(set_to_none=True)

        # Gradient accumulation: k equal micro-batches, each loss scaled by 1/k, at 1/k of the memory. Each micro-batch
        # min-max normalizes with its own Pmin / Pmax, so the summed gradient is close to, not exactly, the full-batch one
        k = int(getattr(self.config, 'grad_accum_steps', 1))
        assert a.shape[0] % k == 0, f'batch {a.shape[0]} not divisible by grad_accum_steps {k}'
        loss = 0.0
        stats = []                                                   # (Pmin, Pmax, target min, target max) per micro-batch
        for a_mb, t_mb in zip(a.chunk(k), target.chunk(k)):
            I_vec = self.model(a_mb)
            mb_loss, _ = self.criterion(I_vec, t_mb)
            with torch.no_grad():                                    # batch statistics the min-max readout used
                P = self.criterion._detector_to_function(I_vec)
                stats.append((P.min().item(), P.max().item(), t_mb.min().item(), t_mb.max().item()))
            (mb_loss / k).backward()
            loss = loss + mb_loss.detach() / k

        slm_gnorm = (self.model.slm_phases.grad.norm().item()
                     if self.model.slm_phases.grad is not None else 0.0)
        layer_gnorm = (self.model.layer_phases.grad.norm().item()
                     if self.model.layer_phases.grad is not None else 0.0)

        if self.opt_slm is not None:
            self.opt_slm.step()
        self.opt_layers.step()
        if self.opt_decoder is not None:
            self.opt_decoder.step()
        if self.opt_readout is not None:
            self.opt_readout.step()

        return loss.item(), slm_gnorm, layer_gnorm, stats

    @torch.no_grad()
    def calibrate_minmax(self):
        '''(Pmin, Pmax) of the frame-averaged intensity over the dense val a-grid and all detectors, eval mode, in chunks of
        batch_size -- the constants train_readout 'globalminmax' trains against for one epoch (the test's normalisation).'''
        was_training = self.is_training
        self._set_mode(False)
        a = torch.linspace(self.config.a_min, self.config.a_max, self.config.val_a_grid_size, device=self.device)
        pmin, pmax = float('inf'), float('-inf')
        for chunk in a.split(int(self.config.batch_size)):
            P = self.criterion._detector_to_function(self.model(chunk))
            pmin, pmax = min(pmin, P.min().item()), max(pmax, P.max().item())
        if was_training:
            self._set_mode(True)
        return (torch.tensor(pmin, device=self.device), torch.tensor(pmax, device=self.device))

    @torch.no_grad()
    def evaluate(self, loader):
        '''
        Single pass over `loader` (typically the dense val/test grid) in eval
        mode: min-max normalizes with Pmin / Pmax over the WHOLE grid and all
        detectors, then computes the MSE loss and per-function RMSE (paper
        Eq. 16-style, approximating the continuous integral with loader's a-samples).

        Restores whatever train/eval mode the trainer was in before returning.

        Returns (loss, per_function_rmse, diffraction_efficiency, (Pmin, Pmax)) --
        per_function_rmse : [Nf] tensor; diffraction_efficiency : float, mean
        over all samples and phase keys (diagnostic only, see loss.py); (Pmin, Pmax)
        of the frame-averaged intensity over the whole grid.
        '''
        was_training = self.is_training
        self._set_mode(False)

        all_I, all_target = [], []
        de_sum, de_count = 0.0, 0
        incident = self.model.incident_energy()
        for a, target in loader:
            a = a.to(self.device)
            I_vec = self.model(a)
            de = self.criterion.diffraction_efficiency(I_vec, incident)   # [B, T]
            de_sum += de.sum().item(); de_count += de.numel()
            all_I.append(self.criterion._detector_to_function(I_vec).cpu()); all_target.append(target)

        if was_training:
            self._set_mode(True)

        I_all      = torch.cat(all_I)                               # [N, Nf] frame-averaged intensity
        target_all = torch.cat(all_target)
        # targets min-max'd over the SAME grid as f_hat (their 2001-point calibration misses the grid's true range)
        target_all = self.criterion.normalize(target_all, *self.criterion.minmax_range(target_all))
        fhat_all   = self.criterion.normalize(I_all, *self.criterion.minmax_range(I_all))
        loss = torch.mean((fhat_all - target_all) ** 2).item()
        rmse = FunctionApproxLoss.per_function_rmse(fhat_all, target_all)

        return loss, rmse, de_sum / max(de_count, 1), tuple(float(v) for v in self.criterion.minmax_range(I_all))

    # ---------------------------------------------------------------------- #

    def _checkpoint_dict(self):
        return {
            'epoch'    : self.epoch,
            'best_val_loss': self.best_val_loss,
            'config'   : config_to_dict(self.config),
            'model'    : self.model.state_dict(),
            'readout'  : CHECKPOINT_READOUT,   # format marker; checkpoints without it are rejected on load
            'opt_slm'  : self.opt_slm.state_dict()   if self.opt_slm   else None,
            'sched_slm': self.sched_slm.state_dict() if self.sched_slm else None,
            'opt_layers'  : self.opt_layers.state_dict(),
            'sched_layers': self.sched_layers.state_dict(),
            'opt_decoder'  : self.opt_decoder.state_dict()   if self.opt_decoder   else None,
            'sched_decoder': self.sched_decoder.state_dict() if self.sched_decoder else None,
            'criterion'    : self.criterion.state_dict(),   # training-only readout (empty for train_readout minmax)
            'opt_readout'  : self.opt_readout.state_dict()   if self.opt_readout   else None,
            'sched_readout': self.sched_readout.state_dict() if self.sched_readout else None,
        }

    def save(self, max_keep=5):
        os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, f'epoch={self.epoch:03d}.pth')
        torch.save(self._checkpoint_dict(), path)

        all_ckpts = sorted(
            glob.glob(os.path.join(self.config.model_dir, 'epoch=*.pth')),
            key=os.path.getmtime, reverse=True
        )
        for old in all_ckpts[max_keep:]:
            os.remove(old)

    def save_best(self):
        os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, 'best.pth')
        torch.save(self._checkpoint_dict(), path)
        print(f'  best.pth updated (val_loss={self.best_val_loss:.3e}, epoch={self.epoch})')

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        check_checkpoint_format(ckpt, ckpt_path)
        # Shared and per-key decoders are not interchangeable: no warm start / resume across the two
        dec_saved = ckpt['model'].get('decoder_phases')
        if self.model.decoder and dec_saved is not None and dec_saved.shape[0] != self.model.n_decoders:
            raise ValueError(f'{ckpt_path} has {dec_saved.shape[0]} decoder(s), this model has '
                             f'{self.model.n_decoders} (decoder_per_key={self.model.decoder_per_key}); '
                             'loading a shared-decoder checkpoint into per-key decoders (or back) is not supported')
        self.model.load_state_dict(ckpt['model'], strict=False)
        if ckpt.get('criterion'):                     # training-only readout (train_readout affine)
            self.criterion.load_state_dict(ckpt['criterion'], strict=False)

        # Weights-only: start a fresh fine-tune from the loaded weights, ignoring the
        # checkpoint's epoch / optimizer / scheduler (fresh cosine schedule from epoch 0).
        if bool(getattr(self.config, 'load_weights_only', False)):
            print('  load_weights_only: loaded model weights only; fresh fine-tune '
                  '(epoch 0, new optimizer + LR schedule)')
            return

        if self.opt_slm is not None and ckpt.get('opt_slm') is not None:
            try:
                self.opt_slm.load_state_dict(ckpt['opt_slm'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] opt_slm state skipped: {e}')

        if self.sched_slm is not None and ckpt.get('sched_slm') is not None:
            try:
                self.sched_slm.load_state_dict(ckpt['sched_slm'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] sched_slm state skipped: {e}')

        if ckpt.get('opt_layers') is not None:
            try:
                self.opt_layers.load_state_dict(ckpt['opt_layers'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] opt_layers state skipped: {e}')

        if ckpt.get('sched_layers') is not None:
            try:
                self.sched_layers.load_state_dict(ckpt['sched_layers'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] sched_layers state skipped: {e}')

        for name in ('opt_decoder', 'sched_decoder', 'opt_readout', 'sched_readout'):
            obj = getattr(self, name)
            if obj is not None and ckpt.get(name) is not None:
                try:
                    obj.load_state_dict(ckpt[name])
                except (ValueError, RuntimeError) as e:
                    print(f'  [warning] {name} state skipped: {e}')

        # the checkpoint was written after epoch ckpt['epoch'] finished (and after its scheduler step)
        self.epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'  Resumed after epoch {ckpt["epoch"]}, continuing at epoch {self.epoch}')

    # ---------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #

def print_and_save_msg(msg, filepath):
    print(msg, end='')
    with open(filepath, 'a') as f:
        f.write(msg)


def main():
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--set', nargs='*', default=[], dest='overrides',
                        metavar='KEY=VALUE')
    args, _ = parser.parse_known_args()

    config = init_params()

    if args.overrides:
        print('Applying config overrides:')
        _apply_overrides(config, args.overrides)
        recompute_derived(config)

    if args.log_dir:
        # Explicit override (e.g. resuming into an exact existing run folder) --
        # takes precedence over whatever recompute_derived() just built from run_name.
        config.log_dir     = args.log_dir
        config.model_dir   = os.path.join(args.log_dir, 'model')
        config.tfboard_dir = os.path.join(args.log_dir, 'tfboard')
    # else: config.log_dir/model_dir/tfboard_dir were already rebuilt from
    # the (possibly overridden) run_name by recompute_derived() above -- config.py is
    # the single source of truth for where a run's output goes, no need to duplicate
    # that construction here.

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    for d in [config.model_dir, config.tfboard_dir]:
        os.makedirs(d, exist_ok=True)

    # Save config to log_dir/config.json for reference
    with open(os.path.join(config.log_dir, 'config.json'), 'w') as f:
        json.dump(config_to_dict(config), f, indent=2, sort_keys=True)

    writer   = SummaryWriter(config.tfboard_dir)
    log_file = os.path.join(config.log_dir, 'LOG.txt')

    train_loader, val_loader, _, _ = get_function_approx_dataloaders(config)
    trainer = TimeMultiplexedNFATrainer(config)
    device  = trainer.device

    print('===> Training Start')
    print(f'Run Name: {config.run_name}')
    print_and_save_msg('Design values ([set] = from --set, [derived] = from formula):\n'
                       + describe_design(config) + '\n', log_file)

    for epoch in range(trainer.epoch, config.max_epoch):
        trainer.epoch = epoch
        if trainer.criterion.train_readout == 'globalminmax':      # Pmin / Pmax of the full a-grid, fixed for this epoch
            trainer.criterion.train_p_range = trainer.calibrate_minmax()
            writer.add_scalar('minmax/calib_Pmin', float(trainer.criterion.train_p_range[0]), epoch)
            writer.add_scalar('minmax/calib_Pmax', float(trainer.criterion.train_p_range[1]), epoch)
        trainer._set_mode(training=True)

        run_loss = run_slm_gnorm = run_layer_gnorm = 0.0
        batch_stats = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}')
        for a, target in pbar:
            a      = a.to(device)
            target = target.to(device)
            loss, slm_gn, layer_gn, st = trainer.train_step(a, target)
            batch_stats += st
            run_loss += loss; run_slm_gnorm += slm_gn; run_layer_gnorm += layer_gn
            pbar.set_postfix({'loss': f'{loss:.3e}'})

        n_train = len(train_loader)
        run_loss /= n_train; run_slm_gnorm /= n_train; run_layer_gnorm /= n_train

        if trainer.sched_slm is not None:
            trainer.sched_slm.step()
        trainer.sched_layers.step()
        if trainer.sched_decoder is not None:
            trainer.sched_decoder.step()
        if trainer.sched_readout is not None:
            trainer.sched_readout.step()
            writer.add_scalar('readout_train/scale', float(torch.nn.functional.softplus(trainer.criterion._raw_scale)), epoch)
            writer.add_scalar('readout_train/bias', float(trainer.criterion.bias), epoch)

        writer.add_scalar('loss/train',        run_loss,        epoch)
        writer.add_scalar('key_gnorm/train',   run_slm_gnorm,   epoch)
        writer.add_scalar('layer_gnorm/train', run_layer_gnorm, epoch)
        # Min-max batch statistics (per micro-batch, epoch mean and relative spread): how much the per-batch
        # normalisation fluctuates, and how far a batch's targets are from the global [0, 1] range
        bs = np.array(batch_stats)                                   # [n, 4]: Pmin, Pmax, target min, target max
        writer.add_scalar('minmax/batch_Pmax_mean', bs[:, 1].mean(), epoch)
        writer.add_scalar('minmax/batch_Pmin_mean', bs[:, 0].mean(), epoch)
        writer.add_scalar('minmax/batch_Pmax_rel_std', bs[:, 1].std() / max(abs(bs[:, 1].mean()), 1e-30), epoch)
        writer.add_scalar('minmax/batch_Pmin_over_Pmax_mean', (bs[:, 0] / np.maximum(bs[:, 1], 1e-30)).mean(), epoch)
        writer.add_scalar('minmax/batch_target_max_mean', bs[:, 3].mean(), epoch)
        writer.add_scalar('minmax/batch_target_max_min', bs[:, 3].min(), epoch)
        writer.add_scalar('minmax/batch_target_min_mean', bs[:, 2].mean(), epoch)

        if epoch % config.checkpoint_print == 0:
            msg = f'<epoch:{epoch:3d}> loss_train:{run_loss:.3e}\n'
            print_and_save_msg(msg, log_file)

        if epoch % config.checkpoint_save == 0 or epoch == config.max_epoch - 1:
            val_loss, rmse, val_de, (val_pmin, val_pmax) = trainer.evaluate(val_loader)
            writer.add_scalar('minmax/val_Pmax', val_pmax, epoch)
            writer.add_scalar('minmax/val_Pmin', val_pmin, epoch)
            writer.add_scalar('minmax/batch_Pmax_over_val_Pmax', bs[:, 1].mean() / max(val_pmax, 1e-30), epoch)

            writer.add_scalar('loss/val',      val_loss,           epoch)
            writer.add_scalar('rmse/val_mean', rmse.mean().item(), epoch)
            writer.add_scalar('rmse/val_max',  rmse.max().item(),  epoch)
            writer.add_scalar('diffraction_efficiency/val', val_de, epoch)

            msg = (f'<epoch:{epoch:3d}> loss_val:{val_loss:.3e}  '
                   f'rmse_mean:{rmse.mean().item():.3e}  rmse_max:{rmse.max().item():.3e}  '
                   f'diff_eff:{val_de:.3e}\n')
            print_and_save_msg(msg, log_file)

            trainer.save()

            if val_loss < trainer.best_val_loss:
                trainer.best_val_loss = val_loss
                trainer.save_best()

    writer.close()
    print('===> Training Complete')


if __name__ == '__main__':
    main()
