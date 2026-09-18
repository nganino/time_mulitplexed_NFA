'''
Time-Multiplexed NFA — training (Nonlinear Function Approximation)

Usage:
    python train.py
    python train.py --set freeze_slm=true
    python train.py --set M=5 lr_slm=5e-3
    python train.py --set Nf=200 pd_num_rows=10 pd_num_cols=20

Logs are written to /logs/.
Shared modules (dataloader, loss, wave_prop) are imported from the parent project.

NOTE: this file was rewritten in place for the NFA pivot (image classification
-> parallel nonlinear function approximation) rather than left side-by-side
with the old classification version, since a single-entry-point training
script can't sensibly host two parallel `main()`s / Trainer classes. Notable
removals rather than renames: `save_images` (classification-only diagnostic
figure -- superseded by `evaluate()`'s target-vs-approximation plot),
per-batch `valid_step` (folded into the single-pass `evaluate()`), and wandb
support (dropped per request -- tensorboard only from now on).
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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from config import init_params, recompute_derived, config_to_dict
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
        if isinstance(current, bool):
            val = val_str.lower() in ('true', '1', 'yes')
        elif isinstance(current, int):
            val = int(float(val_str))
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
        print(f'  Override: {key} = {val}')


# --------------------------------------------------------------------------- #

class TimeMultiplexedNFATrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch  = 0
        self.best_val_loss = float('inf')

        self.model     = self._init_model()
        self.criterion = FunctionApproxLoss(config).to(self.device)
        (self.opt_slm, self.opt_layers, self.opt_readout,
         self.sched_slm, self.sched_layers, self.sched_readout) = self._init_optimizers()

        if config.ckpt_to_load is not None:
            print(f'Loading checkpoint: {config.ckpt_to_load}')
            self.load(config.ckpt_to_load)

        self.is_training = True

    # ---------------------------------------------------------------------- #

    def _init_model(self):
        model = TimeMultiplexedNFA(self.config)
        model.to(self.device)
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
        print(f'  Phase keys : {model.T} x {self.config.slm_x_num}^2'
              f' = {model.T * self.config.slm_x_num ** 2:,} params')
        print(f'  Layers     : {model.num_layers} x {self.config.layer_size}^2'
              f' = {model.num_layers * self.config.layer_size ** 2:,} params')
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
                opt_slm, T_max=self.config.max_epoch, eta_min=1e-5
            )

        # Diffractive layers are always trainable now (num_layers > 0 is
        # required, see model.py) -- no more conditional creation.
        opt_layers = torch.optim.Adam(
            [self.model.layer_phases], lr=self.config.lr_layer, betas=(0.9, 0.999)
        )
        sched_layers = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_layers, T_max=self.config.max_epoch, eta_min=1e-5
        )

        # Learned affine readout (loss.py's scale/bias, replaced the old EMA
        # running Pmin/Pmax -- see loss.py's module docstring for why). Just
        # two scalars with a very direct, well-conditioned gradient; reuses
        # lr_slm's LR since there's no strong reason for a dedicated knob.
        opt_readout = torch.optim.Adam(
            self.criterion.parameters(), lr=self.config.lr_slm, betas=(0.9, 0.999)
        )
        sched_readout = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_readout, T_max=self.config.max_epoch, eta_min=1e-5
        )

        return opt_slm, opt_layers, opt_readout, sched_slm, sched_layers, sched_readout

    # ---------------------------------------------------------------------- #

    def _set_mode(self, training):
        '''Toggle train()/eval() on both the model and the loss module. The
        loss module no longer holds training-mode-dependent state (its old
        running Pmin/Pmax buffers were replaced by a learned scale/bias, see
        loss.py) -- this is kept mainly for good habit / future-proofing.'''
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
        self.opt_readout.zero_grad(set_to_none=True)

        I_vec = self.model(a)
        loss, _ = self.criterion(I_vec, target)
        loss.backward()

        slm_gnorm = (self.model.slm_phases.grad.norm().item()
                     if self.model.slm_phases.grad is not None else 0.0)
        layer_gnorm = (self.model.layer_phases.grad.norm().item()
                     if self.model.layer_phases.grad is not None else 0.0)

        if self.opt_slm is not None:
            self.opt_slm.step()
        self.opt_layers.step()
        self.opt_readout.step()

        return loss.item(), slm_gnorm, layer_gnorm

    @torch.no_grad()
    def evaluate(self, loader, tag='val', plot=True, n_show=4):
        '''
        Single pass over `loader` (typically the dense val/test grid) in eval
        mode: computes the MSE loss and per-function RMSE (paper Eq. 16-style,
        approximating the continuous integral with loader's a-samples), and
        optionally saves a target-vs-approximation curve plot for a handful of
        representative functions (Fig. 2b/2c-style: best-fit, worst-fit, and a
        couple spread across the middle of the error distribution).

        Restores whatever train/eval mode the trainer was in before returning.

        Returns (loss, per_function_rmse) -- per_function_rmse : [Nf] tensor.
        '''
        was_training = self.is_training
        self._set_mode(False)

        all_a, all_fhat, all_target = [], [], []
        total_loss, n_batches = 0.0, 0
        for a, target in loader:
            a = a.to(self.device); target = target.to(self.device)
            I_vec = self.model(a)
            loss, f_hat = self.criterion(I_vec, target)
            total_loss += loss.item(); n_batches += 1
            all_a.append(a.cpu()); all_fhat.append(f_hat.cpu()); all_target.append(target.cpu())

        if was_training:
            self._set_mode(True)

        a_all      = torch.cat(all_a)
        fhat_all   = torch.cat(all_fhat)
        target_all = torch.cat(all_target)
        rmse = FunctionApproxLoss.per_function_rmse(fhat_all, target_all)

        if plot:
            os.makedirs(self.config.image_dir, exist_ok=True)
            order = torch.argsort(a_all)
            a_sorted, fhat_sorted, target_sorted = a_all[order], fhat_all[order], target_all[order]

            Nf = rmse.shape[0]
            order_by_err = torch.argsort(rmse)
            show_idx = sorted(set(int(order_by_err[i]) for i in
                                   [0, Nf // 3, 2 * Nf // 3, Nf - 1]))[:n_show]

            fig, axes = plt.subplots(1, len(show_idx), figsize=(4 * len(show_idx), 3.2),
                                      dpi=150, squeeze=False)
            for i, k in enumerate(show_idx):
                ax = axes[0, i]
                ax.plot(a_sorted.numpy(), target_sorted[:, k].numpy(), '--',
                         color='tab:green', label='target')
                ax.plot(a_sorted.numpy(), fhat_sorted[:, k].numpy(), '-.',
                         color='tab:red', label='approx')
                ax.set_title(f'f_{k}  (RMSE={rmse[k]:.2e})', fontsize=9)
                ax.set_xlabel('a'); ax.set_ylim(-0.05, 1.05)
                if i == 0:
                    ax.legend(fontsize=7)
            fig.suptitle(f'Function approximation ({tag}, epoch {self.epoch})  '
                         f'mean RMSE={rmse.mean():.2e}  max RMSE={rmse.max():.2e}')
            fig.tight_layout()
            path = os.path.join(self.config.image_dir, f'function_approx_{tag}_epoch{self.epoch:03d}.png')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)

        return total_loss / max(n_batches, 1), rmse

    # ---------------------------------------------------------------------- #

    def _checkpoint_dict(self):
        return {
            'epoch'    : self.epoch,
            'config'   : config_to_dict(self.config),
            'model'    : self.model.state_dict(),
            'criterion': self.criterion.state_dict(),   # includes the learned scale/bias readout
            'opt_slm'  : self.opt_slm.state_dict()   if self.opt_slm   else None,
            'sched_slm': self.sched_slm.state_dict() if self.sched_slm else None,
            'opt_layers'  : self.opt_layers.state_dict(),
            'sched_layers': self.sched_layers.state_dict(),
            'opt_readout'  : self.opt_readout.state_dict(),
            'sched_readout': self.sched_readout.state_dict(),
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
        self.model.load_state_dict(ckpt['model'], strict=False)
        if 'criterion' in ckpt and ckpt['criterion'] is not None:
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

        if ckpt.get('opt_readout') is not None:
            try:
                self.opt_readout.load_state_dict(ckpt['opt_readout'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] opt_readout state skipped: {e}')

        if ckpt.get('sched_readout') is not None:
            try:
                self.sched_readout.load_state_dict(ckpt['sched_readout'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] sched_readout state skipped: {e}')

        self.epoch = ckpt['epoch']
        print(f'  Resumed from epoch {self.epoch}')

    # ---------------------------------------------------------------------- #

    def save_phase_keys(self):
        '''Grid of the learned phase-key masks, one per key (M total -- the
        time-multiplexed "wisdom of the crowd" ensembling axis).'''
        with torch.no_grad():
            masks = (torch.sigmoid(self.model.slm_phases) * 2 * np.pi).cpu()
        M = masks.shape[0]

        fig, axes = plt.subplots(1, M, figsize=(2.2 * M, 2.4), dpi=200, squeeze=False)
        im = None
        for m in range(M):
            ax = axes[0, m]
            im = ax.imshow(masks[m, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
            ax.set_title(f'key {m}', fontsize=8)
            ax.axis('off')
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(['0', 'π', '2π'])
        fig.suptitle(f'Phase-key masks  (epoch {self.epoch})')
        path = os.path.join(self.config.image_dir, f'phase_keys_epoch{self.epoch:03d}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def save_layer_masks(self):
        '''Grid of the learned diffractive-layer phase masks (one per layer,
        shared across all M phase keys).'''
        with torch.no_grad():
            masks = (torch.sigmoid(self.model.layer_phases) * 2 * np.pi).cpu()
        K = masks.shape[0]

        fig, axes = plt.subplots(1, K, figsize=(3 * K, 3), dpi=200, squeeze=False)
        im = None
        for k in range(K):
            ax = axes[0, k]
            im = ax.imshow(masks[k, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
            ax.set_title(f'layer {k}', fontsize=8)
            ax.axis('off')
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(['0', 'π', '2π'])
        fig.suptitle(f'Diffractive layer phase masks  (epoch {self.epoch})')
        path = os.path.join(self.config.image_dir, f'layer_masks_epoch{self.epoch:03d}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


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
        config.image_dir   = os.path.join(args.log_dir, 'images')
        config.model_dir   = os.path.join(args.log_dir, 'model')
        config.tfboard_dir = os.path.join(args.log_dir, 'tfboard')
    # else: config.log_dir/image_dir/model_dir/tfboard_dir were already rebuilt from
    # the (possibly overridden) run_name by recompute_derived() above -- config.py is
    # the single source of truth for where a run's output goes, no need to duplicate
    # that construction here.

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    for d in [config.image_dir, config.model_dir, config.tfboard_dir]:
        os.makedirs(d, exist_ok=True)

    # Save config to log_dir/config.json for reference
    with open(os.path.join(config.log_dir, 'config.json'), 'w') as f:
        json.dump(config_to_dict(config), f, indent=2, sort_keys=True)

    writer   = SummaryWriter(config.tfboard_dir)
    log_file = os.path.join(config.log_dir, 'LOG.txt')

    train_loader, val_loader, _, target_fn = get_function_approx_dataloaders(config)
    trainer = TimeMultiplexedNFATrainer(config)
    device  = trainer.device

    print('===> Training Start')
    print(f'Run Name: {config.run_name}')

    for epoch in range(trainer.epoch, config.max_epoch):
        trainer.epoch = epoch
        trainer._set_mode(training=True)

        run_loss = run_slm_gnorm = run_layer_gnorm = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}')
        for a, target in pbar:
            a      = a.to(device)
            target = target.to(device)
            loss, slm_gn, layer_gn = trainer.train_step(a, target)
            run_loss += loss; run_slm_gnorm += slm_gn; run_layer_gnorm += layer_gn
            pbar.set_postfix({'loss': f'{loss:.3e}'})

        n_train = len(train_loader)
        run_loss /= n_train; run_slm_gnorm /= n_train; run_layer_gnorm /= n_train

        if trainer.sched_slm is not None:
            trainer.sched_slm.step()
        trainer.sched_layers.step()
        trainer.sched_readout.step()

        writer.add_scalar('loss/train',        run_loss,        epoch)
        writer.add_scalar('key_gnorm/train',   run_slm_gnorm,   epoch)
        writer.add_scalar('layer_gnorm/train', run_layer_gnorm, epoch)

        if epoch % config.checkpoint_print == 0:
            msg = f'<epoch:{epoch:3d}> loss_train:{run_loss:.3e}\n'
            print_and_save_msg(msg, log_file)

        if epoch % config.checkpoint_save == 0:
            val_loss, rmse = trainer.evaluate(val_loader, tag='val', plot=False)

            writer.add_scalar('loss/val',      val_loss,           epoch)
            writer.add_scalar('rmse/val_mean', rmse.mean().item(), epoch)
            writer.add_scalar('rmse/val_max',  rmse.max().item(),  epoch)

            msg = (f'<epoch:{epoch:3d}> loss_val:{val_loss:.3e}  '
                   f'rmse_mean:{rmse.mean().item():.3e}  rmse_max:{rmse.max().item():.3e}\n')
            print_and_save_msg(msg, log_file)

            trainer.save()
            #trainer.save_phase_keys()
            #trainer.save_layer_masks()

            if val_loss < trainer.best_val_loss:
                trainer.best_val_loss = val_loss
                trainer.save_best()

    writer.close()
    print('===> Training Complete')


if __name__ == '__main__':
    main()
