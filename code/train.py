'''
Time-Multiplexed Classifier — training

Usage:
    python train.py
    python train.py --set freeze_slm=true
    python train.py --set softmax_T=0.05 lr_slm=5e-3
    python train.py --set C=4

Logs are written to /logs/.
Shared modules (dataloader, loss, wave_prop) are imported from the parent project.
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
import wandb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from config import init_params, recompute_derived, config_to_dict
from model import TimeMultiplexedClassifier
from dataloader import get_dataloaders
from loss import ClassificationLoss


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
            # Parameters that default to None (train_samples, mnist_cap, ...) have no
            # type to preserve. Coerce to int, then float, and only then leave it a
            # string -- otherwise `--set train_samples=200` stays "200" and fails on a slice.
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

class TimeMultiplexedClassifierTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch  = 0
        self.best_val_acc = 0.0

        self.model     = self._init_model()
        self.opt_slm, self.opt_layers, self.sched_slm, self.sched_layers = self._init_optimizers()
        self.criterion = ClassificationLoss(config).to(self.device)

        if config.ckpt_to_load is not None:
            print(f'Loading checkpoint: {config.ckpt_to_load}')
            self.load(config.ckpt_to_load)

        self.is_training = True

    # ---------------------------------------------------------------------- #

    def _init_model(self):
        model = TimeMultiplexedClassifier(self.config)
        model.to(self.device)
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
        print(f'  SLM masks  : {model.C} countries × {model.M} members = {model.T} total'
              f'  × {self.config.slm_x_num}²  = {model.T * self.config.slm_x_num ** 2:,} params')
        print(f'  Layers  : {model.num_layers} × {self.config.layer_size}²  = {model.num_layers * self.config.layer_size ** 2:,} params')
        return model

    def _init_optimizers(self):
        freeze_slm = bool(getattr(self.config, 'freeze_slm', False))

        if freeze_slm:
            self.model.slm_phases.requires_grad_(False)
            opt_slm   = None
            sched_slm = None
            print('  SLM masks frozen.')
            # NOTE: slm_phases is currently the model's only learnable parameter, so
            # freeze_slm=True means nothing trains this run. Kept around for when
            # diffractive layers (planned: num_layers = 1..8 instead of plain
            # free-space propagation) give this flag something else to leave trainable.
        else:
            opt_slm = torch.optim.Adam(
                [self.model.slm_phases], lr=self.config.lr_slm, betas=(0.9, 0.999)
            )
            sched_slm = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_slm, T_max=self.config.max_epoch, eta_min=1e-5
            )

        opt_layers = sched_layers = None
        if (self.config.num_layers > 0):
            opt_layers = torch.optim.Adam(
                [self.model.layer_phases], lr=self.config.lr_layer, betas=(0.9, 0.999)
            )
            sched_layers = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_layers, T_max=self.config.max_epoch, eta_min=1e-5
            )

        return opt_slm, opt_layers, sched_slm, sched_layers

    # ---------------------------------------------------------------------- #

    def train_step(self, phi_obj, target):
        if not self.is_training:
            self.model.train()
            self.is_training = True

        if self.opt_slm is not None:
            self.opt_slm.zero_grad(set_to_none=True)
        if self.opt_layers is not None:
            self.opt_layers.zero_grad(set_to_none=True)

        I_vec = self.model(phi_obj)
        loss, _, pred, _ = self.criterion(I_vec, target)

        loss.backward()

        slm_gnorm = (self.model.slm_phases.grad.norm().item()
                     if self.model.slm_phases.grad is not None else 0.0)
        layer_gnorm = (self.model.layer_phases.grad.norm().item()
                     if self.model.layer_phases.grad is not None else 0.0)

        if self.opt_slm is not None:
            self.opt_slm.step()
        if self.opt_layers is not None:
            self.opt_layers.step()

        acc = (pred == target).float().mean().item()
        return loss.item(), acc, slm_gnorm, layer_gnorm

    def valid_step(self, phi_obj, target):
        if self.is_training:
            self.model.eval()
            self.is_training = False

        I_vec = self.model(phi_obj)
        loss, _, pred, _ = self.criterion(I_vec, target)
        acc = (pred == target).float().mean().item()

        return loss.item(), acc

    # ---------------------------------------------------------------------- #

    def _checkpoint_dict(self):
        return {
            'epoch'    : self.epoch,
            'config'   : config_to_dict(self.config),
            'model'    : self.model.state_dict(),
            'opt_slm'  : self.opt_slm.state_dict()   if self.opt_slm   else None,
            'sched_slm': self.sched_slm.state_dict() if self.sched_slm else None,
            'opt_layers'  : self.opt_layers.state_dict()   if self.opt_layers   else None,
            'sched_layers': self.sched_layers.state_dict() if self.sched_layers else None,
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
        print(f'  best.pth updated (val_acc={self.best_val_acc:.4f}, epoch={self.epoch})')

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'], strict=False)

        # Weights-only: start a fresh fine-tune from the loaded SLM weights, ignoring the
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

        if self.opt_layers is not None and ckpt.get('opt_layers') is not None:
            try:
                self.opt_layers.load_state_dict(ckpt['opt_layers'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] opt_layers state skipped: {e}')

        if self.sched_layers is not None and ckpt.get('sched_layers') is not None:
            try:
                self.sched_layers.load_state_dict(ckpt['sched_layers'])
            except (ValueError, RuntimeError) as e:
                print(f'  [warning] sched_layers state skipped: {e}')

        self.epoch = ckpt['epoch']
        print(f'  Resumed from epoch {self.epoch}')

    # ---------------------------------------------------------------------- #

    def save_images(self, phi_obj, label, tag='val', n_show=1):
        '''
        Classification diagnostic figure, first sample in the batch only.
        One row per country r (0..C-1), 4 columns each:
          col 0: input phase image (0..input_phase_max)
          col 1: grouped bar chart -- raw positive vs negative detector intensity for
                 country r, summed over that country's M members only, per class
          col 2: bar chart -- country r's own normalized differential contrast per
                 class ((I+ - I-)/(I+ + I-), true class highlighted)
          col 3: I_ccd for country r (summed over that country's M members only),
                 cropped around the array, with the photodiode regions overlaid
                 (red = positive detector, blue = negative detector)
        Final row (spans all 4 columns): the aggregated (summed-over-countries) class
        scores, true class highlighted, prediction marked.
        '''
        os.makedirs(self.config.image_dir, exist_ok=True)

        was_training = self.is_training
        if was_training:
            self.model.eval()

        n_show = 1
        with torch.no_grad():
            # evaluate only the first sample in the batch
            I_vec, I_ccd = self.model(phi_obj, return_field=True)

            # Diagnostic guard: a NaN/Inf reaching matplotlib's Agg backend (imshow /
            # colorbar / savefig) can crash the process natively, below Python's
            # exception handling -- no traceback, nothing catchable by a try/except
            # around this function. Check finiteness here first so a bad value is
            # reported and skipped instead of silently killing the run.
            non_finite = []
            if not torch.isfinite(I_ccd).all():
                non_finite.append('I_ccd')
            if not torch.isfinite(I_vec).all():
                non_finite.append('I_vec')
            if not torch.isfinite(self.model.slm_phases).all():
                non_finite.append('slm_phases')
            if self.model.num_layers > 0 and not torch.isfinite(self.model.layer_phases).all():
                non_finite.append('layer_phases')
            if non_finite:
                print(f'  [save_images] non-finite values in {non_finite} at epoch '
                      f'{self.epoch} -- skipping plot')
                if was_training:
                    self.model.train()
                return

            aggregated_scores, per_country_scores, I_pos_country, I_neg_country = self.criterion.scores(I_vec)
            pred = aggregated_scores.argmax(dim=1)
            country_contrast = per_country_scores * self.criterion.softmax_T   # undo T-scaling -> [-1, 1] per country
            agg_contrast     = aggregated_scores * self.criterion.softmax_T    # undo T-scaling -> sum of C such contrasts

        if was_training:
            self.model.train()

        # Prepare numpy arrays for plotting (first sample only)
        C, M   = self.criterion.C, self.criterion.M
        half, num_classes = self.criterion.half, self.criterion.num_classes
        pd_px = self.model.pd_px
        half_px = pd_px // 2
        N_sim = self.config.N_sim

        label_np = label.detach().cpu().numpy()[0]
        pred_np  = pred.detach().cpu().numpy()[0]
        class_x  = np.arange(num_classes)

        I_pos_np = I_pos_country.detach().cpu().numpy()[0]        # [C, half, cols]
        I_neg_np = I_neg_country.detach().cpu().numpy()[0]        # [C, half, cols]
        country_contrast_np = country_contrast.detach().cpu().numpy()[0]   # [C, num_classes]
        agg_contrast_np     = agg_contrast.detach().cpu().numpy()[0]       # [num_classes]

        # Per-country CCD sums (summed over that country's M members only)
        B_ = I_ccd.shape[0]
        I_ccd_country_np = I_ccd.view(B_, C, M, N_sim, N_sim).sum(dim=2).detach().cpu().numpy()[0]  # [C, N_sim, N_sim]

        # Crop window around the detector array (geometry is the same for every country)
        margin = max(self.model.pd_row_spacing, self.model.pd_col_spacing)
        y0 = max(0, min(self.model.pd_centers_y) - half_px - margin)
        y1 = min(N_sim, max(self.model.pd_centers_y) + half_px + margin)
        x0 = max(0, min(self.model.pd_centers_x) - half_px - margin)
        x1 = min(N_sim, max(self.model.pd_centers_x) + half_px + margin)

        img = phi_obj[0, 0].detach().cpu().numpy()
        input_phase_max = float(getattr(self.config, 'input_phase_max', np.pi))
        rows_pd, cols_pd = self.model.pd_num_rows, self.model.pd_num_cols

        fig = plt.figure(figsize=(16, 3.6 * C + 3.2), dpi=130)
        gs = fig.add_gridspec(C + 1, 4, height_ratios=[1] * C + [1.15])

        for r in range(C):
            # Col 0: input image (identical every row; 0..input_phase_max colormap)
            ax = fig.add_subplot(gs[r, 0])
            im = ax.imshow(img, cmap='gray', vmin=0, vmax=input_phase_max)
            ax.set_title(f'input  (country {r}, Class={label_np})', fontsize=9)
            ax.axis('off')
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
            cbar.set_ticks([0, input_phase_max])
            cbar.set_ticklabels(['0', 'π'])

            # Col 1: raw per-detector intensity, positive vs negative, this country's M members only
            ax = fig.add_subplot(gs[r, 1])
            bar_w = 0.35
            ax.bar(class_x - bar_w / 2, I_pos_np[r].reshape(-1), width=bar_w, color='tab:red', label='positive')
            ax.bar(class_x + bar_w / 2, I_neg_np[r].reshape(-1), width=bar_w, color='tab:blue', label='negative')
            ax.set_xticks(class_x)
            ax.set_xlabel('class'); ax.set_ylabel(f'Intensity (M={M} members)')
            ax.set_title(f'Country {r} detector signal', fontsize=9)
            ax.legend(fontsize=7)

            # Col 2: this country's own normalized contrast
            ax = fig.add_subplot(gs[r, 2])
            bar_colors = ['tab:blue'] * num_classes
            bar_colors[label_np] = 'tab:orange'
            ax.bar(class_x, country_contrast_np[r], color=bar_colors)
            ax.axhline(0, color='black', linewidth=0.6)
            ax.set_xticks(class_x)
            ax.set_xlabel('class'); ax.set_ylabel('(I+ - I-) / (I+ + I-)')
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(f'Country {r} class scores', fontsize=9)

            # Col 3: this country's I_ccd (summed over its M members) with detector overlay
            ax = fig.add_subplot(gs[r, 3])
            crop = I_ccd_country_np[r, y0:y1, x0:x1]
            # Clip to the 97.5th percentile -- a few saturated hotspots otherwise dominate
            # the colormap and wash out the lower-intensity pattern at the detectors.
            vmax = np.percentile(crop, 97.5)
            im = ax.imshow(crop, cmap='gray', vmin=crop.min(), vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
            for rr in range(rows_pd):
                for cc in range(cols_pd):
                    cy_local = self.model.pd_centers_y[rr] - y0
                    cx_local = self.model.pd_centers_x[cc] - x0
                    color = 'red' if rr < half else 'blue'
                    ax.add_patch(Rectangle(
                        (cx_local - half_px, cy_local - half_px), pd_px, pd_px,
                        fill=False, edgecolor=color, linewidth=1.2
                    ))
            ax.set_title(f'Country {r} I_ccd', fontsize=9)
            ax.axis('off')

        # Final row: aggregated (summed-over-countries) class scores, spans all 4 columns
        ax = fig.add_subplot(gs[C, :])
        bar_colors = ['tab:blue'] * num_classes
        bar_colors[label_np] = 'tab:orange'
        ax.bar(class_x, agg_contrast_np, color=bar_colors)
        marker_color = 'green' if pred_np == label_np else 'black'
        ax.scatter([pred_np], [agg_contrast_np[pred_np]], marker='x', color=marker_color, s=80, zorder=5)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks(class_x)
        ax.set_xlabel('class'); ax.set_ylabel('sum over countries of (I+ - I-) / (I+ + I-)')
        ax.set_title(f'Aggregate class scores  (pred={pred_np}, {"correct" if pred_np==label_np else "wrong"})', fontsize=10)

        fig.tight_layout()
        path = os.path.join(self.config.image_dir, f'classification_{tag}_epoch{self.epoch:03d}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    # ---------------------------------------------------------------------- #

    def save_slm_masks(self):
        '''Grid of the learned SLM phase masks, one row per country: row r shows
        country r's M members (masks r*M .. r*M+M-1). C=1 collapses to the
        original single-row layout.'''
        with torch.no_grad():
            masks = (torch.sigmoid(self.model.slm_phases) * 2 * np.pi).cpu()
        M, C = self.model.M, self.model.C

        fig, axes = plt.subplots(C, M, figsize=(2.2 * M, 2.2 * C), dpi=200, squeeze=False)
        im = None
        m_idx = 0
        for r in range(C):
            for c in range(M):
                ax = axes[r, c]
                im = ax.imshow(masks[m_idx, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
                ax.set_title(f'country {r}  member {c}', fontsize=8)
                ax.axis('off')
                m_idx += 1
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_ticks([0, np.pi, 2*np.pi])
        cbar.set_ticklabels(['0', 'π', '2π'])
        fig.suptitle(f'SLM phase masks  (epoch {self.epoch})')
        path = os.path.join(self.config.image_dir, f'slm_masks_epoch{self.epoch:03d}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def save_layer_masks(self):
        '''Grid of the learned diffractive-layer phase masks (one per layer, shared
        across all T SLM masks -- no country/member structure here). No-op when
        num_layers == 0.'''
        if self.model.num_layers == 0:
            return
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
        cbar.set_ticks([0, np.pi, 2*np.pi])
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

    #Save config to log_dir/config.json for logging
    with open(os.path.join(config.log_dir, 'config.json'), 'w') as f:
        json.dump(config_to_dict(config), f, indent=2, sort_keys=True)

    writer   = SummaryWriter(config.tfboard_dir)
    log_file = os.path.join(config.log_dir, 'LOG.txt')

    run_name = os.path.basename(config.log_dir)
    try:
        wandb.init(project='TimeMultiplexedClassification', name=run_name,
                   config={'M': config.M, 'C':config.C, 'N': config.N, 'num_layers': config.num_layers,
                           'noise_std': config.meas_noise_std,
                           'num_detectors': config.num_photodiodes,
                           'num_classes': config.num_classes,
                           'train_samples': config.train_samples,
                           'pd_row_spacing': config.pd_row_spacing,
                           'pd_col_spacing': config.pd_col_spacing})
        _use_wandb = True
    except Exception as e:
        print(f'[wandb] disabled — {e}')
        _use_wandb = False

    train_loader, val_loader, _ = get_dataloaders(config)
    trainer = TimeMultiplexedClassifierTrainer(config)
    device  = trainer.device

    print('===> Training Start')
    print(f'Run Name: {config.run_name}')

    for epoch in range(trainer.epoch, config.max_epoch):
        trainer.epoch = epoch
        trainer.model.train()
        trainer.is_training = True

        run_loss = run_acc = run_slm_gnorm = run_layer_gnorm = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}')
        for phi_obj, label in pbar:
            phi_obj = phi_obj.to(device)
            label   = label.to(device)
            loss, acc, slm_gn, layer_gn = trainer.train_step(phi_obj, label)
            run_loss += loss; run_acc += acc; run_slm_gnorm += slm_gn; run_layer_gnorm += layer_gn
            pbar.set_postfix({'loss': f'{loss:.3e}', 'acc': f'{acc:.3f}'})

        n_train = len(train_loader)
        run_loss /= n_train; run_acc /= n_train; run_slm_gnorm /= n_train; run_layer_gnorm /= n_train

        if trainer.sched_slm is not None:
            trainer.sched_slm.step()
        if trainer.sched_layers is not None:
            trainer.sched_layers.step()

        writer.add_scalar('loss/train',     run_loss, epoch)
        writer.add_scalar('accuracy/train', run_acc,  epoch)
        writer.add_scalar('slm_gnorm/train', run_slm_gnorm, epoch)
        writer.add_scalar('layer_gnorm/train', run_layer_gnorm, epoch)
        
        if _use_wandb:
            wandb.log({'loss/train': run_loss, 'accuracy/train': run_acc,
                       'slm_gnorm/train': run_slm_gnorm, 'layer_gnorm/train': run_layer_gnorm}, step=epoch)

        if epoch % config.checkpoint_print == 0:
            msg = (f'<epoch:{epoch:3d}> loss_train:{run_loss:.3e}  '
                   f'acc:{run_acc:.3f}\n')
            print_and_save_msg(msg, log_file)

        if epoch % config.checkpoint_save == 0:
            val_loss = val_acc = 0.0

            with torch.no_grad():
                for phi_obj, label in tqdm(val_loader, desc='  Validating'):
                    phi_obj = phi_obj.to(device)
                    label   = label.to(device)
                    loss, acc = trainer.valid_step(phi_obj, label)
                    val_loss += loss; val_acc += acc

            n_val = len(val_loader)
            val_loss /= n_val; val_acc /= n_val

            writer.add_scalar('loss/val',     val_loss, epoch)
            writer.add_scalar('accuracy/val', val_acc,  epoch)
            if _use_wandb:
                wandb.log({'loss/val': val_loss, 'accuracy/val': val_acc}, step=epoch)

            msg = (f'<epoch:{epoch:3d}> loss_val:{val_loss:.3e}  '
                   f'acc:{val_acc:.3f}\n')
            print_and_save_msg(msg, log_file)

            trainer.save()

            #trainer.save_images(phi_obj, label, tag='val') Causing some crashing problems during training, so disabling for now
            trainer.save_slm_masks()
            trainer.save_layer_masks()

            if val_acc > trainer.best_val_acc:
                trainer.best_val_acc = val_acc
                trainer.save_best()

    writer.close()
    if _use_wandb:
        wandb.finish()
    print('===> Training Complete')


if __name__ == '__main__':
    main()
