'''
Time-Multiplexed Classifier — test / evaluation

Usage:
    python test.py
    python test.py --ckpt logs/.../model/epoch=050.pth

Outputs saved to <out_dir>/:
    test_per_class.png        — per-class precision / recall / F1
    test_confusion_matrix.png — row-normalized confusion matrix (true x predicted)
    test_misclassified.png    — grid of misclassified test samples (true vs. pred)
    slm_masks_final.png       — learned SLM phase masks
    layer_masks_final.png     — learned diffractive-layer phase masks (if num_layers > 0)
    mask_similarity.png       — pairwise phase similarity between the M masks per country
Optionally appends a summary row (accuracy, loss, macro-F1) to --csv, if given.
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data as data_utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from config import init_params, recompute_derived
from model import TimeMultiplexedClassifier
from dataloader import MNISTPhaseDataset
from loss import ClassificationLoss


# ─────────────────────────── evaluation loop ──────────────────────────────── #

def _eval_loop(model, loader, criterion, device, desc='Evaluating', max_wrong=32):
    '''
    Single evaluation pass over a dataloader.

    Returns
    -------
    agg : dict of per-sample lists --
        'loss'         — per-batch mean cross-entropy (one value per batch)
        'labels'       — true class per sample
        'preds'        — predicted class per sample (argmax of class_scores)
        'class_scores' — [num_classes] temperature-scaled differential contrast
                         per sample (see ClassificationLoss)
        'wrong_images' — input phase images for up to max_wrong misclassified
                         samples, collected as encountered (cheap -- buffered
                         from this same pass, no second pass over the loader,
                         no I_vec/I_ccd kept)
        'wrong_labels' / 'wrong_preds' — matching true / predicted class
    '''
    agg = defaultdict(list)

    with torch.no_grad():
        for phi_obj, labels in tqdm(loader, desc=desc):
            phi_obj = phi_obj.to(device)
            labels  = labels.to(device)

            I_vec = model(phi_obj)
            loss, agg_class_scores, agg_pred, individual_scores = criterion(I_vec, labels)
            individual_preds = individual_scores.argmax(dim=2) #[B, C]

            agg['loss'].append(loss.item())
            agg['labels'].extend(labels.cpu().tolist())
            agg['preds'].extend(agg_pred.cpu().tolist())
            agg['class_scores'].extend(agg_class_scores.cpu().tolist())
            agg['individual_preds'].extend(individual_preds.cpu().tolist())

            if len(agg['wrong_images']) < max_wrong:
                wrong_idx = (agg_pred != labels).nonzero(as_tuple=True)[0].cpu().tolist()
                for i in wrong_idx:
                    if len(agg['wrong_images']) >= max_wrong:
                        break
                    agg['wrong_images'].append(phi_obj[i, 0].cpu().numpy())
                    agg['wrong_labels'].append(int(labels[i].item()))
                    agg['wrong_preds'].append(int(agg_pred[i].item()))

    return agg


# ──────────────────────────── print summary ───────────────────────────────── #

def _print_summary(agg, tag='Test'):
    labels = np.array(agg['labels'])
    preds  = np.array(agg['preds'])
    correct = int((preds == labels).sum())
    acc     = correct / len(labels)

    print(f'\n── {tag} Results ──────────────────────────────────────────')
    print(f'  Loss (softmax cross-entropy)  : {np.mean(agg["loss"]):.4f}')
    print(f'  Accuracy (aggregated): {acc:.4f}  ({correct}/{len(labels)})')

    individual_preds = np.array(agg['individual_preds'])   # [N, C]
    num_countries = individual_preds.shape[1]
    per_country_acc = (individual_preds == labels[:, np.newaxis]).mean(axis=0)   # [C]
    print(f'  Per-country accuracy (individual prediction vs. true class):')
    for c in range(num_countries):
        print(f'    country {c}: {per_country_acc[c]:.4f}')
    print(f'──────────────────────────────────────────────────────────\n')


# ──────────────────────────── save figures ────────────────────────────────── #

def _per_class_prf(labels, preds, num_classes):
    '''Per-class precision / recall / F1 from TP/FP/FN counts (hand-rolled -- no
    sklearn dependency in this project). Returns three [num_classes] numpy arrays.

    recall is equivalent to per-class accuracy (fraction of that class's true samples
    correctly predicted); precision is how much a class's predictions can be trusted
    (fraction of predictions for that class that were actually correct).
    '''
    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p); recall.append(r); f1.append(f)
    return np.array(precision), np.array(recall), np.array(f1)


def _save_per_class(agg, save_path, config=None, dpi=200):
    '''Bar chart of per-class precision / recall / F1 (see _per_class_prf).'''
    labels = np.array(agg['labels'])
    preds  = np.array(agg['preds'])

    num_classes = int(getattr(config, 'num_classes', labels.max() + 1)) if config is not None \
        else int(labels.max() + 1)
    classes = np.arange(num_classes)
    precision, recall, f1 = _per_class_prf(labels, preds, num_classes)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=dpi)
    for ax, (vals, title) in zip(axes, [
        (precision, 'Precision'),
        (recall,    'Recall (= per-class accuracy)'),
        (f1,        'F1'),
    ]):
        ax.bar(classes, vals, color='steelblue', alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Class')
        ax.set_xticks(classes)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Per-class breakdown saved → {save_path}')


def _save_confusion_matrix(agg, save_path, config=None, dpi=200):
    '''NxN confusion matrix heatmap, row-normalized (each row -- one true class --
    sums to 1, so the color shows what fraction of that class's samples landed in
    each predicted class). Raw counts are annotated in each cell.'''
    labels = np.array(agg['labels'])
    preds  = np.array(agg['preds'])
    acc = int((preds == labels).sum()) / len(labels)


    num_classes = int(getattr(config, 'num_classes', labels.max() + 1)) if config is not None \
        else int(labels.max() + 1)
    classes = np.arange(num_classes)

    cm = np.bincount(labels * num_classes + preds,
                      minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=dpi)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    thresh = cm_norm.max() / 2.0
    for i in classes:
        for j in classes:
            if cm[i, j] == 0:
                continue
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8)

    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_title(f'Confusion matrix (Test Acc = {acc:.4f})', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='fraction of true-class row')

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Confusion matrix saved → {save_path}')


def _save_misclassified(agg, save_path, n_show=16, dpi=200):
    '''Grid of misclassified test samples (from agg['wrong_images'/'wrong_labels'/
    'wrong_preds'], populated by _eval_loop's max_wrong buffer): input image titled
    with true vs. predicted class.'''
    images = agg['wrong_images']
    labels = agg['wrong_labels']
    preds  = agg['wrong_preds']

    n = min(len(images), n_show)
    if n == 0:
        print('No misclassified samples to plot (0 collected -- perfect accuracy, '
              'or max_wrong=0 in _eval_loop).')
        return

    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.3 * rows), dpi=dpi)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'true={labels[i]}  pred={preds[i]}', fontsize=8, color='crimson')
        ax.axis('off')

    fig.suptitle(f'Misclassified examples ({n} of {len(images)} collected)', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Misclassified examples saved → {save_path}')


def _save_mask_similarity(model, config, out_dir, dpi=200):
    '''
    Pairwise circular-phase similarity between the M SLM masks within each country,
    on the model's raw learned phase (post-sigmoid, at its native slm_x_num
    resolution -- the actual learnable parameter, not the upsampled/embedded
    sim-grid version).

    similarity(i, j) = | mean_pixels( exp(i * (phi_i - phi_j)) ) |, in [0, 1].

    This is a circular (phase-aware) similarity, not a naive Euclidean/cosine one --
    raw phase values wrap at 2pi, so comparing them directly would treat e.g. 0.01
    and 2*pi-0.01 as maximally different when they're actually almost identical.
    It's also deliberately invariant to a constant phase offset between two masks:
    a uniform additive shift to an entire mask doesn't change its own detected
    intensity (it cancels under |field|^2), so two masks differing only by such a
    constant really do produce redundant measurements and should score as maximally
    similar (1.0), not different.

    1.0 = identical up to a constant offset (fully redundant); 0.0 = pixel-wise
    phase differences are spread uniformly around the circle (no consistent
    relationship between the two masks).

    Saves one M x M heatmap per country to out_dir/mask_similarity.png.
    Returns {country_idx: [M, M] numpy array}.
    '''
    M, C = model.M, model.C
    with torch.no_grad():
        phi = (torch.sigmoid(model.slm_phases) * 2 * np.pi)[:, 0]   # [T, slm_x_num, slm_x_num]
    phi = phi.cpu().numpy().reshape(M * C, -1)                       # [T, num_pixels]

    if M < 2:
        print('[mask_similarity] M < 2 -- nothing to compare, skipping.')
        return {}

    sims = {}
    fig, axes = plt.subplots(1, C, figsize=(4 * C, 3.6), dpi=dpi, squeeze=False)
    axes = axes[0]
    im = None

    for c in range(C):
        masks = phi[c * M:(c + 1) * M]          # [M, num_pixels]
        sim = np.eye(M)
        for i in range(M):
            for j in range(i + 1, M):
                s = np.abs(np.mean(np.exp(1j * (masks[i] - masks[j]))))
                sim[i, j] = sim[j, i] = s
        sims[c] = sim

        ax = axes[c]
        im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'country {c}', fontsize=9)
        ax.set_xlabel('member'); ax.set_ylabel('member')
        ax.set_xticks(np.arange(M)); ax.set_yticks(np.arange(M))

    fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02, label='similarity')
    off_diag = np.concatenate([sims[c][~np.eye(M, dtype=bool)] for c in range(C)])
    mean_sim = float(off_diag.mean())
    fig.suptitle(f'Pairwise mask phase similarity  (M={M}, C={C})  |  '
                 f'mean off-diagonal similarity = {mean_sim:.3f}', fontsize=10)

    path = os.path.join(out_dir, 'mask_similarity.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Mask similarity saved → {path}  (mean off-diagonal = {mean_sim:.3f})')
    return sims


def _save_slm_masks(model, config, out_dir, dpi=200):
    with torch.no_grad():
        masks = (torch.sigmoid(model.slm_phases) * 2 * np.pi).cpu()
    # Layout masks by country: each row is a country, each column a mask (member)
    T = masks.shape[0]
    M = getattr(model, 'M', getattr(config, 'M', 1))
    C = getattr(model, 'C', getattr(config, 'C', 1))
    # compute columns per row; handle edge cases where T != M*C
    cols = int(np.ceil(T / C))
    rows = C
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    m_idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if m_idx < T:
                im = ax.imshow(masks[m_idx, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
                ax.set_title(f'country {r}  member {c}')
                ax.axis('off')
            else:
                ax.axis('off')
            m_idx += 1
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_ticks([0, np.pi, 2*np.pi])
    cbar.set_ticklabels(['0', 'π', '2π'])
    fig.suptitle('Optimized SLM phase masks')
    save_path = os.path.join(out_dir, 'slm_masks_final.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'SLM masks saved → {save_path}')

def save_layer_masks(model, config, out_dir, dpi=200):
    '''Grid of the learned diffractive-layer phase masks (one per layer, shared
    across all T SLM masks -- no country/member structure here). No-op when
    num_layers == 0.'''
    if model.num_layers == 0:
        return
    with torch.no_grad():
        masks = (torch.sigmoid(model.layer_phases) * 2 * np.pi).cpu()
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
    fig.suptitle('Diffractive layer phase masks')
    path = os.path.join(out_dir, 'layer_masks_final.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Layer masks saved → {path}')


# ──────────────────────────── main entry point ────────────────────────────── #

def _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, config=None):
    '''Append one results row to the sweep CSV, creating the file if needed.'''
    labels = np.array(agg['labels'])
    preds  = np.array(agg['preds'])
    acc    = float((preds == labels).mean())

    num_classes = int(getattr(config, 'num_classes', labels.max() + 1)) if config is not None \
        else int(labels.max() + 1)
    _, _, f1 = _per_class_prf(labels, preds, num_classes)

    row = {
        'sweep'     : sweep_name or '',
        'label'     : run_label  or '',
        'ckpt'      : ckpt_path  or '',
        'n_samples' : len(labels),
        'accuracy'  : round(acc, 4),
        'loss_mean' : round(float(np.mean(agg['loss'])), 4),
        'loss_std'  : round(float(np.std(agg['loss'])),  4),
        'macro_f1'  : round(float(f1.mean()), 4),
    }
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'Results appended → {csv_path}')


def _build_test_loader(config, dataset):
    '''Return a DataLoader for the requested dataset.'''
    if dataset == 'fashion':
        fashion_path = getattr(config, 'fashion_data_path', None)
        if not fashion_path:
            raise ValueError('fashion_data_path is not set in config.')
        orig_path        = config.data_path
        config.data_path = fashion_path
        ds               = MNISTPhaseDataset(config, is_training=False)
        config.data_path = orig_path
    elif dataset == 'grating':
        from dataloader import MNISTGratingPhaseDataset
        config.dataset      = 'mnist_grating'
        config.grating_only = True
        ds = MNISTGratingPhaseDataset(config, split='test')
    elif dataset == 'mnist_grating':
        from dataloader import MNISTGratingPhaseDataset
        config.dataset = 'mnist_grating'
        ds = MNISTGratingPhaseDataset(config, split='test')
    elif dataset == 'cifar10':
        from dataloader import CIFAR10PhaseDataset
        ds = CIFAR10PhaseDataset(config, is_training=False)
    elif dataset == 'tinyimagenet':
        from dataloader import TinyImageNetPhaseDataset
        ds = TinyImageNetPhaseDataset(config, is_training=False)
    else:
        ds = MNISTPhaseDataset(config, is_training=False)

    return data_utils.DataLoader(
        ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
    )


def evaluate(ckpt_path=None, out_dir=None, csv_path=None, sweep_name=None,
             run_label=None, dataset=None):
    config = init_params()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint (config first, then model) ────────────────────────
    # Preserve local paths — they are machine-specific and must not be
    # overwritten by values that were saved inside the checkpoint.
    _path_keys = ('data_path', 'grating_data_path', 'fashion_data_path',
                  'cifar10_data_path', 'model_dir', 'log_dir')
    saved_paths = {k: getattr(config, k, None) for k in _path_keys}

    ckpt_path = ckpt_path or config.ckpt_to_load
    ckpt = None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'config' in ckpt:
            config.__dict__.update(ckpt['config'])
            for k, v in saved_paths.items():
                if v is not None:
                    setattr(config, k, v)
            recompute_derived(config)

        # If the checkpoint contains a saved model state with learned SLM masks,
        # ensure `config.T` matches the saved `slm_phases` first-dimension so
        # the constructed model has the correct shape when we call
        # `model.load_state_dict(...)`. This guards older checkpoints that may
        # not have had `config.T` saved or where `config.T` is inconsistent.
        if 'model' in ckpt:
            state = ckpt['model']
            for k, v in state.items():
                if 'slm_phases' in k:
                    try:
                        T_saved = int(v.shape[0])
                    except Exception:
                        T_saved = None
                    if T_saved is not None:
                        if getattr(config, 'T', None) != T_saved:
                            print(f'[checkpoint] overriding config.T -> {T_saved} (from {k})')
                            config.T = T_saved
                            # If C is present, derive M from T and C when possible
                            try:
                                C = int(getattr(config, 'C', 1))
                                if C > 0:
                                    config.M = int(config.T // C)
                            except Exception:
                                pass
                    break

    if dataset is None:
        _DATASET_ALIASES = {'FashionMNIST': 'fashion'}
        dataset = _DATASET_ALIASES.get(config.dataset, config.dataset)

    model = TimeMultiplexedClassifier(config).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'], strict=False)
        print(f'Loaded: {ckpt_path}  (epoch {ckpt["epoch"]})')
    else:
        print('[WARNING] No checkpoint provided — evaluating random init.')
    model.eval()

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(ckpt_path) if ckpt_path else '.', '..', 'test', dataset
        )
    os.makedirs(out_dir, exist_ok=True)

    # ── Test loader ───────────────────────────────────────────────────────
    print(f'Dataset: {dataset}')
    test_loader = _build_test_loader(config, dataset)
    criterion   = ClassificationLoss(config).to(device)

    agg = _eval_loop(model, test_loader, criterion, device, desc='Testing')
    _print_summary(agg)

    if csv_path is not None:
        _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, config=config)

    _save_per_class(agg, os.path.join(out_dir, 'test_per_class.png'), config=config)
    _save_confusion_matrix(agg, os.path.join(out_dir, 'test_confusion_matrix.png'), config=config)
    _save_misclassified(agg, os.path.join(out_dir, 'test_misclassified.png'))

    # ── SLM masks / Layers / mask similarity ────────────────────────────────
    _save_slm_masks(model, config, out_dir)
    save_layer_masks(model, config, out_dir)
    _save_mask_similarity(model, config, out_dir)

    print(f'\nAll outputs saved to: {os.path.abspath(out_dir)}')


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    # ── Hardcoded fallbacks (used when running test.py directly, no CLI args) ─
    import paths as _paths
    CKPT_PATH = os.path.join(_paths.LOG_DIR, r"20260706-1344-ConvDecoder-M20-N28-simdx4um-slm500px_slmdx8um-objdx8um\model\best.pth")  # <-- EDIT THIS
    DATASET   = None   # None = whatever the checkpoint was trained on; or force
                        # 'mnist' | 'fashion' | 'mnist_grating' | 'grating' | 'cifar10' | 'tinyimagenet'
    # ─────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    default=None, help='Path to checkpoint')
    parser.add_argument('--out_dir', default=None, help='Output directory for plots')
    parser.add_argument('--csv',     default=None, help='CSV file to append results to')
    parser.add_argument('--sweep',   default=None, help='Sweep name (written to CSV)')
    parser.add_argument('--label',   default=None, help='Run label (written to CSV)')
    parser.add_argument('--dataset', default=None,
                        choices=['mnist', 'mnist_grating', 'grating', 'fashion', 'cifar10', 'tinyimagenet'],
                        help='Dataset to evaluate on')
    args = parser.parse_args()

    evaluate(
        ckpt_path  = args.ckpt or CKPT_PATH,
        out_dir    = args.out_dir,
        csv_path   = args.csv,
        sweep_name = args.sweep,
        run_label  = args.label,
        dataset    = args.dataset or DATASET,
    )
