'''
Time Multiplexed Classifier — dataloader

MNIST Fashion objects are treated as phase objects:
  pixel value ∈ [0, 1]  →  phase  = pixel × input_phase_max  ∈ [0, input_phase_max]

The dataloader returns a single tensor per sample:
  phi_obj : [1, N, N]  (phase in radians, on CPU; .to(device) happens in the train loop)
'''

import os
import random
from pathlib import Path
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms


class MNISTPhaseDataset(data.Dataset):
    '''
    Loads raw MNIST binary files and converts images to phase objects.
    Reads the same raw idx-format files used in the original project.
    '''

    def __init__(self, config, is_training=True):
        super(MNISTPhaseDataset, self).__init__()
        self.config        = config
        self.is_training   = is_training
        self.image_transform = config.image_transform   # list or None

        image_data, label_data = self._load_raw(config.data_path, is_training)

        if is_training and config.train_samples is not None:
            image_data = image_data[:config.train_samples]
            label_data = label_data[:config.train_samples]

        # Keep on CPU; GPU transfer happens per-batch in the train loop
        self.image_data = image_data   # [N_samples, H, W]  uint8 tensor
        self.label_data = label_data   # [N_samples]        int64 tensor

    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_raw(data_path, is_training):
        '''Read raw MNIST idx files (same format as original dataloader).'''
        split   = 'train' if is_training else 't10k'
        img_file = os.path.join(data_path, f'{split}-images-idx3-ubyte')
        lbl_file = os.path.join(data_path, f'{split}-labels-idx1-ubyte')

        with open(img_file, 'rb') as f:
            f.read(16)                              # skip magic + dims header
            data_np = np.frombuffer(f.read(), dtype=np.uint8)
            imgs = torch.from_numpy(data_np.reshape(-1, 28, 28).copy())

        with open(lbl_file, 'rb') as f:
            f.read(8)                               # skip magic + count header
            data_np = np.frombuffer(f.read(), dtype=np.uint8)
            labels  = torch.from_numpy(data_np.astype(np.int64).copy())

        return imgs, labels

    # ---------------------------------------------------------------------- #

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        img    = self.image_data[index].float() / 255.0          # [H, W]  in [0,1]
        img    = img.unsqueeze(0)                                  # [1, H, W]
        label  = self.label_data[index]

        # ── Resize from native 28×28 to data_x_num × data_x_num ──────────
        # Raw MNIST is always 28×28.  If data_x_num differs, resample here
        # so that the dataloader output matches the decoder's expected size.
        _MNIST_NATIVE = 28
        target = int(self.config.data_x_num)
        if target != _MNIST_NATIVE:
            img = F.interpolate(
                img.unsqueeze(0),                          # → [1, 1, 28, 28]
                size=(target, target),
                mode='bilinear',
                align_corners=False
            )[0]                                           # → [1, target, target]

        # ── Optional augmentations ────────────────────────────────────────
        if self.is_training and self.image_transform is not None:
            if 'fliplr' in self.image_transform and random.randint(0, 1):
                img = torch.fliplr(img)

            if 'flipud' in self.image_transform and random.randint(0, 1):
                img = torch.flipud(img)

            if 'rotate90' in self.image_transform:
                img = torch.rot90(img, random.randint(0, 3), [1, 2])

            if 'transform' in self.image_transform:
                angle, trans, scale, shear = transforms.RandomAffine.get_params(
                    degrees=(-30, 30),
                    translate=(0.1, 0.1),
                    scale_ranges=(0.8, 1.2),
                    shears=None,
                    img_size=(target, target)
                )
                img = transforms.functional.affine(
                    img, angle=angle, translate=trans,
                    scale=scale, shear=shear
                )

        # ── Convert intensity → phase ─────────────────────────────────────
        phi = img * self.config.input_phase_max     # [1, data_x_num, data_x_num]

        return phi, label


class MNISTGratingPhaseDataset(data.Dataset):
    '''
    Combined MNIST + grating phase dataset.

    MNIST uses the standard train/test binary files; gratings are loaded from
    a single .npy file and split 80 / 10 / 10 (train / val / test).
    Grating samples are assigned label -1.

    split : 'train' | 'val' | 'test'
    '''

    _GRATING_TRAIN_FRAC = 0.8
    _GRATING_VAL_FRAC   = 0.1

    def __init__(self, config, split='train'):
        super().__init__()
        assert split in ('train', 'val', 'test'), f'Unknown split: {split}'
        self.config          = config
        self.split           = split
        self.is_training     = (split == 'train')
        self.image_transform = config.image_transform

        imgs_list, labels_list = [], []

        # Per-source caps for training. Priority:
        #   mnist_cap   → explicit MNIST limit (overrides train_samples split)
        #   train_samples → fallback equal split between MNIST and gratings
        cap = config.train_samples
        explicit_mnist_cap = getattr(config, 'mnist_cap', None)
        if split == 'train':
            if config.grating_only:
                mnist_cap   = 0
                grating_cap = cap
            else:
                mnist_cap   = explicit_mnist_cap if explicit_mnist_cap is not None \
                              else (cap // 2 if cap is not None else None)
                grating_cap = cap // 2 if cap is not None else None
        else:
            mnist_cap = grating_cap = None

        # ── MNIST ──────────────────────────────────────────────────────────
        if not config.grating_only:
            mnist_imgs, mnist_labels = MNISTPhaseDataset._load_raw(
                config.data_path, is_training=(split != 'test')
            )
            if split in ('train', 'val'):
                n_total = len(mnist_imgs)
                n_val   = int(n_total * config.validation_ratio)
                gen     = torch.Generator().manual_seed(config.seed)
                perm    = torch.randperm(n_total, generator=gen)
                idx     = perm[n_val:] if split == 'train' else perm[:n_val]
                mnist_imgs, mnist_labels = mnist_imgs[idx], mnist_labels[idx]
            if mnist_cap is not None:
                mnist_imgs, mnist_labels = mnist_imgs[:mnist_cap], mnist_labels[:mnist_cap]
            imgs_list.append(mnist_imgs)
            labels_list.append(mnist_labels)

        # ── Gratings ───────────────────────────────────────────────────────
        # If grating_data_path is a directory, load the pre-split file for
        # this split directly (prevents data leakage across splits).
        # Otherwise fall back to loading a single file and splitting in-code.
        g_path = Path(config.grating_data_path)
        if g_path.is_dir():
            split_file = {'train': 'data_train.npy', 'val': 'data_valid.npy', 'test': 'data_test.npy'}[split]
            grating_np = np.load(g_path / split_file, allow_pickle=True)
            grating_t  = torch.from_numpy(grating_np.copy()).float()
            if grating_t.max() <= 1.0:
                grating_t = grating_t * 255.0
            grating_t = grating_t.to(torch.uint8)
            if grating_t.ndim == 4:
                grating_t = grating_t.squeeze(1)
        else:
            grating_np = np.load(g_path, allow_pickle=True)
            grating_t  = torch.from_numpy(grating_np.copy()).float()
            if grating_t.max() <= 1.0:
                grating_t = grating_t * 255.0
            grating_t = grating_t.to(torch.uint8)
            if grating_t.ndim == 4:
                grating_t = grating_t.squeeze(1)

            n_g       = len(grating_t)
            n_g_val   = int(n_g * self._GRATING_VAL_FRAC)
            n_g_test  = int(n_g * (1.0 - self._GRATING_TRAIN_FRAC - self._GRATING_VAL_FRAC))
            n_g_train = n_g - n_g_val - n_g_test
            gen_g     = torch.Generator().manual_seed(config.seed)
            g_perm    = torch.randperm(n_g, generator=gen_g)
            if split == 'train':
                g_perm = g_perm[:n_g_train]
            elif split == 'val':
                g_perm = g_perm[n_g_train:n_g_train + n_g_val]
            else:
                g_perm = g_perm[n_g_train + n_g_val:]
            grating_t = grating_t[g_perm]

        if grating_cap is not None:
            grating_t = grating_t[:grating_cap]

        # Load per-sample period labels if available (saved by generator).
        # Period values (pixels, float): >=1 for gratings, -2 = smooth shape, -3 = blob.
        # Fall back to -1 for datasets generated without period metadata.
        period_file = None
        if g_path.is_dir():
            pf = {'train': 'periods_train.npy', 'val': 'periods_valid.npy', 'test': 'periods_test.npy'}[split]
            candidate = g_path / pf
            if candidate.exists():
                period_file = candidate

        if period_file is not None:
            periods_np = np.load(period_file)
            if grating_cap is not None:
                periods_np = periods_np[:grating_cap]
            grating_labels = torch.from_numpy(periods_np.round().astype(np.int64))
        else:
            grating_labels = torch.full((len(grating_t),), -1, dtype=torch.long)

        imgs_list.append(grating_t)
        labels_list.append(grating_labels)

        self.image_data = torch.cat(imgs_list, dim=0)
        self.label_data = torch.cat(labels_list, dim=0)

        n_mnist   = 0 if config.grating_only else len(mnist_imgs)
        n_grating = len(grating_t)
        # MNIST is concatenated before gratings, so index < n_mnist identifies the source.
        # Labels cannot do this: grating labels are period values (>=1) that overlap MNIST 0-9.
        self.n_mnist = n_mnist
        self.n_grating = n_grating
        print(f'MNISTGratingPhaseDataset [{split}]: '
              f'{n_mnist} MNIST + {n_grating} grating = {len(self.image_data)} total')

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        img   = self.image_data[index].float() / 255.0   # [H, W]
        label = self.label_data[index]
        img   = img.unsqueeze(0)                          # [1, H, W]

        target = int(self.config.data_x_num)
        if img.shape[-1] != target or img.shape[-2] != target:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(target, target),
                mode='bilinear',
                align_corners=False,
            )[0]

        if self.is_training and self.image_transform is not None:
            if 'fliplr' in self.image_transform and random.randint(0, 1):
                img = torch.fliplr(img)
            if 'flipud' in self.image_transform and random.randint(0, 1):
                img = torch.flipud(img)
            if 'rotate90' in self.image_transform:
                img = torch.rot90(img, random.randint(0, 3), [1, 2])
            if 'transform' in self.image_transform:
                angle, trans, scale, shear = transforms.RandomAffine.get_params(
                    degrees=(-30, 30),
                    translate=(0.1, 0.1),
                    scale_ranges=(0.8, 1.2),
                    shears=None,
                    img_size=(target, target),
                )
                img = transforms.functional.affine(
                    img, angle=angle, translate=trans, scale=scale, shear=shear,
                )

        phi = img * self.config.input_phase_max
        return phi, label


class CIFAR10PhaseDataset(data.Dataset):
    '''
    CIFAR-10 converted to grayscale phase objects.

    RGB images are averaged to grayscale, normalised to [0, 1], then scaled
    by input_phase_max to produce phase maps in [0, input_phase_max].
    Images are resized to data_x_num × data_x_num if needed.
    '''

    def __init__(self, config, is_training=True):
        super().__init__()
        self.config          = config
        self.is_training     = is_training
        self.image_transform = config.image_transform

        ds = torchvision.datasets.CIFAR10(
            root=config.cifar10_data_path, train=is_training, download=False
        )
        # imgs: numpy [N, 32, 32, 3] uint8  →  grayscale tensor [N, 32, 32] uint8
        imgs_np = ds.data                                          # [N, 32, 32, 3]
        gray_np = imgs_np.mean(axis=-1).astype(np.uint8)          # [N, 32, 32]
        self.image_data = torch.from_numpy(gray_np)               # uint8
        self.label_data = torch.tensor(ds.targets, dtype=torch.long)

        if is_training and config.train_samples is not None:
            self.image_data = self.image_data[:config.train_samples]
            self.label_data = self.label_data[:config.train_samples]

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        img   = self.image_data[index].float() / 255.0   # [32, 32]  in [0, 1]
        img   = img.unsqueeze(0)                           # [1, 32, 32]
        label = self.label_data[index]

        target = int(self.config.data_x_num)
        if img.shape[-1] != target or img.shape[-2] != target:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(target, target),
                mode='bilinear',
                align_corners=False,
            )[0]

        if self.is_training and self.image_transform is not None:
            if 'fliplr' in self.image_transform and random.randint(0, 1):
                img = torch.fliplr(img)
            if 'flipud' in self.image_transform and random.randint(0, 1):
                img = torch.flipud(img)
            if 'rotate90' in self.image_transform:
                img = torch.rot90(img, random.randint(0, 3), [1, 2])
            if 'transform' in self.image_transform:
                angle, trans, scale, shear = transforms.RandomAffine.get_params(
                    degrees=(-30, 30),
                    translate=(0.1, 0.1),
                    scale_ranges=(0.8, 1.2),
                    shears=None,
                    img_size=(target, target),
                )
                img = transforms.functional.affine(
                    img, angle=angle, translate=trans, scale=scale, shear=shear,
                )

        phi = img * self.config.input_phase_max
        return phi, label


class ArrayPhaseDataset(data.Dataset):
    '''
    Generic grayscale image array -> phase objects, for datasets loaded as numpy arrays
    (EMNIST / BloodMNIST / OrganAMNIST / ...). Mirrors CIFAR10PhaseDataset's conversion:
    image in [0,1] -> resize to data_x_num -> * input_phase_max.

    images : numpy [N, H, W] grayscale, uint8 [0,255] or float [0,1].
    labels : numpy [N] (optional; defaults to zeros).
    '''

    def __init__(self, config, images, labels=None, is_training=False):
        super().__init__()
        self.config      = config
        self.is_training = is_training
        imgs = np.asarray(images)
        if imgs.dtype != np.uint8:
            imgs = (imgs * 255.0).clip(0, 255).astype(np.uint8) if imgs.max() <= 1.0 else imgs.astype(np.uint8)
        self.image_data = torch.from_numpy(imgs)                                   # uint8 [N,H,W]
        self.label_data = (torch.zeros(len(imgs), dtype=torch.long) if labels is None
                           else torch.as_tensor(np.asarray(labels).ravel().astype('int64')))

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        img = self.image_data[index].float() / 255.0        # [H, W]
        img = img.unsqueeze(0)                                # [1, H, W]
        target = int(self.config.data_x_num)
        if img.shape[-1] != target or img.shape[-2] != target:
            img = F.interpolate(img.unsqueeze(0), size=(target, target),
                                mode='bilinear', align_corners=False)[0]
        phi = img * self.config.input_phase_max
        return phi, self.label_data[index]


class TinyImageNetPhaseDataset(data.Dataset):
    '''
    Tiny ImageNet converted to grayscale phase objects.

    Expected directory structure (standard TinyImageNet download):
        <root>/train/<class_id>/images/*.JPEG
        <root>/val/images/*.JPEG
        <root>/val/val_annotations.txt   (tab-sep: filename, class_id, ...)

    Images are converted to grayscale, resized to config.data_x_num, and
    scaled to [0, input_phase_max].
    '''

    def __init__(self, config, is_training=True):
        super().__init__()
        self.config      = config
        self.is_training = is_training
        root = config.tinyimagenet_data_path

        train_dir    = os.path.join(root, 'train')
        class_names  = sorted(os.listdir(train_dir))
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        if is_training:
            ds = torchvision.datasets.ImageFolder(train_dir)
            self.image_paths = [s[0] for s in ds.samples]
            self.labels      = torch.tensor([s[1] for s in ds.samples], dtype=torch.long)
        else:
            val_img_dir = os.path.join(root, 'val', 'images')
            ann_file    = os.path.join(root, 'val', 'val_annotations.txt')
            paths, lbls = [], []
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    paths.append(os.path.join(val_img_dir, parts[0]))
                    lbls.append(class_to_idx.get(parts[1], 0))
            self.image_paths = paths
            self.labels      = torch.tensor(lbls, dtype=torch.long)

        if is_training and getattr(config, 'train_samples', None) is not None:
            _n = int(config.train_samples)
            self.image_paths = self.image_paths[:_n]
            self.labels      = self.labels[:_n]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        from PIL import Image as _PIL
        img = _PIL.open(self.image_paths[index]).convert('L')
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.unsqueeze(0)

        target = int(self.config.data_x_num)
        if img.shape[-1] != target or img.shape[-2] != target:
            img = F.interpolate(img.unsqueeze(0), size=(target, target),
                                mode='bilinear', align_corners=False)[0]

        phi = img * self.config.input_phase_max
        return phi, self.labels[index]


# --------------------------------------------------------------------------- #

def _split_train_val(config, build_train_full):
    '''
    Shared train/val split for the get_dataloaders() branches below.

    build_train_full : zero-arg callable that constructs and returns the training-split
        dataset object, reading config.train_samples (and config.data_path, etc.) as they
        currently stand. May be called with config.train_samples temporarily widened --
        see the val_samples path below.

    config.val_samples is None (default) -- ORIGINAL behavior: build the pool once, capped
        to config.train_samples as normal; val = validation_ratio fraction of that pool,
        carved OUT of it (so a small train_samples means a small, noisy val set).

    config.val_samples is an int -- DECOUPLED behavior: draw val_samples samples from the
        pool *beyond* train_samples instead. config.train_samples is temporarily widened to
        train_samples + val_samples so build_train_full() loads enough data, then restored
        before returning, so callers (run_name, wandb config, checkpoints, ...) still see
        the original train_samples value.
    '''
    val_samples = getattr(config, 'val_samples', None)

    if val_samples is not None:
        if config.train_samples is None:
            raise ValueError('config.val_samples requires config.train_samples to be set -- '
                              'decoupled validation draws extra samples beyond the training subset.')
        train_size = config.train_samples
        val_size   = int(val_samples)
        orig_cap = config.train_samples
        config.train_samples = train_size + val_size   # widen the pool for this load only
        train_full = build_train_full()
        config.train_samples = orig_cap                 # restore
    else:
        train_full = build_train_full()
        val_size   = int(len(train_full) * config.validation_ratio)
        train_size = len(train_full) - val_size

    return data.random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )


def get_dataloaders(config):
    '''
    Returns (train_loader, val_loader, test_loader).

    config.dataset == 'mnist'          : standard MNIST phase dataset
    config.dataset == 'mnist_grating'  : MNIST + grating combined dataset
    config.dataset == 'FashionMNIST'   : Fashion-MNIST phase dataset
    config.dataset == 'cifar10'        : CIFAR-10 grayscale phase dataset
    config.dataset == 'tinyimagenet'   : Tiny ImageNet grayscale phase dataset
    '''
    if config.dataset == 'mnist_grating':
        train_ds = MNISTGratingPhaseDataset(config, split='train')
        val_ds   = MNISTGratingPhaseDataset(config, split='val')
        test_ds  = MNISTGratingPhaseDataset(config, split='test')
    elif config.dataset == 'FashionMNIST':
        # Same idx-file format as MNIST; just point at the Fashion-MNIST raw dir.
        orig_path        = config.data_path
        config.data_path = config.fashion_data_path
        train_ds, val_ds = _split_train_val(
            config, lambda: MNISTPhaseDataset(config, is_training=True))
        test_ds = MNISTPhaseDataset(config, is_training=False)
        config.data_path = orig_path
    elif config.dataset == 'cifar10':
        train_ds, val_ds = _split_train_val(
            config, lambda: CIFAR10PhaseDataset(config, is_training=True))
        test_ds = CIFAR10PhaseDataset(config, is_training=False)
    elif config.dataset == 'tinyimagenet':
        train_ds, val_ds = _split_train_val(
            config, lambda: TinyImageNetPhaseDataset(config, is_training=True))
        test_ds = TinyImageNetPhaseDataset(config, is_training=False)
    else:
        train_ds, val_ds = _split_train_val(
            config, lambda: MNISTPhaseDataset(config, is_training=True))
        test_ds = MNISTPhaseDataset(config, is_training=False)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
