# time-multiplexed-classifier

A phase object is encoded optically by a bank of **M = 20 learned SLM phase
masks applied in time-multiplexed sequence**. Each object + phase mask pair propagates
through successive diffractive phase layers before reaching a differential detector array plane.
Each detector sums the signal acquired across the M phase masks, predicting the object class 
through the max differential signal of a pair of photodiodes corresponding to the object's class.
```
            phase object + SLM phase bias (M=20 learned masks, time-multiplexed)
                                    |
                    free-space propagation -- K learned diffractive phase
                                    layers in series
                                    |
                                    v
                    20 Photodiode Regions to Perform Differential Detection on 10 Classes
                                    |
                                    v
            Detected Class = max{I+_1 - I-_1 / (I+_1 + I-_1), .. ,{I+_10 - I-_10 / (I+_10 + I-_10)}
```

## Optical and training parameters (`code/config.py`)

| quantity | value | note |
|---|---|---|
| wavelength | 635 nm | |
| simulation grid | `sim_dx=4 um`, `N_sim=2000` | 8.0 x 8.0 mm, angular-spectrum grid |
| SLM | `slm_dx=8 um`, `slm_x_num=500`, `M=20` | 4.0 x 4.0 mm active; time-multiplexed, `sigmoid(slm_phases)*2pi` |
| object | `data_x_num=100`, `obj_dx=8 um` | 800 x 800 um phase object, `phi = image * input_phase_max` |
| phase range | `input_phase_max=pi` | objects span [0, pi] |
| diffractive layers | `num_layers=0` (off by default) | K learned phase plates between SLM and detector; each `layer_dx=8 um`, `layer_size=500` |
| propagation | `z_slm_ccd=5.2 cm` (K=0) | with layers: `slm_first_layer_spacing=5 cm`, `interlayer_spacing=15 um`, `last_layer_ccd_spacing=5 cm` |
| detector array | `pd_num_rows=4`, `pd_num_cols=5`, `photodiode_size=0.1 mm` | 20 detectors, `pd_row/col_spacing=0.2 mm`; rows 0-1 positive, rows 2-3 negative |
| measurement noise | `meas_noise_std=0` | additive, injected during training only |
| classes | `num_classes=10`, `T=0.1` | softmax temperature divides the [-1,1] differential contrast |
| encoder | 20 x 500 x 500 SLM phases | 5.0 M parameters at K=0; + `num_layers * layer_size^2` if diffractive layers are enabled |
| optimisers | Adam; `lr_slm=1e-2`, `lr_layer=1e-3` | cosine annealing, independent optimizer per parameter group |
| loss | softmax cross-entropy over differential contrast | `ClassificationLoss`; no reconstruction loss |

## Layout

```
code/
    config.py       every parameter, with CLI overrides via --set KEY=VALUE
    paths.py        filesystem resolution (environment-overridable)
    wave_prop.py    band-limited angular-spectrum propagator
    model.py        TimeMultiplexedClassifier: SLM -> diffractive layers -> 4x5 photodiode array
    loss.py         ClassificationLoss: differential contrast + softmax cross-entropy
    dataloader.py   image datasets mapped to phase objects
    train.py        end-to-end training (SLM + diffractive layers), figures, TensorBoard/wandb logging
    test.py         evaluation: accuracy, confusion matrix, per-class P/R/F1, misclassified examples
    finetune.py     not applicable to current project.
```

The pipeline files sit next to the modules they import, so `train.py` finds `config`,
`model`, `loss`, `dataloader` and `wave_prop` as plain siblings.

## Setup

```bash
pip install -r requirements.txt
python code/paths.py            # print every resolved location and whether it exists
```

Set these before anything else; the defaults put everything inside the repo.

| variable | holds | default |
|---|---|---|
| `SPQPI_DATA_ROOT` | object datasets | `<repo>/data` |
| `SPQPI_CKPT_ROOT` | trained checkpoints | `<repo>/checkpoints` |
| `SPQPI_PROJECT_ROOT` | parent of the two above | `<repo>` |

The `SPQPI_` prefix is inherited from the project this code came from and is kept
deliberately, so one set of variables serves both repos on the same machine.

## Datasets

`config.py` expects them under `SPQPI_DATA_ROOT`:

| dataset | path | fetch |
|---|---|---|
| FashionMNIST (default) | `FashionMNIST/FashionMNIST/raw/` | `torchvision.datasets.FashionMNIST(root=<DATA_ROOT>/FashionMNIST, download=True)` |
| MNIST | `MNIST/raw/` | `torchvision.datasets.MNIST(root=..., download=True)` |
| EMNIST | `EMNIST/` | torchvision |
| CIFAR-10 | `CIFAR10/` | torchvision |
| Grating | `Grating/` | generated; the generator is not in this repo |

Images are normalised to [0, 1], resized to the 100 x 100 object region, and mapped
linearly to phase as `phi = image * input_phase_max`. The label is the integer
FashionMNIST class (0-9), used directly as the classification target.

## Training

```bash
cd code

# FashionMNIST (default dataset), no diffractive layers
python train.py

# with 2 learned diffractive layers between the SLM and the detector array
python train.py --set num_layers=2

# short smoke run
python train.py --set max_epoch=1 mnist_cap=200 batch_size=5
```

`--set KEY=VALUE` overrides any attribute in `config.py`, preserving its type.

**Where output actually lands: `code/logs/<run-name>/`**, holding `model/`, `images/`,
`tfboard/` and `LOG.txt`. Note this is *not* `paths.LOG_DIR`. `config.py` sets
`log_dir` from `paths.LOG_DIR` (`<repo>/logs`), but `train.py`'s `main()` rebuilds it
after applying `--set` overrides so the run name reflects them, and does so relative to
its own file:

```python
conv_dir = os.path.dirname(os.path.abspath(__file__))   # .../code
config.log_dir = os.path.join(conv_dir, 'logs', run_name)
```

So `paths.LOG_DIR` is effectively dead in that path. Both are covered by the `logs/`
entry in `.gitignore`, so nothing is at risk of being committed either way. Pass
`--log_dir` explicitly to put output somewhere else.

`get_dataloaders` dispatches on `config.dataset`: `FashionMNIST` (default),
`mnist_grating`, `cifar10`, `tinyimagenet`, or anything else (falls through to plain
`MNISTPhaseDataset`).

Training logs both to TensorBoard (`loss/train`, `loss/val`, `accuracy/train`,
`accuracy/val`, one event file per run) and to wandb (`project=TimeMultiplexedClassification`,
same metrics plus `slm_gnorm/train`), gated on whether `wandb.init` succeeds.

## Evaluation

```bash
python test.py --ckpt ../logs/<run>/model/best.pth
python test.py --ckpt ../logs/<run>/model/best.pth --dataset mnist   # force a different dataset
```

With no `--dataset`, `test.py` evaluates against whichever dataset the checkpoint was
actually trained on (from its saved config); pass `--dataset` to force `mnist`,
`mnist_grating`, `grating`, `fashion`, `cifar10` or `tinyimagenet` explicitly, e.g. for
a generalization check. Outputs: an accuracy/loss printout, a per-class
precision/recall/F1 chart, a row-normalized confusion matrix, a grid of misclassified
examples, and the learned SLM phase masks.

## Running on yijie

Verified 2 Sept 2026: clone at `C:\luxnet\repos\time-multiplexed-classifier`, conda env
`qpi` (python 3.12.7, torch 2.5.1+cu121), 2x RTX 4090. All eleven requirements were already
present in that env. FashionMNIST lives at
`I:\lab-data\time-multiplexed-classifier\FashionMNIST\FashionMNIST\raw`.

Three environment settings are needed; none require code changes:

```bat
set "SPQPI_DATA_ROOT=I:\lab-data\time-multiplexed-classifier"
set "WANDB_MODE=offline"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
```

`PYTHONUTF8` / `PYTHONIOENCODING` matter because the machine's console codepage is GBK, and
`train.py` prints a `x` and a superscript-2 in its parameter summary. Without them the run
dies with `UnicodeEncodeError` before the first step.

**Batch size.** Memory scales with `batch_size * M * (num_layers + 1)` -- the forward
pass holds `[B, M, N_sim, N_sim]` intensity per propagation hop en route to the detector
array. The figures below (`23.1 GB @ batch_size=4`, epoch timings) were measured before
this session's classification pivot, which removed the ~62 M-parameter reconstruction
decoder that dominated memory at the time -- treat them as rough orientation only, not
current fact; they have not been re-measured against the current (decoder-free, optionally
layered) model.

At `batch_size=4` the model used 23.1 GB of the 4090's 24.5 GB. The default `batch_size=5`
was tuned for a 32 GB card and will very likely OOM here. Throughput at batch 4 was
~3.8 it/s, so one FashionMNIST epoch (13,500 steps) took ~58 minutes on one GPU.

### Capping the training set

`--set train_samples=N` caps the dataset *before* the train/val split, so
`validation_ratio` then carves its 10% out of `N`. Use it for quick runs (epoch times
below carry the same pre-pivot caveat as above):

| `train_samples` | steps @ batch 4 | epoch time on one 4090 |
|---|---|---|
| 2,000 | 450 train + 50 val | ~2 min |
| 5,000 | 1,125 + 125 | ~5 min |
| 10,000 | 2,250 + 250 | ~11 min |
| unset (54,000) | 13,500 + 1,500 | ~62 min |

This needed a fix. `_apply_overrides` preserves the type of the existing value, and
`train_samples` and `mnist_cap` both default to `None` -- neither `bool`, `int` nor
`float` -- so the override fell through to the string branch and `--set
train_samples=200` died with `TypeError: slice indices must be integers`. A `None` case
now coerces to int, then float, then leaves it a string.


## Launching detached

An ssh-launched run dies when the session drops. Spawn it so it is not a child of the ssh
session:

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c C:\luxnet\runs\tmc_smoke.bat' }
```

The repo is cloned with a read-only deploy key at `C:/luxnet/ssh/id_ed25519_tmc`, wired in
as `core.sshCommand` in the local clone, so `git pull` works non-interactively and pushes
from that machine are refused by design.

## Provenance

The nine files in `code/` were copied from
[`ary-portes/single-pixel-qpi-paper`](https://github.com/ary-portes/single-pixel-qpi-paper)
at commit `f5b388c1d8191181444970c7c397b7249aa60fd9` (29 Aug 2026). Since then (Sep 2026)
the pipeline was pivoted from phase-image reconstruction to classification:
`config.py`, `model.py`, `loss.py`, `dataloader.py`, `train.py` and `test.py` have all
been substantially rewritten -- the reconstruction decoder is gone, replaced by a 4x5
photodiode array, differential-contrast softmax cross-entropy, and optional learned
diffractive phase layers. `wave_prop.py` gained an optional per-instance `z` argument
so multiple propagators (different fixed hop distances) can coexist; `paths.py` is
unchanged from the original copy. 
