'''
Time-Multiplexed NFA — config parameters 

Data paths point to the parent project directory so the same datasets are shared.
Logs are written to conv_decoder/logs/ to keep runs separate.
'''

import sys, os
# Make parent project importable (dataloader, loss, wave_prop, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as _paths   # every filesystem location resolves here

from configobj import ConfigObj
import numpy as np
import datetime


def _build_run_name(tc):
    datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    samples = getattr(tc, 'train_samples', None)
    samples_str = f'samples{samples}' if samples is not None else 'samplesMax'
    loss = f'sum' if getattr(tc, 'loss_mode', None) == 'sum' else 'vote'
    return (f'{datetime_str}'
            f'-M{tc.M}-C{tc.C}-K{tc.num_layers}'
            f'-Spacings{1000*tc.object_slm_spacing:.0f}mm-{1000*tc.slm_first_layer_spacing:.0f}mm-{1000*tc.interlayer_spacing:.1f}mm-{1000*tc.last_layer_ccd_spacing:.0f}mm'
            f'-batchsize{tc.batch_size}-lrslm{tc.lr_slm:.0e}-lrlayer{tc.lr_layer:.0e}-pdsize{1000*tc.photodiode_size:.1f}mm'
            f'-{samples_str}-{loss}')


def config_to_dict(tc):
    return {k: (v.item() if hasattr(v, 'item') else v)
            for k, v in vars(tc).items()
            if not k.startswith('_') and (isinstance(v, (int, float, str, bool, type(None)))
                                           or hasattr(v, 'item'))}


def _build_log_paths(tc):
    '''Single source of truth for where a run's output goes, derived from tc.run_name.
    '''
    tc.log_dir     = os.path.join(_paths.LOG_DIR, 'maj_voting_C_sweep', tc.run_name)
    tc.image_dir   = os.path.join(tc.log_dir, 'images')
    tc.model_dir   = os.path.join(tc.log_dir, 'model')
    tc.tfboard_dir = os.path.join(tc.log_dir, 'tfboard')


def recompute_derived(tc):
    '''Recompute all params that depend on base physical values.'''
    tc.slm_bin        = int(tc.slm_dx / tc.sim_dx)
    tc.slm_x_num_sim  = tc.slm_x_num * tc.slm_bin
    tc.obj_bin        = int(tc.obj_dx / tc.sim_dx)
    tc.obj_x_num_sim  = tc.data_x_num * tc.obj_bin
    tc.photodiode_pixels = int(tc.photodiode_size / tc.sim_dx)

    tc.pd_row_spacing_px = int(tc.pd_row_spacing / tc.sim_dx)
    tc.pd_col_spacing_px = int(tc.pd_col_spacing / tc.sim_dx)

    tc.layer_bin      = int(tc.layer_dx / tc.sim_dx)
    tc.layer_size_sim = tc.layer_size * tc.layer_bin
    # total number of learnable masks (T = M * C)
    tc.T = int(getattr(tc, 'M', 1) * getattr(tc, 'C', 1))
    # Keep run_name (and everything derived from it) in sync with M/C/spacings/etc --
    # train.py's main() calls recompute_derived() right after applying --set overrides,
    # before training starts, so this must be refreshed here too (init_params() alone
    # only captures the *default* values in the name/paths).
    tc.run_name = _build_run_name(tc)
    _build_log_paths(tc)


def init_params():
    tc = ConfigObj()

    # ------------------------------------------------------------------ #
    #  Physical constants & optical parameters                            #
    # ------------------------------------------------------------------ #
    nm, um, mm, cm = 1e-9, 1e-6, 1e-3, 1e-2

    tc.wavelength = 635 * nm
    tc.ridx_air   = 1.0

    # ------------------------------------------------------------------ #
    #  Simulation grid                                                    #
    # ------------------------------------------------------------------ #
    tc.sim_dx = 4 * um
    tc.N_sim  = 2000
    tc.asm_pad_factor = 1

    # ------------------------------------------------------------------ #
    #  SLM  (hardware device)                                             #
    # ------------------------------------------------------------------ #
    tc.slm_dx      = 8 * um
    tc.slm_x_num   = 500
    tc.slm_bin     = int(tc.slm_dx / tc.sim_dx)
    tc.slm_x_num_sim = tc.slm_x_num * tc.slm_bin

    tc.slm_hw_x      = 1920
    tc.slm_hw_y      = 1080
    tc.slm_bit_depth = 8

    tc.M               = 5 # # of "members" of a country (# of phase biases shown per integration period)
    tc.C               = 1  # # of countries (# of seperate integration periods) 
    tc.T               = tc.M * tc.C # total number of learnable phase biases
    tc.mask_init_method = 'normal'
    tc.mask_init_std   = 0.5

    # ------------------------------------------------------------------ #
    #  Object (MNIST phase images)                                        #
    # ------------------------------------------------------------------ #
    tc.num_classes     = 10
    tc.softmax_T       = 0.1   # softmax temperature; loss.py divides the [-1,1]
                                # differential contrast by softmax_T before cross-entropy

    tc.loss_mode = 'vote' # 'sum' = sum over countries, 'vote' = majority vote (each country votes for its argmax, then the final prediction is the majority vote of the individual votes;
                        # ties are broken by the country with the highest confidence score among the tied votes)

    tc.data_x_num   = 100
    tc.obj_dx       = tc.slm_dx
    tc.obj_bin      = int(tc.obj_dx / tc.sim_dx)
    tc.obj_x_num_sim = tc.data_x_num * tc.obj_bin

    tc.N                = 32   #  read by the legacy image-reconstruction decoder in finetune.py
    tc.input_phase_max = np.pi

    # ------------------------------------------------------------------ #
    #  Diffractive Layers                                                 #
    # Each layer is shared across all T time-multiplexed measurements.                   #
    # ------------------------------------------------------------------ #
    tc.num_layers    = 3
    tc.layer_dx      = 8 * um
    tc.layer_size    = 500
    tc.layer_bin      = int(tc.layer_dx / tc.sim_dx)
    tc.layer_size_sim = tc.layer_size * tc.layer_bin
    tc.object_slm_spacing = 3 * cm
    tc.interlayer_spacing = 5 * mm # set after sweeping interlayer spacing 9/11-9/14
    tc.slm_first_layer_spacing = 3 * cm
    tc.last_layer_ccd_spacing = 3 * cm

    # ------------------------------------------------------------------ #
    #  Propagation distance                                               #
    # ------------------------------------------------------------------ #
    tc.z_slm_ccd = 5.2 * cm

    # ------------------------------------------------------------------ #
    #  Photodiode array  (differential-detection classification head)     #
    #  pd_num_rows x pd_num_cols detectors, each photodiode_size square.  #
    #  Rows 0..pd_num_rows/2-1 = positive detectors,                      #
    #  rows pd_num_rows/2..end = negative detectors (paired by column).   #
    # ------------------------------------------------------------------ #
    tc.photodiode_size   = .1 * mm            # each detector: 0.1mm x 0.1mm
    tc.photodiode_pixels = int(tc.photodiode_size / tc.sim_dx)
    tc.pd_num_rows = 4
    tc.pd_num_cols = 5
    tc.num_photodiodes = tc.pd_num_rows * tc.pd_num_cols

    tc.pd_row_spacing = 2 * tc.photodiode_size  # center-to-center spacing, row direction
    tc.pd_col_spacing = 2 * tc.photodiode_size  # center-to-center spacing, column direction
    tc.pd_row_spacing_px = int(tc.pd_row_spacing / tc.sim_dx)
    tc.pd_col_spacing_px = int(tc.pd_col_spacing / tc.sim_dx)

    # Offset of the whole array's center, in sim-grid pixels relative to the optical
    # axis (N_sim/2). Default 0 centers the array on the axis -- for the default 4x5
    # array that puts the axis exactly on the middle column and exactly between the
    # two middle rows. Nonzero shifts the whole array (e.g. to model misalignment).
    tc.detector_offset_y = 0
    tc.detector_offset_x = 0

    # ------------------------------------------------------------------ #
    #  Dataset  (paths point to parent project's data directory)          #
    # ------------------------------------------------------------------ #
    tc.dataset      = 'cifar10'  # 'mnist', 'FashionMNIST', 'mnist_grating',
                                      # 'grating', 'cifar10', 'tinyimagenet'
    # Datasets live outside this repo. paths.py resolves the root and honours
    # SPQPI_DATA_ROOT, so a clone elsewhere only has to set one variable.
    _data = _paths.DATA_ROOT
    tc.data_path         = os.path.join(_data, 'MNIST', 'raw')
    tc.grating_data_path = os.path.join(_data, 'Grating')
    tc.fashion_data_path = os.path.join(_data, 'FashionMNIST', 'FashionMNIST', 'raw')
    tc.cifar10_data_path       = os.path.join(_data, 'CIFAR10', 'cifar-10-python')
    tc.tinyimagenet_data_path  = os.path.join(_data, 'tiny-imagenet-200')
    tc.emnist_data_path        = os.path.join(_data, 'EMNIST')
    tc.train_samples = 20000 # cap the number of training samples (None = use all)

    tc.val_samples   = None # hardcode a validation set size (None = fall back to validation_ratio), Requires train_samples to be set (not None) when used.
    tc.mnist_cap     = None
    tc.grating_only  = False
    tc.image_transform = None

    # ------------------------------------------------------------------ #
    #  Training hyper-parameters                                          #
    # ------------------------------------------------------------------ #
    tc.batch_size       = 12
    tc.test_batch_size  = 4
    tc.max_epoch        = 100
    tc.validation_ratio = 0.1   # only used when val_samples is None -- see val_samples above
    tc.seed             = 59

    tc.num_workers = 0

    tc.lr_slm     = 1e-2
    tc.lr_layer   = 1e-2

    tc.slm_warmup_epochs = 0

    # End-to-end measurement-noise injection (SLM + decoder co-adapt to the sim->real gap).
    tc.meas_noise_std = 0

    tc.freeze_slm = False

    # ------------------------------------------------------------------ #
    #  Logging & checkpoints  (written to conv_decoder/logs/)             #
    # ------------------------------------------------------------------ #
    tc.checkpoint_save  = 3
    tc.checkpoint_print = 1

    tc.run_name = _build_run_name(tc)
    _build_log_paths(tc)

    tc.ckpt_to_load = None  # path to a checkpoint to load (None = start from scratch)

    # When loading ckpt_to_load, load ONLY the model weights (skip epoch/optimizer/scheduler) so
    # training starts a fresh fine-tune from epoch 0 with a new LR schedule. False = resume.
    tc.load_weights_only = False

    return tc
