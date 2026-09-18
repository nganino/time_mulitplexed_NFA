import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================== #
#  NONLINEAR FUNCTION APPROXIMATION (NFA) -- loss                             #
#                                                                              #
#  model.py returns the RAWEST per-detector intensity, I_vec : [B, T, rows,   #
#  cols] (T == M phase keys) -- no averaging or normalization happens there.  #
#  Everything downstream of the raw physics lives here:                      #
#    1. average over the M phase-key axis (the "wisdom of the crowd"         #
#       ensembling step)                                                     #
#    2. flatten the detector grid to one value per target function, k = r *  #
#       pd_num_cols + c (config.py's stated convention -- a plain reshape,   #
#       since PyTorch's reshape is row-major)                                #
#    3. affine-normalize into f_hat(a) = I_summed * scale + bias (paper      #
#       Eq. 9's min-max idea, but scale/bias are LEARNED parameters, updated #
#       by backprop alongside the phases -- NOT an EMA of running Pmin/Pmax  #
#       batch statistics, see design discussion below)                       #
#    4. MSE against the target (already normalized to [0,1] by               #
#       target_functions.TargetFunctionSet)                                  #
#                                                                              #
#  HISTORY: this used to track running_min/running_max via an EMA (momentum  #
#  = config.norm_momentum, now OBSOLETE/unused), updated only during         #
#  training and frozen at eval -- mirroring the paper's Eq. 9 more literally.#
#  A learning-rate sweep at M=1 (2026-09-17) showed val loss spiking to      #
#  values like 146 (impossible for a bounded-[0,1] MSE) at EVERY tested LR,  #
#  just less violently at lower LR. Inspecting checkpoints directly showed   #
#  why: raw detector power collapses by 3-6 orders of magnitude over         #
#  training AND swings unpredictably by up to ~1000x between checkpoints few #
#  epochs apart -- nothing in the loss constrained overall optical power, so #
#  the optimizer was free to let it wander. The EMA (a fixed-momentum,       #
#  out-of-loop tracker) couldn't react fast enough to those sudden swings,   #
#  so for a batch or two the normalization denominator was badly mismatched #
#  with the current scale, producing exactly these blow-ups. Replacing it   #
#  with a plain LEARNED affine transform removes that failure mode: scale/  #
#  bias move smoothly via the SAME gradient descent as everything else (no  #
#  separate momentum hyperparameter to desync), while still preserving the   #
#  reason Eq. 9-style normalization exists in the first place -- the model   #
#  only has to learn the right *shape* of f(a), not hit an absolute physical #
#  intensity unit.                                                           #
# =========================================================================== #

class FunctionApproxLoss(nn.Module):
    '''
    MSE between the model's simulated, phase-key-averaged, normalized output
    f_hat(a) and the true (already [0,1]-normalized) target function values.
    '''

    def __init__(self, config):
        super().__init__()
        self.rows = int(config.pd_num_rows)
        self.cols = int(config.pd_num_cols)
        self.Nf   = int(config.Nf)
        assert self.rows * self.cols == self.Nf, (
            f'pd_num_rows * pd_num_cols ({self.rows * self.cols}) != config.Nf ({self.Nf})'
        )

        self.loss_type = getattr(config, 'loss_type', 'mse')
        assert self.loss_type == 'mse', f"only 'mse' is implemented, got {self.loss_type!r}"
        self.mse_fn = nn.MSELoss()

        # Learned affine readout: f_hat = I_summed * scale + bias. scale is
        # parameterized through softplus to stay positive (higher intensity
        # -> higher f_hat, a monotonic mapping) and numerically stable; both
        # are trained by backprop like any other parameter (see train.py's
        # opt_readout). Init: scale=1, bias=0 (raw_scale = softplus^-1(1)).
        self._raw_scale = nn.Parameter(torch.tensor(0.5413))
        self.bias       = nn.Parameter(torch.zeros(()))

    @property
    def scale(self):
        return F.softplus(self._raw_scale)

    def _detector_to_function(self, I_vec):
        '''
        I_vec : [B, T, rows, cols] raw per-detector intensity (model.forward()
                output, T == M phase keys)
        returns [B, Nf] : averaged over T, flattened per-detector -> per-function
        '''
        B = I_vec.shape[0]
        I_mean = I_vec.mean(dim=1)          # [B, rows, cols] -- average over phase keys
        return I_mean.reshape(B, self.Nf)   # [B, Nf], k = r * cols + c

    def normalize(self, I_summed):
        '''Learned affine transform -- see module-level HISTORY comment for
        why this replaced an EMA-tracked running Pmin/Pmax.'''
        return I_summed * self.scale + self.bias

    def forward(self, I_vec, target):
        '''
        I_vec  : [B, T, rows, cols]  raw per-detector intensity, model.forward() output
        target : [B, Nf]  true (already [0,1]-normalized) function values

        Returns
        -------
        loss  : scalar training loss (MSE)
        f_hat : [B, Nf]  normalized model output (for logging/plotting)
        '''
        I_summed = self._detector_to_function(I_vec)   # [B, Nf]
        f_hat = self.normalize(I_summed)                 # [B, Nf]
        loss = self.mse_fn(f_hat, target)
        return loss, f_hat

    @staticmethod
    def per_function_rmse(f_hat, target):
        '''
        f_hat, target : [N, Nf]  (e.g. concatenated over an entire dense
        val/test grid pass) -- returns [Nf] RMSE per target function,
        approximating the paper's continuous RMSE integral (Eq. 16) with
        these N samples of a. NOT the training loss -- a diagnostic/plotting
        metric (see Fig. 2a/2b/2c-style reporting).
        '''
        return torch.sqrt(((f_hat - target) ** 2).mean(dim=0))
