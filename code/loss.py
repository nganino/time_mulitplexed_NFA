import torch
import torch.nn as nn

# =========================================================================== #
#  NONLINEAR FUNCTION APPROXIMATION (NFA) -- loss                             #
#                                                                              #
#  model.py returns the RAWEST per-detector intensity, I_vec : [B, T, rows,   #
#  cols] (T == M phase keys) -- no averaging or normalization happens there.  #
#  Everything downstream of the raw physics lives here:                      #
#    1. average over the M phase-key axis                                    #
#    2. flatten the detector grid to one value per target function, k = r *  #
#       pd_num_cols + c (a plain row-major reshape)                          #
#    3. readout to f_hat. Val / test (every reported number, p_range given):  #
#       min-max, paper Eq. 9, since 2026-10-23:                              #
#         f_hat_k(a) = (P_k(a) - Pmin) / (Pmax - Pmin),                       #
#         Pmin = min_k min_a P_k(a),  Pmax = max_k max_a P_k(a)               #
#       over the full dense a-grid, against targets normalized on the same    #
#       grid. Training (p_range None) uses config.train_readout:              #
#         'affine' (default since 2026-10-24): learned global scale * P +     #
#           offset against the global targets, discarded after training      #
#           (min-max trains 20-680x worse; the mismatch is intentional)       #
#         'minmax': per-batch Pmin / Pmax, targets min-max'd over the batch   #
#         'globalminmax': Pmin / Pmax from the full a-grid, fixed per epoch   #
#    4. MSE against the target (already normalized to [0,1] by               #
#       target_functions.TargetFunctionSet)                                  #
# =========================================================================== #

class FunctionApproxLoss(nn.Module):
    '''
    MSE between the model's frame-averaged output f_hat(a) and the true (already [0,1]-normalized) target
    function values. Evaluation readout: min-max. Training readout: train_readout; only 'affine' has trainable
    parameters (a global scale and offset, used in the training loss only).
    '''

    def __init__(self, config):
        super().__init__()
        self.rows = int(config.pd_num_rows)
        self.cols = int(config.pd_num_cols)
        self.Nf   = int(config.Nf)
        assert self.rows * self.cols == self.Nf, (
            f'pd_num_rows * pd_num_cols ({self.rows * self.cols}) != config.Nf ({self.Nf})'
        )

        # detector window side in sim pixels -- I_vec holds each detector's MEAN
        # intensity over a pd_px x pd_px window (see diffraction_efficiency)
        self.pd_px = int(getattr(config, 'photodiode_pixels', 1))

        self.loss_type = getattr(config, 'loss_type', 'mse')
        assert self.loss_type == 'mse', f"only 'mse' is implemented, got {self.loss_type!r}"
        self.mse_fn = nn.MSELoss()

        # Training-only readout (2026-10-24): 'affine' (default) = a learned global scale (softplus, init 1)
        # and offset (init 0), f_hat = scale * I + bias against the globally normalised targets, used ONLY in the
        # training loss. Evaluation (p_range given) is always global min-max, so reported numbers never use it.
        self.train_readout = getattr(config, 'train_readout', 'affine')   # or 'minmax' (per-batch min-max, see below)
        # or 'globalminmax': the test's global min-max, with Pmin / Pmax measured on the full a-grid at the start of every
        # epoch (train.py sets train_p_range) and held fixed as constants during the epoch, against the global targets
        # or 'perfuncminmax' (2026-11-13, matches the bench's per-function readout): every function k min-max'd over the
        # batch's a values, f_hat_k = (I_k - min_a I_k) / (max_a I_k - min_a I_k), targets the same; validation in
        # train.py then also uses per-function min-max over the a-grid (test.py stays global)
        # or 'perfuncaffine' (2026-11-14): a learned scale + offset PER FUNCTION (Nf each, training only) against targets
        # min-max'd per function on the 2001-point calibration grid; independent of the micro-batch size (perfuncminmax
        # gets min == max at micro-batch 1). Val per-function min-max, as perfuncminmax.
        assert self.train_readout in ('minmax', 'affine', 'globalminmax', 'perfuncminmax', 'perfuncaffine'), \
            f'unknown train_readout {self.train_readout!r}'
        self.per_function = self.train_readout in ('perfuncminmax', 'perfuncaffine')
        self.train_p_range = None
        if self.train_readout in ('affine', 'perfuncaffine'):
            shape = (self.Nf,) if self.train_readout == 'perfuncaffine' else ()
            self._raw_scale = nn.Parameter(torch.full(shape, 0.5413))  # softplus(0.5413) = 1
            self.bias = nn.Parameter(torch.zeros(shape))
        if self.train_readout == 'perfuncaffine':
            from target_functions import TargetFunctionSet
            t = TargetFunctionSet(config)(torch.linspace(config.a_min, config.a_max, 2001))   # [2001, Nf], global-normalised
            self.register_buffer('t_min', t.amin(0), persistent=False)
            self.register_buffer('t_max', t.amax(0), persistent=False)

    def _detector_to_function(self, I_vec):
        '''
        I_vec : [B, T, rows, cols] raw per-detector intensity (model.forward() output, T == M phase keys)
        returns [B, Nf] : averaged over T, flattened per-detector -> per-function
        '''
        B = I_vec.shape[0]
        I_integrated = I_vec.mean(dim=1)          # [B, rows, cols] average over M phase keys
        return I_integrated.reshape(B, self.Nf)   # [B, Nf], k = r * cols + c

    @staticmethod
    def minmax_range(I, per_function=False):
        '''(Pmin, Pmax) over every entry of I (all samples a and all detectors k); per_function: [1, Nf] each, over a only.'''
        if per_function:
            return I.amin(0, keepdim=True), I.amax(0, keepdim=True)
        return I.min(), I.max()

    @staticmethod
    def normalize(I, pmin, pmax):
        '''Min-max readout (paper Eq. 9): (I - Pmin) / (Pmax - Pmin).'''
        return (I - pmin) / (pmax - pmin).clamp_min(1e-12)

    def forward(self, I_vec, target, p_range=None):
        '''
        I_vec   : [B, T, rows, cols]  raw per-detector intensity, model.forward() output
        target  : [B, Nf]  true (already [0,1]-normalized) function values
        p_range : (Pmin, Pmax) to normalize with (evaluation: from the full a-grid); None -> training: f_hat AND
                  target each min-max normalized over this batch (jointly over a and functions), differentiable;
                  with train_readout 'affine' instead the learned scale * I + bias against the global targets

        Returns
        -------
        loss  : scalar training loss (MSE)
        f_hat : [B, Nf]  normalized model output (for logging/plotting)
        '''
        I = self._detector_to_function(I_vec)                    # [B, Nf]
        if p_range is None and self.train_readout in ('affine', 'perfuncaffine'):
            f_hat = I * torch.nn.functional.softplus(self._raw_scale) + self.bias   # training-only learned readout
            if self.train_readout == 'perfuncaffine':
                target = self.normalize(target, self.t_min, self.t_max)            # per-function [0, 1]
        elif p_range is None and self.train_readout == 'globalminmax':
            assert self.train_p_range is not None, 'globalminmax: train.py must set train_p_range each epoch'
            f_hat = self.normalize(I, *self.train_p_range)          # constants from the epoch's full-grid calibration
        elif p_range is None and self.per_function:
            f_hat = self.normalize(I, *self.minmax_range(I, True))          # per function, over the batch's a values
            target = self.normalize(target, *self.minmax_range(target, True))
        elif p_range is None:
            # Training: a batch rarely holds the targets' global extremes (Np64 Nf1024: batch-64 target max ~0.84), so
            # normalising only f_hat by the batch's own Pmin / Pmax would demand a batch-dependent rescaling. Normalise the
            # targets jointly over the same batch too; if P matches f up to one affine map, both sides are then identical.
            f_hat = self.normalize(I, *self.minmax_range(I))
            target = self.normalize(target, *self.minmax_range(target))
        else:
            f_hat = self.normalize(I, *p_range)                   # evaluation: global Pmin / Pmax, global targets
        loss = self.mse_fn(f_hat, target)
        return loss, f_hat

    def diffraction_efficiency(self, I_vec, incident_energy):
        '''
        DIAGNOSTIC ONLY -- not part of the training loss (forward() never calls
        it). Fraction of the energy incident on the phase-key plane that is
        captured by the detector array, for each input sample and phase key.

        I_vec           : [B, T, rows, cols] raw per-detector intensity
                          (model.forward() output, BEFORE the min-max readout
                          -- efficiency is a physical energy ratio)
        incident_energy : scalar, model.incident_energy()

        Each I_vec entry is a detector's MEAN intensity over its pd_px x pd_px
        window, so it is multiplied by pd_px^2 to get the energy that detector
        captures
        returns [B, T] efficiency; average over both axes for a single number
        (sum over detectors and mean over B/T commute).
        '''
        captured = I_vec.sum(dim=(-2, -1)) * self.pd_px ** 2   # [B, T]
        return captured / incident_energy

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
