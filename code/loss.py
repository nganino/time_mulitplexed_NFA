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

        # detector window side in sim pixels -- I_vec holds each detector's MEAN
        # intensity over a pd_px x pd_px window (see diffraction_efficiency)
        self.pd_px = int(getattr(config, 'photodiode_pixels', 1))

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

        I_integrated = I_vec.mean(dim=1)          # [B, rows, cols]
        return I_integrated.reshape(B, self.Nf)   # [B, Nf], k = r * cols + c

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

    def diffraction_efficiency(self, I_vec, incident_energy):
        '''
        DIAGNOSTIC ONLY -- not part of the training loss (forward() never calls
        it). Fraction of the energy incident on the phase-key plane that is
        captured by the detector array, for each input sample and phase key.

        I_vec           : [B, T, rows, cols] raw per-detector intensity
                          (model.forward() output, BEFORE the learned scale/bias
                          -- efficiency is a physical energy ratio, the affine
                          readout is not part of it)
        incident_energy : scalar, model.incident_energy()

        Each I_vec entry is a detector's MEAN intensity over its pd_px x pd_px
        window, so it is multiplied by pd_px^2 to get the energy that detector
        captures -- otherwise efficiency would not be comparable across
        detector sizes. Energy lost between detectors (dead space) is NOT
        counted: this measures light usefully landing on detectors, not total
        power surviving to the detector plane.

        incident_energy is over the phase-key footprint only (see
        model.incident_energy), so the ratio is independent of N_sim. It is
        close to, but not strictly, bounded by 1: background plane-wave light
        from outside the key footprint can diffract into the layer apertures.

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
