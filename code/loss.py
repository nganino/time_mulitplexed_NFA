import torch
import torch.nn as nn

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
#    3. min-max normalize into f_hat(a) using RUNNING Pmin/Pmax buffers       #
#       (paper Eq. 9, made "online" via an EMA -- not specified in paper,    #
#       see design discussion) -- updated only while self.training is True,  #
#       frozen at eval                                                       #
#    4. MSE against the target (already normalized to [0,1] by               #
#       target_functions.TargetFunctionSet)                                  #
#                                                                              #
#  IMPORTANT: this module has real state (the running buffers), so           #
#  train.py must call .train() / .eval() on the LOSS module too, not just    #
#  the model -- otherwise running-stat updates would (or wouldn't) happen    #
#  at the wrong times.                                                       #
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

        self.eps = 1e-8
        # NOTE: not defined in paper -- EMA rate for the running Pmin/Pmax
        # buffers (analogous to BatchNorm momentum), see config.norm_momentum.
        self.momentum = float(getattr(config, 'norm_momentum', 0.1))

        self.register_buffer('running_min', torch.zeros(()))
        self.register_buffer('running_max', torch.ones(()))
        self.register_buffer('_stats_initialized', torch.tensor(False))

    def _detector_to_function(self, I_vec):
        '''
        I_vec : [B, T, rows, cols] raw per-detector intensity (model.forward()
                output, T == M phase keys)
        returns [B, Nf] : averaged over T, flattened per-detector -> per-function
        '''
        B = I_vec.shape[0]
        I_mean = I_vec.mean(dim=1)          # [B, rows, cols] -- average over phase keys
        return I_mean.reshape(B, self.Nf)   # [B, Nf], k = r * cols + c

    def _update_running_stats(self, I_summed):
        batch_min = I_summed.min().detach()
        batch_max = I_summed.max().detach()
        if not bool(self._stats_initialized):
            self.running_min.copy_(batch_min)
            self.running_max.copy_(batch_max)
            self._stats_initialized.fill_(True)
        else:
            self.running_min.mul_(1 - self.momentum).add_(self.momentum * batch_min)
            self.running_max.mul_(1 - self.momentum).add_(self.momentum * batch_max)

    def normalize(self, I_summed):
        '''
        PAPER Eq. 9 min-max normalization, using running Pmin/Pmax stats
        (updated during training via self.training, frozen at eval).
        '''
        if self.training:
            self._update_running_stats(I_summed)
        denom = (self.running_max - self.running_min).clamp(min=self.eps)
        return (I_summed - self.running_min) / denom

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
