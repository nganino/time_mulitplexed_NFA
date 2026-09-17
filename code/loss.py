'''
Time-Multiplexed Classifier — loss functions

ClassificationLoss: softmax cross-entropy over normalized differential
photodiode-pair contrasts.

The photodiode array (config.pd_num_rows x config.pd_num_cols) is read out by
model.py as I_vec : [B, T, pd_num_rows, pd_num_cols] -- one mean-intensity reading
per detector, per one of the T = M * C time-multiplexed SLM phase biases (M members
per country, C countries -- see class docstring for the country/member split).

1. Split T into C countries of M members each and integrate over each country's M
   members separately: I_country = I_vec.view(B, C, M, rows, cols).sum(dim=2) ->
   [B, C, pd_num_rows, pd_num_cols]. Any single phase bias might not by itself
   produce the right classification; summing a country's M biases lets that
   country's ensemble average out into one vote.
2. Split rows into a positive half (rows 0 .. pd_num_rows/2 - 1) and a negative half
   (rows pd_num_rows/2 .. pd_num_rows - 1), paired row-for-row by the offset
   pd_num_rows/2 -- row r (positive) with row r + pd_num_rows/2 (negative), for every
   column. That gives (pd_num_rows/2) x pd_num_cols == num_classes pairs. With the
   default 4x5 array: row0 pairs with row2, row1 pairs with row3, each across all 5
   columns.
3. Per pair, per country, the class score is the normalized contrast, divided by
   the softmax temperature softmax_T:
       score = [ (I+ - I-) / (I+ + I- + eps) ] / softmax_T
   The un-divided contrast is confined to [-1, 1] (sum-over-M vs mean-over-M give the
   exact same value here -- the M-count is a constant factor on both I+ and I-, so it
   cancels out of the ratio); dividing by softmax_T < 1 widens that range before
   softmax -- see class docstring. This gives per_country_scores : [B, C, num_classes].
4. Combine the C countries' scores into a final prediction, per config.loss_mode:

   'sum' (default): aggregated_scores = per_country_scores.sum(dim=1) -> [B, num_classes]
       ("wisdom of the crowd" -- each country's independent vote adds into the final
       decision). loss = CE(aggregated_scores, target); pred = argmax(aggregated_scores).

   'vote': each country casts one vote for its own argmax class; the class with the
       most votes wins. Ties are broken by the highest individual confidence (max
       score) among the countries that voted for a tied class.

   'vote' uses a hard per-country argmax to decide/tally votes, which has no usable
   gradient -- so for this mode, `loss` is instead the per-country-averaged
   cross-entropy: mean_c( CE(per_country_scores[:, c, :], target) ). `pred` (used for
   accuracy) is the actual vote decision rule described above, not derived from this
   loss.
'''

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    '''
    Softmax cross-entropy over normalized differential photodiode-pair contrasts.

    Class index for row-pair r (0-indexed within the positive half) and column c is
    `r * pd_num_cols + c` -- row-pair 0's columns fill classes 0..pd_num_cols-1,
    row-pair 1's columns fill the next pd_num_cols, and so on.

    '''

    def __init__(self, config):
        super().__init__()
        self.rows = config.pd_num_rows
        self.cols = config.pd_num_cols
        assert self.rows % 2 == 0, 'pd_num_rows must be even (positive/negative halves)'
        self.half = self.rows // 2
        self.num_classes = self.half * self.cols
        assert self.num_classes == config.num_classes, (
            f'pd_num_rows/2 * pd_num_cols ({self.num_classes}) != '
            f'config.num_classes ({config.num_classes})'
        )

        self.eps = 1e-8
        # softmax temperature -- NOT the same thing as config.T / self.T_total below,
        # which is the total mask count (M * C). Do not fall back to config.T here.
        self.softmax_T = float(getattr(config, 'softmax_T', 0.1))

        self.M = int(getattr(config, 'M', 1))
        self.C = int(getattr(config, 'C', 1))
        # total number of learnable masks (T == M * C). Prefer config.T if present.
        self.T_total = int(getattr(config, 'T', self.M * self.C))

        self.loss_mode = getattr(config, 'loss_mode', 'sum')
        assert self.loss_mode in ('sum', 'vote'), (
            f"config.loss_mode must be one of 'sum' / 'vote', "
            f"got {self.loss_mode!r}"
        )

        self.ce_fn = nn.CrossEntropyLoss()

    def scores(self, I_vec):
        '''
        I_vec   : [B, T, pd_num_rows, pd_num_cols]  model.py's measurement output
                  where T == M * C (total masks). Masks are ordered so that the
                  first M belong to country 0, next M to country 1, etc.

        returns : (aggregated_scores, per_country_scores, I_pos, I_neg)
                  `per_country_scores` : [B, C, num_classes]  temperature-scaled
                                      contrast scores computed per country
                  `aggregated_scores`   : [B, num_classes]  sum over countries
                                      (used for final classification loss)
                  `I_pos`, `I_neg`      : [B, C, half, cols]  raw per-country detector
                                      intensities (positive / negative halves), before
                                      the contrast ratio or softmax_T scaling -- exposed
                                      for diagnostics/plotting (see train.py save_images)
        '''
        B, T, rows, cols = I_vec.shape
        assert T == self.T_total, (
            f'I_vec T dimension ({T}) does not match expected total masks ({self.T_total})'
        )

        # reshape to [B, C, M, rows, cols]
        I_resh = I_vec.view(B, self.C, self.M, rows, cols)

        # sum over members (M) to get per-country integrated detector readings: [B, C, rows, cols]
        I_country = I_resh.sum(dim=2) # [B, C, rows, cols]

        I_pos = I_country[:, :, :self.half, :]   # [B, C, half, cols]
        I_neg = I_country[:, :, self.half:, :]   # [B, C, half, cols]
        contrast = (I_pos - I_neg) / (I_pos + I_neg + self.eps)   # in [-1, 1]
        contrast = contrast / self.softmax_T

        per_country_scores = contrast.reshape(B, self.C, -1)   # [B, C, num_classes]
        aggregated_scores = per_country_scores.sum(dim=1)     # [B, num_classes]
        return aggregated_scores, per_country_scores, I_pos, I_neg

    def forward(self, I_vec, target):
        '''
        I_vec  : [B, T, pd_num_rows, pd_num_cols]  model.py's measurement output,
                 T = M * C (total masks across all countries)
        target : [B]  integer class labels in [0, num_classes)

        Returns
        -------
        loss         : softmax cross-entropy (scalar, for .backward())
        class_scores : [B, num_classes]  aggregated (summed-over-countries) temperature-
                       scaled contrast scores (for logging; this is what argmax'd for pred)
        pred         : [B]  predicted class = argmax(class_scores)
        per_country_scores : [B, C, num_classes]  each country's individual vote,
                       before summing (for diagnostics/plotting)
        '''
        aggregated_scores, per_country_scores, _, _ = self.scores(I_vec)

        if self.loss_mode == 'sum':
            loss = self.ce_fn(aggregated_scores, target)
            pred = aggregated_scores.argmax(dim=1)
            return loss, aggregated_scores, pred, per_country_scores

        B = per_country_scores.shape[0]
        per_country_flat = per_country_scores.reshape(B * self.C, self.num_classes) #[B*C, num_classes]
        target_flat      = target.unsqueeze(1).expand(-1, self.C).reshape(-1) # [B*C]
        loss = self.ce_fn(per_country_flat, target_flat)

        individual_preds      = per_country_scores.argmax(dim=2)       # [B, C]
        individual_confidence = per_country_scores.max(dim=2).values   # [B, C]
        pred = self._vote_predict(individual_preds, individual_confidence)

        return loss, aggregated_scores, pred, per_country_scores

    def _vote_predict(self, individual_preds, individual_confidence):
        '''
        Majority vote: each country casts one flat vote for its own argmax class;
        the class with the most votes wins. Ties are broken by the highest individual
        confidence (max score) among the countries that voted for a tied class.

        individual_preds      : [B, C]  each country's own argmax class
        individual_confidence : [B, C]  each country's own max score (confidence)
        '''
        B, C = individual_preds.shape
        device = individual_preds.device
        pred = torch.empty(B, dtype=torch.long, device=device)

        for b in range(B):
            votes = individual_preds[b]
            conf  = individual_confidence[b]
            tally     = torch.zeros(self.num_classes, device=device)
            best_conf = torch.full((self.num_classes,), float('-inf'), device=device)
            for c in range(C):
                cls = int(votes[c].item())
                tally[cls] += 1.0
                if conf[c] > best_conf[cls]:
                    best_conf[cls] = conf[c]

            tied = (tally == tally.max()).nonzero(as_tuple=True)[0]
            pred[b] = tied[0] if tied.numel() == 1 else tied[best_conf[tied].argmax()]

        return pred
