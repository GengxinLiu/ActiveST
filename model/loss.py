from torch import nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Distillation Loss
    :param T:
    :param mode: ['ce', 'kl']
    """

    def __init__(self, T, mode='ce'):
        super(DistillationLoss, self).__init__()
        self.T = T
        if mode == 'ce':
            self.fn = self._cross_entropy
        elif mode == 'kl':
            self.fn = self._kl_divergence
        else:
            raise ValueError(f'Invalid mode {mode}, mode can only in [`ce`, `kl`]')

    def _cross_entropy(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return -(p_t * p_s).sum(1).mean()

    def _kl_divergence(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / p_s.shape[0]

    def forward(self, y_s, y_t):
        return self.fn(y_s, y_t)
