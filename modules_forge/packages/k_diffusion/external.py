import torch
from torch import nn

from . import sampling


class ForgeScheduleLinker(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    @property
    def sigmas(self):
        return self.predictor.sigmas

    @property
    def log_sigmas(self):
        return self.predictor.sigmas.log()

    @property
    def sigma_min(self):
        return self.predictor.sigma_min()

    @property
    def sigma_max(self):
        return self.predictor.sigma_max()

    def get_sigmas(self, n=None):
        if n is None:
            return sampling.append_zero(self.sigmas.flip(0))
        # Use the predictor's own timestep for sigma_max so that flow models
        # with multiplier != 1000 (e.g. Anima with multiplier=1.0) produce a
        # correctly-ranged linspace. For standard DDPM predictors this returns
        # the same integer index (~999) as before.
        t_max = self.sigma_to_t(self.sigmas[-1])
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        return self.predictor.timestep(sigma)

    def t_to_sigma(self, t):
        return self.predictor.sigma(t)
