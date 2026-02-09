# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.75/comfy/k_diffusion/sampling.py

import math
from functools import partial

import torch
import torchsde
from scipy import integrate
from tqdm.auto import trange

from backend.patcher.base import set_model_options_post_cfg_function

from . import utils

def _sigma_fn(t):
    return t.neg().exp()

def _t_fn(sigma):
    return sigma.log().neg()

def _is_const(sampling) -> bool:
    return sampling.prediction_type == "const"

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)"""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule"""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)

def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.0, device="cpu"):
    """Constructs an polynomial in log sigma noise schedule"""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)

def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Constructs a continuous VP noise schedule"""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.special.expm1(beta_d * t**2 / 2 + beta_min * t))
    return append_zero(sigmas)

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step"""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy"""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = kwargs.pop("cpu", True)
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.pop("w0", None)
        if w0 is None:
            w0 = torch.zeros_like(x)
        self.batched = False
        if seed is None:
            seed = (torch.randint(0, 2**63 - 1, ()).item(),)
        elif isinstance(seed, (tuple, list)):
            if len(seed) != x.shape[0]:
                raise ValueError("Passing a list or tuple of seeds to BatchedBrownianTree requires a length matching the batch size.")
            self.batched = True
            w0 = w0[0]
        else:
            seed = (seed,)
        if self.cpu_tree:
            t0, w0, t1 = t0.detach().cpu(), w0.detach().cpu(), t1.detach().cpu()
        self.trees = tuple(torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed)

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        device, dtype = t0.device, t0.dtype
        if self.cpu_tree:
            t0, t1 = t0.detach().cpu().float(), t1.detach().cpu().float()
        w = torch.stack([tree(t0, t1) for tree in self.trees]).to(device=device, dtype=dtype) * (self.sign * sign)
        return w if self.batched else w[0]

class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)"""
    if _is_const(model_sampling):
        # log((1 - t) / t) = log((1 - sigma) / sigma)
        return sigma.logit().neg()
    return sigma.log().neg()

def half_log_snr_to_sigma(half_log_snr, model_sampling):
    """Convert half-logSNR log(alpha_t / sigma_t) to sigma"""
    if _is_const(model_sampling):
        # 1 / (1 + exp(half_log_snr))
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()

def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR"""
    if len(sigmas) <= 1:
        return sigmas
    if _is_const(model_sampling):
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas

@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    if _is_const(model.inner_model.predictor):
        return sample_euler_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
    """Ancestral sampling with Euler method steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_euler_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """Ancestral sampling with Euler method steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2) ** 0.5
            # Euler method
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
    return x

@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x

def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    if _is_const(model.inner_model.predictor):
        return sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)

    """Ancestral sampling with DPM-Solver++(2S) second-order steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda lbda: (lbda.exp() + 1) ** -1
    lambda_fn = lambda sigma: ((1 - sigma) / sigma).log()

    # logged_x = x.unsqueeze(0)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2) ** 0.5
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            if sigmas[i] == 1.0:
                sigma_s = 0.9999
            else:
                t_i, t_down = lambda_fn(sigmas[i]), lambda_fn(sigma_down)
                r = 1 / 2
                h = t_down - t_i
                s = t_i + r * h
                sigma_s = sigma_fn(s)
            # sigma_s = sigmas[i+1]
            sigma_s_i_ratio = sigma_s / sigmas[i]
            u = sigma_s_i_ratio * x + (1 - sigma_s_i_ratio) * denoised
            D_i = model(u, sigma_s * s_in, **extra_args)
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * D_i
            # print("sigma_i", sigmas[i], "sigma_ip1", sigmas[i+1],"sigma_down", sigma_down, "sigma_down_i_ratio", sigma_down_i_ratio, "sigma_s_i_ratio", sigma_s_i_ratio, "renoise_coeff", renoise_coeff)
        # Noise addition
        if sigmas[i + 1] > 0 and eta > 0:
            x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        # logged_x = torch.cat((logged_x, x.unsqueeze(0)), dim=0)
    return x

@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic)"""
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            lambda_s_1 = lambda_s + r * h
            fac = 1 / (2 * r)

            sigma_s_1 = sigma_fn(lambda_s_1)

            alpha_s = sigmas[i] * lambda_s.exp()
            alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
            alpha_t = sigmas[i + 1] * lambda_t.exp()

            # Step 1
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_s_1.neg().exp(), eta)
            lambda_s_1_ = sd.log().neg()
            h_ = lambda_s_1_ - lambda_s
            x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (-h_).expm1() * denoised
            if eta > 0 and s_noise > 0:
                x_2 = x_2 + alpha_s_1 * noise_sampler(sigmas[i], sigma_s_1) * s_noise * su
            denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_t.neg().exp(), eta)
            lambda_t_ = sd.log().neg()
            h_ = lambda_t_ - lambda_s
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (-h_).expm1() * denoised_d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * su
    return x

@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, solver_type="midpoint"):
    """DPM-Solver++(2M) SDE"""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    old_denoised = None
    h, h_last = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """DPM-Solver++(3M) SDE"""

    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                # DPM-Solver++(3M) SDE
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                # DPM-Solver++(2M) SDE
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + (alpha_t * phi_2) * d

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_flow(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """DPM-Solver++(3M) SDE adapted for Flow Matching"""
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # Force cleaner handling of lambda for Flow
    def robust_sigma_to_log_snr(sigma):
        # Clamp sigma to avoid infs at 0 and 1
        sigma = sigma.clamp(min=1e-4, max=1.0 - 1e-4)
        return sigma.logit().neg() # log((1-sigma)/sigma)

    lambda_fn = robust_sigma_to_log_snr
    
    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            # DPM-Solver++ Logic
            lambda_s = lambda_fn(sigmas[i])
            lambda_t = lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()
            
            # Core DPM++ Update
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                # 3M
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                # 2M
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + (alpha_t * phi_2) * d

            # SDE Noise Injection
            if eta > 0 and s_noise > 0:
                 x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x = model.inner_model.predictor.noise_scaling(sigmas[i + 1], noise_sampler(sigmas[i], sigmas[i + 1]), x)
    return x

@torch.no_grad()
def sample_euler_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """Ancestral sampling with Euler method steps (CFG++)"""
    extra_args = {} if extra_args is None else extra_args
    extra_args['cond_scale'] /= 12.5 # Adjust for CFG++

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)

    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            alpha_s = sigmas[i] * lambda_fn(sigmas[i]).exp()
            alpha_t = sigmas[i + 1] * lambda_fn(sigmas[i + 1]).exp()
            d = to_d(x, sigmas[i], alpha_s * uncond_denoised)  # to noise

            # DDIM stochastic sampling
            sigma_down, sigma_up = get_ancestral_step(sigmas[i] / alpha_s, sigmas[i + 1] / alpha_t, eta=eta)
            sigma_down = alpha_t * sigma_down

            # Euler method
            x = alpha_t * denoised + sigma_down * d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_euler_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Euler method steps (CFG++)"""
    return sample_euler_ancestral_cfg_pp(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=0.0, s_noise=0.0, noise_sampler=None)

@torch.no_grad()
def sample_dpmpp_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    eta = 1.0
    s_noise = 1.0
    r = 0.5

    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None)) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    extra_args['cond_scale'] /= 12.5 # Adjust for CFG++

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        if sigmas[i + 1] == 0:
            d = to_d(x, sigmas[i], temp[0])
            x = denoised + d * sigmas[i + 1]
        else:
            t, t_next = _t_fn(sigmas[i]), _t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            sd, su = get_ancestral_step(_sigma_fn(t), _sigma_fn(s), eta)
            s_ = _t_fn(sd)
            x_2 = (_sigma_fn(s_) / _sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(_sigma_fn(t), _sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, _sigma_fn(s) * s_in, **extra_args)

            sd, su = get_ancestral_step(_sigma_fn(t), _sigma_fn(t_next), eta)
            denoised_d = (1 - fac) * temp[0] + fac * temp[0]
            x = denoised_2 + to_d(x, sigmas[i], denoised_d) * sd
            x = x + noise_sampler(_sigma_fn(t), _sigma_fn(t_next)) * s_noise * su
    return x

@torch.no_grad()
def sample_dpmpp_2m_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)"""
    extra_args = {} if extra_args is None else extra_args
    extra_args['cond_scale'] /= 12.5 # Adjust for CFG++
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()

    old_uncond_denoised = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, solver_type="midpoint"):
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError('solver_type must be "heun" or "midpoint"')

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    extra_args['cond_scale'] /= 12.5 # Adjust for CFG++
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * (x + (denoised - temp[0])) + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    extra_args['cond_scale'] /= 12.5 # Adjust for CFG++
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * (x + (denoised - temp[0])) + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

class ZigZagController:
    """
    Adaptive controller using 'Trajectory Rotation'.
    Includes Cooldown to allow DPM++ 3M to rebuild high-order history.
    """
    def __init__(self, start_sigma, end_sigma=0.8, spike_sensitivity=1.15, hard_force_every=None):
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.spike_sensitivity = spike_sensitivity
        self.hard_force_every = hard_force_every
        
        self.step_count = 0
        self.last_flat = None # Store flattened directly
        self.moving_avg_rotation = None
        self.ema_decay = 0.6

        self.cooldown_steps = 2 
        self.current_cooldown = 0

    def should_trigger(self, current_sigma, next_sigma, current_denoised):
        self.step_count += 1
        
        # Flatten once here to save compute
        curr_flat = current_denoised.flatten(1)

        # 1. Cooldown & Bounds Check
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            self.last_flat = curr_flat.detach().clone()
            return False, 0.0

        if not (self.end_sigma <= current_sigma <= self.start_sigma):
            self.last_flat = curr_flat.detach().clone()
            return False, 0.0
        
        # 2. Hard Force
        if self.hard_force_every and (self.step_count % self.hard_force_every == 0):
            self.last_flat = curr_flat.detach().clone()
            self.current_cooldown = self.cooldown_steps
            return True, 1.0

        should_zag = False
        severity = 0.0
        d_sigma = current_sigma - next_sigma

        if self.last_flat is not None:
            # Cosine Similarity on flattened tensors
            cos_sim = torch.nn.functional.cosine_similarity(self.last_flat, curr_flat, dim=1)
            avg_sim = cos_sim.mean().item()
            
            # Rotation metric
            raw_score = max(0.0, 1.0 - avg_sim)
            ref_d_sigma = max(d_sigma, 0.1)
            normalized_rotation = raw_score / ref_d_sigma

            if self.moving_avg_rotation is None:
                self.moving_avg_rotation = normalized_rotation
            else:
                threshold = max(self.moving_avg_rotation * self.spike_sensitivity, 0.005)

                if normalized_rotation > threshold:
                    overshoot = normalized_rotation - threshold
                    severity = min(max(overshoot / threshold, 0.0), 1.0)
                    should_zag = True
                    self.current_cooldown = self.cooldown_steps
                    # Optional: Print trigger for debug
                    # print(f"⚡ ZigZag: Rot={normalized_rotation:.4f} > Thr={threshold:.4f}")

                # Update EMA
                self.moving_avg_rotation = (self.ema_decay * self.moving_avg_rotation) + \
                                           ((1 - self.ema_decay) * normalized_rotation)

        self.last_flat = curr_flat.detach().clone()
        return should_zag, severity
    
    def update_history(self, final_denoised):
        self.last_flat = final_denoised.flatten(1).detach().clone()

def _dpm_solver_step(x, t, s, denoised, denoised_1, denoised_2, h_1, h_2, eta, noise_sampler, s_noise, 
                     uncond_denoised, sigma_t, sigma_s):
    h = s - t
    h_eta = h * (eta + 1)

    # Standard DPM-Solver++ First Order
    x = torch.exp(-h_eta) * (x + (denoised - uncond_denoised)) + (-h_eta).expm1().neg() * denoised

    # High-Order Corrections (DPM++ 2M / 3M)
    if h_2 is not None:
        # 3rd Order (3M)
        r0 = h_1 / h
        r1 = h_2 / h
        d1_0 = (denoised - denoised_1) / r0
        d1_1 = (denoised_1 - denoised_2) / r1
        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
        d2 = (d1_0 - d1_1) / (r0 + r1)
        phi_2 = h_eta.neg().expm1() / h_eta + 1
        phi_3 = phi_2 / h_eta - 0.5
        x = x + phi_2 * d1 - phi_3 * d2
        
    elif h_1 is not None:
        # 2nd Order (2M)
        r = h_1 / h
        d = (denoised - denoised_1) / r
        phi_2 = h_eta.neg().expm1() / h_eta + 1
        x = x + phi_2 * d

    # SDE Noise Injection
    if eta > 0:
        x = x + noise_sampler(sigma_t, sigma_s) * sigma_s * (-2 * h * eta).expm1().neg().sqrt() * s_noise

    return x

def _zigzag_handler(model, x, sigma_t, sigma_s, t, s, extra_args, 
                    denoised_1, denoised_2, h_1, h_2, 
                    eta, noise_sampler, s_noise, gamma_scale, 
                    temp_storage,
                    current_denoised_probe, current_uncond_probe):
    s_in = x.new_ones([x.shape[0]])

    # Local extra_args copy for isolation
    local_extra_args = extra_args.copy() if extra_args is not None else {}

    # Define a local hook that writes to the passed temp_storage list
    def _local_post_cfg(args):
        temp_storage[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = local_extra_args.get("model_options", {}).copy()
    model_options["sampler_post_cfg_function"] = [_local_post_cfg]
    local_extra_args["model_options"] = model_options

    # Flag for internal use (optional, depending on model wrapper)
    local_extra_args["__zigzag_internal"] = True

    # --- 1. PROBE (Zig) - Forward ---
    # We use the probe values already calculated in the main loop
    denoised_probe = current_denoised_probe
    uncond_probe = current_uncond_probe

    # Step forward to the "bad" spot using current history
    intermediate_x = _dpm_solver_step(
        x, t, s, 
        denoised_probe, denoised_1, denoised_2, h_1, h_2, 
        0, noise_sampler, s_noise, 
        uncond_denoised=uncond_probe, sigma_t=sigma_t, sigma_s=sigma_s
    )

    # --- 2. REFLECTION (Zag) - Backward ---
    # Invert from s -> t (Backtracking)
    extra_args_low = local_extra_args.copy()
    if 'cond_scale' in extra_args_low and gamma_scale > 0:
        extra_args_low['cond_scale'] *= gamma_scale

    denoised_invert = model(intermediate_x, sigma_s * s_in, **extra_args_low)
    uncond_invert = temp_storage[0] 

    # Note: Invert swaps sigmas (s -> t)
    refined_x = _dpm_solver_step(
        intermediate_x, s, t, 
        denoised_invert, None, None, None, None, # No history for inversion
        0, noise_sampler, s_noise, 
        uncond_denoised=uncond_invert, sigma_t=sigma_s, sigma_s=sigma_t
    )

    # --- 3. COMMIT (Zig) - Forward ---
    # Step forward again with the corrected trajectory
    x_for_commit = refined_x 
    denoised_final = model(x_for_commit, sigma_t * s_in, **local_extra_args)
    uncond_final = temp_storage[0]

    # Use DPM Solver to move to next step, but WITHOUT history (restart trajectory)
    # We do NOT use denoised_1/h_1 here because we have "moved" the latent space
    x_next = _dpm_solver_step(
        x_for_commit, t, s, 
        denoised_final, None, None, None, None, 
        eta, noise_sampler, s_noise, 
        uncond_denoised=uncond_final, sigma_t=sigma_t, sigma_s=sigma_s
    )

    return x_next, denoised_final

@torch.no_grad()
def sample_dpmpp_3m_sde_cfgpp_ctrlz(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, gamma_scale=0.5):
    """
    Robust Hybrid DPM-Solver++(3M) SDE with ZigZag Sampling.
    """
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    extra_args['cfgpp'] = True # Indicate CFG++ for cond_scale adaptation
    extra_args['cond_scale'] /= 12.5
    s_in = x.new_ones([x.shape[0]])

    # Initialize Controller
    controller = ZigZagController(start_sigma=sigmas.max().item())
    
    # DPM++ History
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    # --- SETUP POST-CFG HOOK ---
    # CFG++ / Uncond storage
    temp_storage = [None] # Use a list to pass by reference
    def post_cfg_function(args):
        temp_storage[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    # Assuming set_model_options_post_cfg_function is available in scope
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_t, sigma_s = sigmas[i], sigmas[i + 1]
        
        # Calculate t, s for DPM solver
        # Handle sigma_s=0 (last step)
        if sigma_s == 0:
            # For the very last step, we usually just want to denoise one last time or return
            # But standard DPM loop logic requires t,s.
            # We'll rely on the sigma_s=0 check later.
            t, s = -sigma_t.log(), -100.0 # arbitrary low val
        else:
            t, s = -sigma_t.log(), -sigma_s.log()

        # 1. Main Model Call
        denoised = model(x, sigma_t * s_in, **extra_args)
        uncond_denoised = temp_storage[0] # Retrieved via hook

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_t, 'sigma_hat': sigma_t, 'denoised': denoised})

        # 2. Solver / ZigZag Logic
        if sigma_s == 0:
            x = denoised
            break # Exit loop at end
        
        # Check ZigZag Trigger
        # Pass raw values to controller
        is_zag, severity = controller.should_trigger(sigma_t.item(), sigma_s.item(), denoised)
        dynamic_gamma = gamma_scale * (1.0 - (0.5 * severity))
        
        if is_zag:
            # Perform ZigZag
            x, denoised_final = _zigzag_handler(
                model, x, sigma_t, sigma_s, t, s, extra_args,
                denoised_1, denoised_2, h_1, h_2,
                eta, noise_sampler, s_noise, dynamic_gamma,
                temp_storage,
                denoised, uncond_denoised
            )
            
            #Smart History Reset (2nd Order Continuation)
            h_1 = s - t
            h_2 = None
            denoised_1 = denoised_final
            denoised_2 = None  
            controller.update_history(denoised_final)
            
        else:
            # Standard DPM++ 3M Step
            x = _dpm_solver_step(
                x, t, s, 
                denoised, denoised_1, denoised_2, h_1, h_2, 
                eta, noise_sampler, s_noise, 
                uncond_denoised=uncond_denoised, sigma_t=sigma_t, sigma_s=sigma_s
            )
            # Update history
            h_2, h_1 = h_1, s - t
            denoised_2, denoised_1 = denoised_1, denoised
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_flow_cfgpp_ctrlz(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, gamma_scale=0.5):
    """
    Robust Hybrid DPM-Solver++(3M) SDE with ZigZag Sampling, adapted for Flow Matching.
    """
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Initialize Controller
    controller = ZigZagController(start_sigma=sigmas.max().item())
    
    # DPM++ History
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    # --- SETUP POST-CFG HOOK ---
    temp_storage = [None] 
    def post_cfg_function(args):
        temp_storage[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    # Force cleaner handling of lambda for Flow
    def robust_sigma_to_log_snr(sigma):
        sigma = sigma.clamp(min=1e-4, max=1.0 - 1e-4)
        return sigma.logit().neg() 

    lambda_fn = robust_sigma_to_log_snr

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_t, sigma_s = sigmas[i], sigmas[i + 1]
        
        # Robust t, s calculation
        if sigma_s == 0:
            t = lambda_fn(sigma_t)
            s = -100.0 
        else:
            t = lambda_fn(sigma_t)
            s = lambda_fn(sigma_s)

        # 1. Main Model Call
        denoised = model(x, sigma_t * s_in, **extra_args)
        uncond_denoised = temp_storage[0] 

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_t, 'sigma_hat': sigma_t, 'denoised': denoised})

        # 2. Solver / ZigZag Logic
        if sigma_s == 0:
            # Last step Euler
            d = to_d(x, sigma_t, uncond_denoised)
            x = denoised 
            break 
        
        is_zag, severity = controller.should_trigger(sigma_t.item(), sigma_s.item(), denoised)
        dynamic_gamma = gamma_scale * (1.0 - (0.5 * severity))
        
        if is_zag:
            # Perform ZigZag
            x, denoised_final = _zigzag_handler(
                model, x, sigma_t, sigma_s, t, s, extra_args,
                denoised_1, denoised_2, h_1, h_2,
                eta, noise_sampler, s_noise, dynamic_gamma,
                temp_storage,
                denoised, uncond_denoised
            )
            
            h_1 = s - t
            h_2 = None
            denoised_1 = denoised_final
            denoised_2 = None  
            controller.update_history(denoised_final)
            
        else:
            # Standard DPM++ 3M Step
            x = _dpm_solver_step(
                x, t, s, 
                denoised, denoised_1, denoised_2, h_1, h_2, 
                eta, noise_sampler, s_noise, 
                uncond_denoised=uncond_denoised, sigma_t=sigma_t, sigma_s=sigma_s
            )
            h_2, h_1 = h_1, s - t
            denoised_2, denoised_1 = denoised_1, denoised
    return x

def _dpm_solver_step_flow(x, t, s, denoised, denoised_1, denoised_2, h_1, h_2, eta, noise_sampler, s_noise, 
                          uncond_denoised, sigma_t, sigma_s, alpha_t):
    h = s - t
    h_eta = h * (eta + 1)

    # Flow Matching specific update (matches sample_dpmpp_3m_sde_flow)
    # x scale factor: sigma_s / sigma_t
    x_scaled = (sigma_s / sigma_t) * (-h * eta).exp() * x
    
    # Denoised contribution
    # Note: We use (denoised - uncond_denoised) if CFG++ is active, but here passed as 'denoised' logic
    # In CFG++, 'denoised' is usually the final result, and 'uncond_denoised' is used for the mix.
    # The standard function used: x = exp(...) * (x + (denoised - uncond))
    # But Flow version uses: x = ... + alpha_t * (...) * denoised
    
    # We must match the working sampler's logic EXACTLY.
    # Working: x = sigmas[i+1]/sigmas[i] * ... * x + alpha_t * ... * denoised
    
    # Apply CFG++ correction to 'x' if needed? 
    # Standard CFG++: x_next = exp(-h)(x + (denoised - uncond)) ...
    # This implies 'x' is effectively shifted by the cfg term.
    
    # For Flow with CFG++:
    # We likely want to treat 'denoised' as the guided result.
    # The standard _dpm_solver_step logic for CFG++ modifies 'x' directly.
    # Let's stick to the WORKING sampler's update equation, but using the 'denoised' passed in.
    # If CFG++ is active, 'denoised' is already the guided result.
    # The `x + (denoised - uncond)` part in standard CFG++ is tricky. It's an Euler-like correction.
    
    # Let's assume for Flow Matching we just trust the 'denoised' output (which includes CFG)
    # and the standard DPM update.
    
    phi_1 = (-h_eta).expm1().neg()
    
    x = x_scaled + alpha_t * phi_1 * denoised

    # High-Order Corrections
    if h_2 is not None:
        r0 = h_1 / h
        r1 = h_2 / h
        d1_0 = (denoised - denoised_1) / r0
        d1_1 = (denoised_1 - denoised_2) / r1
        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
        d2 = (d1_0 - d1_1) / (r0 + r1)
        phi_2 = h_eta.neg().expm1() / h_eta + 1
        phi_3 = phi_2 / h_eta - 0.5
        x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
        
    elif h_1 is not None:
        r = h_1 / h
        d = (denoised - denoised_1) / r
        phi_2 = h_eta.neg().expm1() / h_eta + 1
        x = x + (alpha_t * phi_2) * d

    # SDE Noise Injection
    if eta > 0:
        x = x + noise_sampler(sigma_t, sigma_s) * sigma_s * (-2 * h * eta).expm1().neg().sqrt() * s_noise

    return x

def _zigzag_handler_flow(model, x, sigma_t, sigma_s, t, s, extra_args, 
                         denoised_1, denoised_2, h_1, h_2, 
                         eta, noise_sampler, s_noise, gamma_scale, 
                         temp_storage,
                         current_denoised_probe, current_uncond_probe,
                         lambda_fn):
    s_in = x.new_ones([x.shape[0]])
    local_extra_args = extra_args.copy() if extra_args is not None else {}

    def _local_post_cfg(args):
        temp_storage[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = local_extra_args.get("model_options", {}).copy()
    model_options["sampler_post_cfg_function"] = [_local_post_cfg]
    local_extra_args["model_options"] = model_options
    local_extra_args["__zigzag_internal"] = True

    # Pre-calc alphas
    alpha_t = sigma_t * lambda_fn(sigma_t).exp()
    alpha_s = sigma_s * lambda_fn(sigma_s).exp()

    # 1. PROBE (Zig)
    intermediate_x = _dpm_solver_step_flow(
        x, t, s, 
        current_denoised_probe, denoised_1, denoised_2, h_1, h_2, 
        0, noise_sampler, s_noise, 
        uncond_denoised=current_uncond_probe, sigma_t=sigma_t, sigma_s=sigma_s, alpha_t=alpha_s
    )

    # 2. REFLECTION (Zag) - Backward (s -> t)
    extra_args_low = local_extra_args.copy()
    if 'cond_scale' in extra_args_low and gamma_scale > 0:
        extra_args_low['cond_scale'] *= gamma_scale

    denoised_invert = model(intermediate_x, sigma_s * s_in, **extra_args_low)
    # uncond_invert = temp_storage[0]

    refined_x = _dpm_solver_step_flow(
        intermediate_x, s, t, 
        denoised_invert, None, None, None, None, 
        0, noise_sampler, s_noise, 
        uncond_denoised=None, sigma_t=sigma_s, sigma_s=sigma_t, alpha_t=alpha_t
    )

    # 3. COMMIT (Zig) - Forward
    x_for_commit = refined_x 
    denoised_final = model(x_for_commit, sigma_t * s_in, **local_extra_args)
    
    x_next = _dpm_solver_step_flow(
        x_for_commit, t, s, 
        denoised_final, None, None, None, None, 
        eta, noise_sampler, s_noise, 
        uncond_denoised=None, sigma_t=sigma_t, sigma_s=sigma_s, alpha_t=alpha_s
    )

    return x_next, denoised_final

@torch.no_grad()
def sample_dpmpp_3m_sde_flow_cfgpp_ctrlz(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, gamma_scale=0.5):
    """
    Robust Hybrid DPM-Solver++(3M) SDE with ZigZag Sampling, adapted for Flow Matching.
    """
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    extra_args['cfgpp'] = True 
    # extra_args['cond_scale'] /= 12.5 # Disabled for Flow Matching
    s_in = x.new_ones([x.shape[0]])

    controller = ZigZagController(start_sigma=sigmas.max().item())
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    temp_storage = [None] 
    def post_cfg_function(args):
        temp_storage[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    def robust_sigma_to_log_snr(sigma):
        sigma = sigma.clamp(min=1e-4, max=1.0 - 1e-4)
        return sigma.logit().neg() 

    lambda_fn = robust_sigma_to_log_snr

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_t, sigma_s = sigmas[i], sigmas[i + 1]
        
        if sigma_s == 0:
            t = lambda_fn(sigma_t)
            s = -100.0 
        else:
            t = lambda_fn(sigma_t)
            s = lambda_fn(sigma_s)

        denoised = model(x, sigma_t * s_in, **extra_args)
        uncond_denoised = temp_storage[0] 

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_t, 'sigma_hat': sigma_t, 'denoised': denoised})

        if sigma_s == 0:
            d = to_d(x, sigma_t, uncond_denoised)
            x = denoised 
            break 
        
        alpha_t = sigma_s * s.exp() # alpha for the *next* step (destination)

        is_zag, severity = controller.should_trigger(sigma_t.item(), sigma_s.item(), denoised)
        dynamic_gamma = gamma_scale * (1.0 - (0.5 * severity))
        
        if is_zag:
            x, denoised_final = _zigzag_handler_flow(
                model, x, sigma_t, sigma_s, t, s, extra_args,
                denoised_1, denoised_2, h_1, h_2,
                eta, noise_sampler, s_noise, dynamic_gamma,
                temp_storage,
                denoised, uncond_denoised, lambda_fn
            )
            h_1 = s - t
            h_2 = None
            denoised_1 = denoised_final
            denoised_2 = None  
            controller.update_history(denoised_final)
        else:
            x = _dpm_solver_step_flow(
                x, t, s, 
                denoised, denoised_1, denoised_2, h_1, h_2, 
                eta, noise_sampler, s_noise, 
                uncond_denoised=uncond_denoised, sigma_t=sigma_t, sigma_s=sigma_s, alpha_t=alpha_t
            )
            h_2, h_1 = h_1, s - t
            denoised_2, denoised_1 = denoised_1, denoised
    return x

@torch.no_grad()
def res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, eta=1.0, cfg_pp=False):
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

    old_sigma_down = None
    old_denoised = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0 or old_denoised is None:
            # Euler method
            if cfg_pp:
                d = to_d(x, sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                d = to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
        else:
            # Second order multistep method in https://arxiv.org/pdf/2308.02157
            t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            if cfg_pp:
                x = x + (denoised - uncond_denoised)
                x = sigma_fn(h) * x + h * (b1 * uncond_denoised + b2 * old_denoised)
            else:
                x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)

        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if cfg_pp:
            old_denoised = uncond_denoised
        else:
            old_denoised = denoised
        old_sigma_down = sigma_down
    return x

@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0.0, cfg_pp=False)

@torch.no_grad()
def sample_Kohaku_LoNyu_Yog(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=None, s_tmin=None, s_tmax=float("inf"), s_noise=None, noise_sampler=None, eta=None):
    s_churn = 0.0 if s_churn is None else s_churn
    s_tmin = 0.0 if s_tmin is None else s_tmin
    s_noise = 1.0 if s_noise is None else s_noise
    eta = 1.0 if eta is None else eta

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigma_down - sigmas[i]
        if i <= (len(sigmas) - 1) / 2:
            x2 = -x
            denoised2 = model(x2, sigma_hat * s_in, **extra_args)
            d2 = to_d(x2, sigma_hat, denoised2)
            x3 = x + ((d + d2) / 2) * dt
            denoised3 = model(x3, sigma_hat * s_in, **extra_args)
            d3 = to_d(x3, sigma_hat, denoised3)
            real_d = (d + d3) / 2
            x = x + real_d * dt
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        else:
            x = x + d * dt
    return x


@torch.no_grad()
def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, noise_scaler=None, max_stage=3):
    """
    Extended Reverse-Time SDE solver
    arXiv: https://arxiv.org/abs/2309.06169
    reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    def default_er_sde_noise_scaler(x):
        return x * ((x**0.3).exp() + 10.0)

    noise_scaler = default_er_sde_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    model_sampling = model.inner_model.predictor
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt**2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (er_lambda_t**2 - er_lambda_s**2 * r**2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised

    return x

