import dataclasses
from math import atan, pi, log, exp
from typing import Callable

import k_diffusion
import numpy as np
import torch
from modules import shared
from scipy import stats


def to_d(x: torch.Tensor, sigma: float, denoised: torch.Tensor):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / sigma


k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: Callable

    default_rho: float = -1.0
    need_inner_model: bool = False
    aliases: list[str] = None


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [inner_model.t_to_sigma(ts) for ts in torch.linspace(start, end, n + 1)[:-1]]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device):
    """
    Continuous 'Align Your Steps' (AYS) Schedule.
    
    Implements the Log-Logistic distribution found in Nvidia's AYS paper.
    Includes a 'Smart Cap' to prevent step dilution at high sigma values.
    """
    
    # 1. Environment & Model Context
    # Gracefully handle missing 'shared' object for non-A1111/Forge environments
    try:
        is_sdxl = getattr(shared.sd_model, 'is_sdxl', False)
        is_anima = getattr(shared.sd_model, 'is_anima', False)
    except (ImportError, NameError, AttributeError):
        is_sdxl = False
        is_anima = False

    # 2. AYS Parameters (Log-Logistic)
    # Derived from Nvidia's optimized discrete lists

    if is_anima:
        use_log_logistic = False
        optimal_start = 80.0
        shift = 3.0
    elif is_sdxl:
        use_log_logistic = True
        optimal_start = 14.61
        loc = 0.0699
        scale = 1.4059
    else:
        use_log_logistic = True
        optimal_start = 14.61
        loc = 1.3114
        scale = 1.6607

    # 3. Smart Cap: Prevent Step Dilution
    # Standard UIs often request huge sigmas (e.g., 120+). AYS is ineffective there.
    # We cap the start to the optimal AYS range so steps aren't wasted on 
    # "dead" high-noise zones, ensuring maximum density where it counts.
    if sigma_max > optimal_start:
        sigma_max = optimal_start

    if use_log_logistic:
        # 4a. Solve for Exact Start Point (t_max)
        # Map sigma_max to its quantile 't' on the distribution curve
        def sigma_to_t(sigma, loc, scale):
            sigma = max(sigma, 1e-5)
            y = (log(sigma) - loc) / scale
            return 1 / (1 + exp(-y))

        t_max = sigma_to_t(sigma_max, loc, scale)
        t_min = 0.0  # Target the asymptote for natural ramp-down

        # 5a. Generate Schedule
        t = torch.linspace(t_max, t_min, n + 1, device=device)
        
        # Clamp for numerical stability (avoid log(0))
        t = t.clamp(min=1e-5, max=1-1e-5)
        
        # Inverse CDF of Log-Logistic Distribution
        log_sigmas = loc + scale * torch.log(t / (1 - t))
        sigmas = torch.exp(log_sigmas)

    else:
        # 4b. Create Linear Steps (0.0 to 1.0)
        t = torch.linspace(1.0, 0.0, n + 1, device=device)

        # 5b. Apply Time-Shift (The "AYS" for Flow)
        t_shifted = (t * shift) / (1 + (shift - 1) * t)

        # 6b. Map to Sigma Range
        # Flow models usually map t linear to sigma
        sigmas = t_shifted * (sigma_max - sigma_min) + sigma_min

    # Force Exact Boundaries
    sigmas[0] = sigma_max
    sigmas[-1] = 0.0

    return sigmas


def linear_quadratic(n, sigma_min, sigma_max, device, *, threshold_noise=0.025):
    if n == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        linear_steps = n // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * n
        quadratic_steps = n - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, n)]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule).to(device) * sigma_max


def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    """
    Beta scheduler
    Based on "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)
    """
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta

    total_timesteps = len(inner_model.sigmas) - 1
    ts = 1 - np.linspace(0, 1, n, endpoint=False)
    ts = np.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(inner_model.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    unet = inner_model.inner_model.forge_objects.unet
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    sigmas = unet.model.predictor.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)


def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2

    srange = smax - smin
    sscale = start - end

    sigmas = [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in range(steps)]

    return sigmas


def bong_tangent_scheduler(n, sigma_min, sigma_max, device, *, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    """https://github.com/ClownsharkBatwing/RES4LYF/blob/main/sigmas.py#L4076"""
    n += 2

    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    pivot_1 = int(n * pivot_1)
    pivot_2 = int(n * pivot_2)

    slope_1 = slope_1 / (n / 40)
    slope_2 = slope_2 / (n / 40)

    stage_2_len = n - midpoint
    stage_1_len = n - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)

    tan_sigmas_1 = tan_sigmas_1[:-1]
    if pad:
        tan_sigmas_2 = tan_sigmas_2 + [0]

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas.to(device)

def phi_scheduler(n, sigma_min, sigma_max, device):
    """
    The 'Golden Warp' Scheduler. 
    Warps a log-linear distribution using Phi to balance high-noise exploration 
    and low-noise refinement naturally.
    """
    phi = 1.618033988749895
    
    # Standard linear steps 0 -> 1
    t = torch.linspace(0, 1, n, device=device)
    
    # Warp function: 1 - (1 - x)^phi
    # This creates a convex curve similar to Karras but derived from the Golden Ratio.
    # It drops from high sigma slightly faster than linear, spending 'Phi' more time
    # in the structure-forming phase.
    t = 1 - (1 - t) ** phi
    
    # Log-Linear Interpolation
    log_min = log(sigma_min)
    log_max = log(sigma_max)
    
    log_sigmas = log_max + t * (log_min - log_max)
    sigmas = torch.exp(log_sigmas)
    
    # Append zero for the final step
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
    
    return sigmas

def flow_match_euler_discrete_scheduler(n, sigma_min, sigma_max, inner_model, device):
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    unet = inner_model.inner_model.forge_objects.unet

    config = {
        "num_train_timesteps": 1000,
        "shift": getattr(unet.model.predictor, "shift", 1.0),
        "use_dynamic_shifting": shared.opts.use_dynamic_shifting,
        "invert_sigmas": shared.opts.invert_sigmas,
        "shift_terminal": None,
        "use_karras_sigmas": shared.opts.use_karras_sigmas,
        "use_exponential_sigmas": shared.opts.use_exponential_sigmas,
        "use_beta_sigmas": shared.opts.use_beta_sigmas,
        "time_shift_type": "exponential",
        "stochastic_sampling": shared.opts.stochastic_sampling,
    }

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    scheduler.set_timesteps(n, device=device, mu=0.0)
    sigmas = scheduler.sigmas

    return torch.FloatTensor(sigmas).to(device)


schedulers = [
    Scheduler("automatic", "Automatic", None),
    Scheduler("karras", "Karras", k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler("exponential", "Exponential", k_diffusion.sampling.get_sigmas_exponential),
    Scheduler("polyexponential", "Polyexponential", k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler("phi", "Phi", phi_scheduler),
    Scheduler("normal", "Normal", normal_scheduler, need_inner_model=True),
    Scheduler("simple", "Simple", simple_scheduler, need_inner_model=True),
    Scheduler("uniform", "Uniform", uniform, need_inner_model=True),
    Scheduler("sgm_uniform", "SGM Uniform", sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler("linear_quadratic", "Linear Quadratic", linear_quadratic),
    Scheduler("kl_optimal", "KL Optimal", kl_optimal),
    Scheduler("ddim", "DDIM", ddim_scheduler, need_inner_model=True),
    Scheduler("align_your_steps", "Align Your Steps", get_align_your_steps_sigmas),
    Scheduler("beta", "Beta", beta_scheduler, need_inner_model=True),
    Scheduler("turbo", "Turbo", turbo_scheduler, need_inner_model=True),
    Scheduler("bong_tangent", "Bong Tangent", bong_tangent_scheduler),
    Scheduler("flow_match", "FlowMatchEulerDiscrete", flow_match_euler_discrete_scheduler, need_inner_model=True),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
