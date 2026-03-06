import dataclasses
from math import atan, exp, log, pi
from typing import Callable, Optional

import k_diffusion
import numpy as np
import torch
from scipy import stats

from modules import shared


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
    need_width_height: bool = False
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

def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device, width=1024, height=1024):
    """
    Align Your Steps scheduler, based on "Align Your Steps: Optimal Noise Schedules for Diffusion Models" [arXiv:2406.16157] (Zhao et al., 2024).
    
    Key features:
    - Log-Logistic Distribution: For SDXL and SD1.5, uses a log-logistic distribution in t-space to optimally space noise levels based on model-specific parameters (loc, scale).
    - Dynamic Shift Tuning: For Anima models, dynamically adjusts the shift based on resolution and step count to tailor the noise schedule.
    - Smart Cap: Limits maximum sigma to an optimal value (14.61 for SDXL/SD1.5, 80.0 for Anima) to prevent training instability.
    - Resolution-Aware: Scales parameters based on image resolution for better adaptability across different input sizes.
    """
    # Use the longer dimension as the primary resolution reference
    resolution = max(width, height)
    try:
        is_sdxl = getattr(shared.sd_model, 'is_sdxl', False)
        is_anima = getattr(shared.sd_model, 'is_anima', False)
    except (ImportError, NameError, AttributeError):
        is_sdxl = False
        is_anima = False

def _get_ays_diffusion_sigmas(
    n: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    is_sdxl: bool,
    apply_beta: bool = False,
) -> torch.Tensor:
    """SD1.5 / SDXL branch – pure parametric log-logistic (original AYS paper spirit)."""
    loc = 0.0699 if is_sdxl else 1.3114
    scale = 1.4059 if is_sdxl else 1.6607

    optimal_start = 14.61
    if sigma_max > optimal_start:
        sigma_max = optimal_start

    # Build fine table for beta (or just n)
    m = n * 3 if apply_beta and n < 50 else n

    t = torch.linspace(1.0, 0.0, m + 1, device=device)

    def sigma_to_t(sigma: float, loc: float, scale: float) -> float:
        sigma = max(sigma, 1e-5)
        y = (log(sigma) - loc) / scale
        return 1.0 / (1.0 + exp(-y))

    t_max = sigma_to_t(sigma_max, loc, scale)
    t_min = sigma_to_t(sigma_min, loc, scale)

    t = t * (t_max - t_min) + t_min
    t = t.clamp(min=1e-5, max=1.0 - 1e-5)

    log_sigmas = loc + scale * torch.log(t / (1.0 - t))
    sigmas = torch.exp(log_sigmas)

    return sigmas


def _get_ays_flow_sigmas(
    n: int,
    width: int,
    height: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    inner_model: Optional[object] = None,
) -> torch.Tensor:
    """Anima / Rectified-Flow branch.
    Uses the user's Shift setting directly from the predictor — no resolution
    scaling, which would silently override the user's explicit choice."""

    # Flow models must terminate exactly at 0.0.
    flow_sigma_max = min(1.0, float(sigma_max))

    # Read shift directly from the predictor (set from the UI 'Shift' / 'Distilled CFG' slider).
    if inner_model is None:
        shift = 3.0
    else:
        unet = inner_model.inner_model.forge_objects.unet
        shift = getattr(unet.model.predictor, "shift", 3.0)

    # Linear timesteps in [sigma_max, 0], then apply the flow-matching shift warp.
    timesteps = torch.linspace(flow_sigma_max, 0.0, n + 1, dtype=torch.float32, device=device)
    sigmas = (timesteps * shift) / (1.0 + (shift - 1.0) * timesteps)

    # Enforce exact boundaries.
    sigmas[0] = flow_sigma_max
    sigmas[-1] = 0.0

    return sigmas


def get_align_your_steps_sigmas(
    n: int,
    width: int,
    height: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    apply_beta: bool = False,
    inner_model: Optional[object] = None,
) -> torch.Tensor:
    """
    Align Your Steps scheduler (refactored 2026).
    Dispatcher + cleaned beta layer.
    """
    try:
        is_sdxl = getattr(shared.sd_model, "is_sdxl", False)
        is_anima = getattr(shared.sd_model, "is_anima", False)
        is_flow = getattr(shared.sd_model, "is_flow", False)
    except (ImportError, NameError, AttributeError):
        is_sdxl = False
        is_anima = False
        is_flow = False

    is_flow_model = is_anima or is_flow

    # Decide table size for beta (finer table → better remapping)
    m = n * 3 if apply_beta and n < 50 else n

    if is_flow_model:
        # Flow branch always builds exactly m+1 sigmas (beta will remap later)
        sigmas = _get_ays_flow_sigmas(m, width, height, sigma_min, sigma_max, device, inner_model)
    else:
        sigmas = _get_ays_diffusion_sigmas(m, sigma_min, sigma_max, device, is_sdxl, apply_beta=apply_beta)

    # ====================== BETA REMAPPING LAYER ======================
    if apply_beta:
        alpha = shared.opts.beta_dist_alpha
        beta_param = shared.opts.beta_dist_beta  # renamed to avoid shadowing

        linear_timesteps = np.linspace(0, 1, n + 1)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta_probabilities = stats.beta.ppf(linear_timesteps, alpha, beta_param)

        beta_probabilities = np.nan_to_num(beta_probabilities, nan=0.0, posinf=1.0, neginf=0.0)

        table_indices = np.rint(beta_probabilities * m).astype(int)
        table_indices = np.clip(table_indices, 0, m)

        # Deduplicate
        valid_indices = []
        for idx in table_indices:
            if not valid_indices or idx != valid_indices[-1]:
                valid_indices.append(idx)

        valid_indices = np.array(valid_indices, dtype=float)
        original_timeline = np.linspace(0, 1, len(valid_indices))
        target_timeline = np.linspace(0, 1, n + 1)
        interpolated_indices = np.interp(target_timeline, original_timeline, valid_indices)

        # Vectorised lerp (cleaner than old loop)
        idx_floor = interpolated_indices.astype(int)
        idx_ceil = np.minimum(idx_floor + 1, m)
        weight = torch.from_numpy(interpolated_indices - idx_floor).to(device).to(torch.float32)

        sigmas_floor = sigmas[idx_floor]
        sigmas_ceil = sigmas[idx_ceil]

        sigmas = (sigmas_floor * (1 - weight) + sigmas_ceil * weight).to(device)

    # Final boundary enforcement
    sigmas[-1] = 0.0

    return sigmas


def get_align_your_steps_with_beta_selection_sigmas(n, width, height, sigma_min, sigma_max, device, inner_model=None):
    """
    Hybrid Scheduler: Align Your Steps with Beta Distribution Selection
    
    Combines the best of both approaches by applying the Beta Distribution
    to the linear time variable *before* mapping it through the AYS curve.
    
    This allows for dynamic emphasis on high-noise structure formation or low-noise refinement
    based on alpha/beta parameters, while maintaining AYS's mathematically optimized noise spacing,
    without "leapfrogging" or starving the extreme edges.
    
    Parameters are controlled via shared.opts.beta_dist_alpha and shared.opts.beta_dist_beta
    (typically 0.6/0.6 for balanced, 0.4/0.6 for more structure emphasis)
    """
    return get_align_your_steps_sigmas(n, width, height, sigma_min, sigma_max, device, apply_beta=True, inner_model=inner_model)


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
    
    # Warp function: t -> t^Phi
    # This creates a convex curve similar to Karras but derived from the Golden Ratio.
    # It drops from high sigma slightly faster than linear, spending 'Phi' more time
    # in the structure-forming phase.
    t = t ** phi
    
    # Log-Linear Interpolation
    log_min = log(max(sigma_min, 1e-5))
    log_max = log(max(sigma_max, 1e-5))
    
    log_sigmas = log_max + t * (log_min - log_max)
    sigmas = torch.exp(log_sigmas)
    
    # Append zero for the final step
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
    
    return sigmas


def flow_match_euler_discrete_scheduler(n, width, height, sigma_min, sigma_max, inner_model, device, ):
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    unet = inner_model.inner_model.forge_objects.unet

    use_dynamic_shifting = getattr(shared.opts, "use_dynamic_shifting", False)

    config = {
        "num_train_timesteps": 1000,
        "shift": getattr(unet.model.predictor, "shift", 1.0),
        "use_dynamic_shifting": use_dynamic_shifting,
        "invert_sigmas": getattr(shared.opts, "invert_sigmas", False),
        "shift_terminal": None,
        "use_karras_sigmas": getattr(shared.opts, "use_karras_sigmas", False),
        "use_exponential_sigmas": getattr(shared.opts, "use_exponential_sigmas", False),
        "use_beta_sigmas": getattr(shared.opts, "use_beta_sigmas", False),
        "time_shift_type": "exponential",
        "stochastic_sampling": getattr(shared.opts, "stochastic_sampling", False),
    }

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    
    mu = 0.0
    if use_dynamic_shifting:
        seq_len = width * height / (16 * 16)
        mu = compute_empirical_mu(round(seq_len), n)
        
    scheduler.set_timesteps(n, device=device, mu=mu)
    sigmas = scheduler.sigmas

    return torch.FloatTensor(sigmas).to(device)


def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float) -> float:
    return exp(mu) / (exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps


def flux2_scheduler(n: int, width: int, height: int, sigma_min, sigma_max, device):
    # https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_flux.py
    seq_len = width * height / (16 * 16)
    sigmas = get_schedule(n, round(seq_len))
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
    Scheduler("align_your_steps", "Align Your Steps", get_align_your_steps_sigmas, need_width_height=True, need_inner_model=True),
    Scheduler("align_your_steps_beta", "Align Your Steps Beta", get_align_your_steps_with_beta_selection_sigmas, need_width_height=True, need_inner_model=True),
    Scheduler("beta", "Beta", beta_scheduler, need_inner_model=True),
    Scheduler("turbo", "Turbo", turbo_scheduler, need_inner_model=True),
    Scheduler("bong_tangent", "Bong Tangent", bong_tangent_scheduler),
    Scheduler("flow_match", "FlowMatchEulerDiscrete", flow_match_euler_discrete_scheduler, need_width_height=True, need_inner_model=True),
    Scheduler("flux2", "Flux2", flux2_scheduler, need_width_height=True),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
