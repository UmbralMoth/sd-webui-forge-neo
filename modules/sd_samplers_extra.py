import k_diffusion.sampling
import torch
import tqdm

from modules.uni_pc import uni_pc as unipc_impl


@torch.no_grad()
def restart_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, restart_list=None):
    """
    Implements restart sampling in Restart Sampling for Improving Generative Processes (2023)
    Restart_list format: {min_sigma: [ restart_steps, restart_times, max_sigma]}
    If restart_list is None: will choose restart_list automatically, otherwise will use the given restart_list
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0
    from k_diffusion.sampling import get_sigmas_karras, to_d

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({"x": x, "i": step_id, "sigma": new_sigma, "sigma_hat": old_sigma, "denoised": denoised})
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(), device=sigmas.device)
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(), device=sigmas.device)[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            x = x + k_diffusion.sampling.torch.randn_like(x) * s_noise * (old_sigma**2 - last_sigma**2) ** 0.5
        x = heun_step(x, old_sigma, new_sigma)
        last_sigma = new_sigma

    return x


class SigmaConvert:
    schedule = ""

    def marginal_log_mean_coeff(self, sigma):
        return 0.5 * torch.log(1 / ((sigma * sigma) + 1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T]
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


def predict_eps_sigma(model, input, sigma_in, **kwargs):
    sigma = sigma_in.view(sigma_in.shape[:1] + (1,) * (input.ndim - 1))
    input = input * ((sigma**2 + 1.0) ** 0.5)
    return (input - model(input, sigma_in, **kwargs)) / sigma


def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=False, variant="bh1"):
    timesteps = sigmas.clone()
    if sigmas[-1] < 0.001:
        timesteps[-1] = 0.001

    ns = SigmaConvert()

    x = x / torch.sqrt(1.0 + timesteps[0] ** 2.0)
    model_type = "noise"

    model_fn = unipc_impl.model_wrapper(
        lambda input, sigma, **kwargs: predict_eps_sigma(model, input, sigma, **kwargs),
        ns,
        model_type=model_type,
        guidance_type="uncond",
        model_kwargs=extra_args,
    )

    order = min(3, len(timesteps) - 2)
    uni_pc = unipc_impl.UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant=variant)
    x = uni_pc.sample(x, timesteps=timesteps, skip_type="time_uniform", method="multistep", order=order, lower_order_final=True, callback=callback, disable_pbar=disable)
    x /= ns.marginal_alpha(timesteps[-1])
    return x

import math
from modules import shared

@torch.no_grad()
def sample_forge_chimera(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, **kwargs):
    from modules_forge.packages.k_diffusion.sampling import _dpm_solver_step_standard, sigma_to_half_log_snr, BrownianTreeNoiseSampler
    from functools import partial
    import math
    import torch
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    is_flow = False
    if hasattr(shared, 'sd_model'):
        is_flow = getattr(shared.sd_model, 'is_flow', False) or getattr(shared.sd_model, 'is_anima', False)
        
    N = len(sigmas) - 1
    base_cfg = extra_args.get('cond_scale', 7.0)
    
    # Initialize robust DPM++ lambda function
    if is_flow or (sigmas[0] <= 2.0):
        def lambda_fn(sigma):
            sigma = torch.clamp(sigma, min=1e-4, max=1.0 - 1e-4)
            return -torch.logit(sigma)
    else:
        model_sampling = model.inner_model.predictor
        lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
        
    # SDE noise setup
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=True)
        
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None
    
    for i in range(N):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Calculate strictly step-based progress to guarantee UI-scheduler immunity
        progress = float(i) / max(1, N - 1)
            
        # Trapezoidal CFG curve: Ramp up -> Hold -> Mild Decay
        if progress <= 0.05:
            p = progress / 0.05
            p_smooth = p * p * (3 - 2 * p) # Smoothstep
            current_cfg = 1.0 + (base_cfg - 1.0) * p_smooth
        elif progress <= 0.65:
            current_cfg = base_cfg
        else:
            p = (progress - 0.65) / 0.35
            p_smooth = p * p * (3 - 2 * p)
            current_cfg = base_cfg * (1.0 - 0.2 * p_smooth)
            
        if 'cond_scale' in extra_args:
            extra_args['cond_scale'] = current_cfg
            
        denoised = model(x, sigma * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
            
        # Dynamic Eta calculation
        if progress < 0.70:
            current_eta = eta
        elif progress < 0.85:
            current_eta = eta * (1.0 - (progress - 0.70) / 0.15)
        else:
            current_eta = 0.0
            
        if sigma_next == 0 or i == N - 1:
            d = (x - denoised) / sigma
            dt = 0 - sigma
            x = x + d * dt
            continue
            
        lambda_s, lambda_t = lambda_fn(sigma), lambda_fn(sigma_next)
        h = lambda_t - lambda_s
        
        # User requested Architecture Phase Drops
        if progress >= 0.70:
            h_2 = None # Drop from 3M to 2M SDE/ODE
        
        # Core DPM++ 3M SDE Solver Step (Automatically handles 1M/2M warmup and phase drops)
        x = _dpm_solver_step_standard(
            x=x, t=lambda_s, s=lambda_t, 
            denoised=denoised, denoised_1=denoised_1, denoised_2=denoised_2, 
            h_1=h_1, h_2=h_2, 
            eta=current_eta, noise_sampler=noise_sampler, s_noise=s_noise, 
            sigma_t=sigma, sigma_s=sigma_next, lambda_fn=lambda_fn
        )
        
        # Shift history
        denoised_2, denoised_1 = denoised_1, denoised
        h_2, h_1 = h_1, h
        
    return x

@torch.no_grad()
def sample_aflops(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, **kwargs):
    """
    Implements A-FloPS (Adaptive Flow Path Sampler) 
    Reference: A-FloPS: Accelerating Diffusion Sampling with Adaptive Flow Path Sampler
    """
    from k_diffusion.sampling import to_d, trange
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    x_prev = None
    d_prev = None
    t_prev = None
    
    # Explicit model property detection for schedule mapping
    from modules import shared
    sd_model = getattr(shared, "sd_model", None)
    is_flow = getattr(sd_model, "is_flow", False) or getattr(sd_model, "is_anima", False)
    
    shift = 1.0
    if is_flow and sd_model:
        try:
            # Safe nested lookup for Forge architecture shift
            forge_objects = getattr(sd_model, "forge_objects", None)
            if forge_objects:
                # Support both attribute and dictionary-style access for robustness
                unet = getattr(forge_objects, "unet", None) if not isinstance(forge_objects, dict) else forge_objects.get("unet")
                if unet and hasattr(unet, "model") and hasattr(unet.model, "predictor"):
                    shift = getattr(unet.model.predictor, "shift", 3.0)
        except Exception:
            pass
            
    N = len(sigmas) - 1
    
    for i in trange(N, disable=disable):
        old_sigma = sigmas[i]
        new_sigma = sigmas[i + 1]
        
        denoised = model(x, old_sigma * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': old_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
            
        # 1. Manifold Linearization
        # Rectified Flow models use a shifted time domain. We must solve in the linear 't' space [1, 0].
        if is_flow:
            t = old_sigma / (shift - old_sigma * (shift - 1.0) + 1e-8)
            t_next = new_sigma / (shift - new_sigma * (shift - 1.0) + 1e-8)
            
            # Velocity in linearized t-space: v = dx/dt = dx/dsigma * dsigma/dt
            v_scaling = shift / ((1.0 + (shift - 1.0) * t)**2 + 1e-8)
            v = to_d(x, old_sigma, denoised) * v_scaling
        else:
            # Standard Diffusion (EPS/V) or unshifted Flow
            t = old_sigma
            t_next = new_sigma
            v = to_d(x, old_sigma, denoised)
            
        h = t_next - t  # negative step size
        
        if i == 0:
            # First order jump for the very first step (no history) and the final jump to 0.
            # This prevents numerical instability and error amplification near the sigma=0 singularity.
            x_next = x + v * h
            
            x_prev = x.detach()
            d_prev = v.detach()
            t_prev = t
            x = x_next

        else:
            # Second order A-FloPS for all intermediate steps.
            h_prev = t - t_prev  # negative
            
            deltax = x - x_prev
            deltav = v - d_prev
            
            # Optimized Global Rayleigh quotient calculation
            # Reshape is safer than view for non-contiguous tensors
            a_flat = deltax.reshape(deltax.size(0), -1)
            b_flat = deltav.reshape(deltav.size(0), -1)
            
            # Sum with float32 accumulator for overflow protection
            numerator = torch.sum(b_flat * a_flat, dim=1, dtype=torch.float32)
            denominator = torch.sum(a_flat ** 2, dim=1, dtype=torch.float32) + 1e-10
            c = (numerator / denominator).to(x.dtype)
            
            # Clamp c as per paper to prevent over-correction/instability
            c = torch.clamp(c, -1.0, 1.0)
            c_batch = c.reshape(-1, *([1] * (x.ndim - 1)))
            
            # Predict linear offset and slope in linearized space
            v1 = v - c_batch * x
            history_record = d_prev - c_batch * x_prev
            v2 = (v1 - history_record) / (h_prev + 1e-10)
            
            # Numerical Stability: Clamp exponent to prevent overflow (fp16 limit is ~11.0, but we use safe logic)
            ch = torch.clamp(c_batch * h, min=-40.0, max=40.0)
            c_abs = torch.abs(c_batch)
            is_small = c_abs < 1e-4
            
            # Safe division: Replace zeros in denominator branch to avoid NaNs even in unused branches
            safe_c = torch.where(is_small, torch.ones_like(c_batch), c_batch)
            
            # Use expm1 for high precision at small values and to avoid catastrophic cancellation
            # x_next = x * exp(ch) + d1 * (exp(ch)-1)/c + d2 * (exp(ch)-1-ch)/c^2
            exp_ch = torch.exp(ch)
            expm1_ch = torch.expm1(ch)
            
            term_A = torch.where(is_small, h + 0.5 * c_batch * (h ** 2), expm1_ch / safe_c)
            term_B = torch.where(is_small, 0.5 * (h ** 2) + (c_batch * (h ** 3)) / 6.0, (expm1_ch - ch) / (safe_c ** 2))
            
            x_next = x * exp_ch + term_A * v1 + term_B * v2
            
            x_prev = x.detach()
            d_prev = v.detach()
            t_prev = t
            x = x_next
            
    return x
