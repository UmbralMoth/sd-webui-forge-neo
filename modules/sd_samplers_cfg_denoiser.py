import torch

from backend.sampling.sampling_function import sampling_function
from modules import prompt_parser, sd_samplers_common
from modules.script_callbacks import AfterCFGCallbackParams, CFGDenoiserParams, cfg_after_cfg_callback, cfg_denoiser_callback
from modules.shared import opts, state


def catenate_conds(conds):
    if not isinstance(conds[0], dict):
        return torch.cat(conds)

    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


def subscript_cond(cond, a, b):
    if not isinstance(cond, dict):
        return cond[a:b]

    return {key: vec[a:b] for key, vec in cond.items()}


def pad_cond(tensor, repeats, empty):
    if not isinstance(tensor, dict):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)

    tensor["crossattn"] = pad_cond(tensor["crossattn"], repeats, empty)
    return tensor


class CFGDenoiser(torch.nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, sampler):
        super().__init__()
        self.model_wrap = None
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        """number of steps as specified by user in UI"""

        self.total_steps = None
        """expected number of calls to denoiser calculated from self.steps and specifics of the selected sampler"""

        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        self.sampler = sampler
        self.p = None

        self.need_last_noise_uncond = False
        self.last_noise_uncond = None

        # Backward Compatibility
        self.mask_before_denoising = False

        self.classic_ddim_eps_estimation = False

    @property
    def inner_model(self):
        raise NotImplementedError()

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale, timestep, x_in, cond):
        denoised_uncond = x_out[-uncond.shape[0] :]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    def get_pred_x0(self, x_in, x_out, sigma):
        return x_out

    def update_inner_model(self):
        self.model_wrap = None

        c, uc = self.p.get_conds()
        self.sampler.sampler_extra_args["cond"] = c
        self.sampler.sampler_extra_args["uncond"] = uc

        # TraSCE: keep cond_empty in sync with cond/uncond during refiner switches.
        if getattr(self.p, 'empty_c', None) is not None:
             self.sampler.sampler_extra_args["cond_empty"] = self.p.empty_c
        elif "cond_empty" in self.sampler.sampler_extra_args:
             del self.sampler.sampler_extra_args["cond_empty"]

    def pad_cond_uncond(self, *args, **kwargs):
        raise NotImplementedError

    def pad_cond_uncond_v0(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond, **kwargs):
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        original_x_device = x.device
        original_x_dtype = x.dtype

        if self.classic_ddim_eps_estimation:
            acd = self.inner_model.inner_model.alphas_cumprod
            fake_sigmas = ((1 - acd) / acd) ** 0.5
            real_sigma = fake_sigmas[sigma.round().long().clip(0, int(fake_sigmas.shape[0]))]
            real_sigma_data = 1.0
            x = x * (((real_sigma**2.0 + real_sigma_data**2.0) ** 0.5)[:, None, None, None])
            sigma = real_sigma

        if sd_samplers_common.apply_refiner(self, x, sigma[0]):
            cond = self.sampler.sampler_extra_args["cond"]
            uncond = self.sampler.sampler_extra_args["uncond"]

        cond_composition, cond = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step) if uncond is not None else None

        if self.mask is not None:
            predictor = self.inner_model.inner_model.forge_objects.unet.model.predictor
            noisy_initial_latent = predictor.noise_scaling(sigma[:, None, None, None], torch.randn_like(self.init_latent).to(self.init_latent), self.init_latent, max_denoise=False)
            x = x * self.nmask + noisy_initial_latent * self.mask

        denoiser_params = CFGDenoiserParams(x, image_cond, sigma, state.sampling_step, state.sampling_steps, cond, uncond, self)

        # Internal ZigZag sampling uses several internal model() calls (probe/invert/commit).
        # Those calls should not count as "real" denoiser steps nor should they trigger
        # UI-facing callbacks (prompt-edit swaps, preview storage, etc.). The ZigZag
        # implementation marks internal calls with `__zigzag_internal=True` in
        # `extra_args` so downstream code can detect and ignore them.
        zigzag_internal = kwargs.get("__zigzag_internal", False)
        if not zigzag_internal:
            cfg_denoiser_callback(denoiser_params)

        # NGMS
        if self.p.is_hr_pass == True:
            # We check if the sampler is using CFG++ or CFG and then adjust accordingly
            if kwargs.get("cfgpp", False):
                cond_scale = self.p.hr_cfg / 12.5
            else:
                cond_scale = self.p.hr_cfg

        if 0 < self.step / self.total_steps <= opts.skip_early_cond:
            cond_scale = 1.0
            self.p.extra_generation_params["Skip Early CFG"] = opts.skip_early_cond
        elif opts.zero_cfg_init:
            warmup_threshold = 0.35
            progress = self.step / max(self.total_steps * warmup_threshold, 1e-5)
            if progress < 1.0:
                target_cfg = cond_scale
                cond_scale = 1.0 + (target_cfg - 1.0) * progress
                self.p.extra_generation_params["Dynamic Zero-CFG"] = True
        elif (self.step % 2 or opts.s_min_uncond_all) and (0 < sigma[0] < s_min_uncond):
            cond_scale = 1.0
            self.p.extra_generation_params["NGMS"] = s_min_uncond
            if opts.s_min_uncond_all:
                self.p.extra_generation_params["NGMS all steps"] = opts.s_min_uncond_all

        extra_model_options = kwargs.get("model_options", {})
        # TraSCE: reconstruct empty_c for the current step (handles prompt-edit schedules)
        # and forward it into model_options for sampling_function to compile and apply.
        if "cond_empty" in self.sampler.sampler_extra_args:
            if getattr(self, "cached_cond_empty", None) is None:
                self.cached_cond_empty = prompt_parser.reconstruct_cond_batch(self.sampler.sampler_extra_args["cond_empty"], self.step)
            extra_model_options["cond_empty"] = self.cached_cond_empty
        denoised, cond_pred, uncond_pred = sampling_function(self, denoiser_params=denoiser_params, cond_scale=cond_scale, cond_composition=cond_composition, extra_model_options=extra_model_options)

        if self.need_last_noise_uncond:
            self.last_noise_uncond = (x - uncond_pred) / sigma[:, None, None, None]

        if self.mask is not None:
            blended_latent = denoised * self.nmask + self.init_latent * self.mask

            if self.p.scripts is not None:
                from modules import scripts

                mba = scripts.MaskBlendArgs(denoised, self.nmask, self.init_latent, self.mask, blended_latent, denoiser=self, sigma=sigma)
                self.p.scripts.on_mask_blend(self.p, mba)
                blended_latent = mba.blended_latent

            denoised = blended_latent

        if not zigzag_internal:
            preview = self.sampler.last_latent = denoised
            sd_samplers_common.store_latent(preview)


            after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
            cfg_after_cfg_callback(after_cfg_callback_params)
            denoised = after_cfg_callback_params.x

            self.step += 1

        if self.classic_ddim_eps_estimation:
            eps = (x - denoised) / sigma[:, None, None, None]
            return eps

        return denoised.to(device=original_x_device, dtype=original_x_dtype)
