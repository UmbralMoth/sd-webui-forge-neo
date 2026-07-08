import inspect

import k_diffusion
import k_diffusion.external
import torch

import modules.shared as shared
from backend.sampling.sampling_function import sampling_cleanup, sampling_prepare
from modules import devices, sd_samplers_cfg_denoiser, sd_samplers_common, sd_samplers_extra, sd_schedulers
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
from modules.sd_samplers_cfg_denoiser import CFGDenoiser  # noqa: F401
from modules.shared import opts

samplers_k_diffusion = [
    ("DPM++ 2M", "sample_dpmpp_2m", ["k_dpmpp_2m"], {"scheduler": "karras"}),
    ("DPM++ SDE", "sample_dpmpp_sde", ["k_dpmpp_sde"], {"scheduler": "karras", "second_order": True, "brownian_noise": True}),
    ("DPM++ 2M SDE", "sample_dpmpp_2m_sde", ["k_dpmpp_2m_sde"], {"scheduler": "exponential", "brownian_noise": True}),
    ("DPM++ 3M SDE", "sample_dpmpp_3m_sde", ["k_dpmpp_3m_sde"], {"scheduler": "exponential", "brownian_noise": True}),
    ("DPM++ 3M SDE Ctrl-Z", "sample_dpmpp_3m_sde_ctrlz", ["k_dpmpp_3m_sde_ctrlz"], {"scheduler": "exponential", "brownian_noise": True}),
    ("Flux Realistic" if opts.forbidden_knowledge else "DPM++ 2s a RF", "sample_dpmpp_2s_ancestral_RF", ["sample_dpmpp_2s_ancestral_RF"], {}),
    ("Euler a2 RF", "sample_euler_a2", ["euler_a2_rf"], {}),
    ("Euler a", "sample_euler_ancestral", ["k_euler_a", "k_euler_ancestral"], {"uses_ensd": True}),
    ("Euler", "sample_euler", ["k_euler"], {}),
    ("ER SDE", "sample_er_sde", ["er_sde"], {}),
    ("LCM", "sample_lcm", ["k_lcm"], {}),
    ("LMS", "sample_lms", ["k_lms"], {}),
    ("Heun", "sample_heun", ["k_heun"], {"second_order": True}),
    ("DPM2", "sample_dpm_2", ["k_dpm_2"], {"scheduler": "karras", "second_order": True}),
    ("Res Multistep", "sample_res_multistep", ["res_multistep"], {}),
    ("Kohaku LoNyu Yog", "sample_Kohaku_LoNyu_Yog", ["Kohaku_LoNyu_Yog"], {}),
    ("Restart", sd_samplers_extra.restart_sampler, ["restart"], {"scheduler": "karras", "second_order": True}),
    ("UniPC", sd_samplers_extra.sample_unipc, ["unipc"], {}),
    ("Forge Chimera", sd_samplers_extra.sample_forge_chimera, ["k_forge_chimera"], {"scheduler": "karras", "brownian_noise": True}),
    ("A-FloPS", sd_samplers_extra.sample_aflops, ["k_aflops"], {}),
]


samplers_data_k_diffusion = [sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options) for label, funcname, aliases, options in samplers_k_diffusion if callable(funcname) or hasattr(k_diffusion.sampling, funcname)]

sampler_extra_params = {
    "sample_dpmpp_sde": ["eta", "s_noise", "r"],
    "sample_dpmpp_2m_sde": ["eta", "s_noise", "solver_type"],
    "sample_dpmpp_3m_sde": ["eta", "s_noise"],
    "sample_dpmpp_3m_sde_ctrlz": ["eta", "s_noise", "gamma_scale"],
    "sample_dpmpp_3m_sde_cfgpp_ctrlz": ["eta", "s_noise"],
    "sample_euler_ancestral": ["eta", "s_noise"],
    "sample_euler_a2": ["eta", "s_noise"],
    "sample_forge_chimera": ["eta", "s_noise"],
    "sample_aflops": ["s_noise"],
    "sample_euler": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
    "sample_heun": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
    "sample_dpm_2": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
}

k_diffusion_samplers_map = {x.name: x for x in samplers_data_k_diffusion}
k_diffusion_scheduler = {x.name: x.function for x in sd_schedulers.schedulers}


class CFGDenoiserKDiffusion(sd_samplers_cfg_denoiser.CFGDenoiser):
    @property
    def inner_model(self):
        if self.model_wrap is None:
            self.model_wrap = k_diffusion.external.ForgeScheduleLinker(shared.sd_model.forge_objects.unet.model.predictor)
            self.model_wrap.inner_model = shared.sd_model

        return self.model_wrap


class KDiffusionSampler(sd_samplers_common.Sampler):
    def __init__(self, funcname, sd_model, options=None):
        super().__init__(funcname)

        self.extra_params = sampler_extra_params.get(funcname, [])

        self.options = options or {}
        self.func = funcname if callable(funcname) else getattr(k_diffusion.sampling, self.funcname)

        self.model_wrap_cfg = CFGDenoiserKDiffusion(self)
        self.model_wrap = self.model_wrap_cfg.inner_model

    def get_sigmas(self, p, steps):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get("discard_next_to_last_sigma", False)

        # Discard the penultimate sigma for samplers that need it to avoid a final
        # NaN-producing step at sigma=0. ComfyUI v0.24.0's canonical set is
        # ('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'). Forge's historical
        # list was larger; the 3M SDE entries were an older k_diffusion workaround
        # that is no longer needed. We additionally skip the discard on flow
        # architectures, where the warped schedule can produce a duplicate
        # non-zero penultimate sigma that meaningfully shifts the trajectory.
        if self.funcname in ["sample_dpmpp_3m_sde", "sample_dpmpp_3m_sde_ctrlz", "sample_dpm_2", "sample_unipc"]:
            is_flow = getattr(shared.sd_model, 'is_flow', False) or getattr(shared.sd_model, 'is_anima', False)
            if not is_flow:
                discard_next_to_last_sigma = True

        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            
        if discard_next_to_last_sigma:
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        scheduler_name = (p.hr_scheduler if p.is_hr_pass else p.scheduler) or "Automatic"
        if scheduler_name == "Automatic":
            from backend.args import dynamic_args

            if dynamic_args.klein:
                scheduler_name = "Flux2"
            else:
                scheduler_name = self.config.options.get("scheduler", None)

        scheduler = sd_schedulers.schedulers_map.get(scheduler_name)

        m_sigma_min, m_sigma_max = self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item()
        sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (m_sigma_min, m_sigma_max)

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif scheduler is None or scheduler.function is None:
            sigmas = self.model_wrap.get_sigmas(steps)
        else:
            sigmas_kwargs = {"sigma_min": sigma_min, "sigma_max": sigma_max}

            if scheduler.label != "Automatic" and not p.is_hr_pass:
                p.extra_generation_params["Schedule type"] = scheduler.label
            elif scheduler.label != p.extra_generation_params.get("Schedule type"):
                p.extra_generation_params["Hires schedule type"] = scheduler.label

            if opts.sigma_min != 0 and opts.sigma_min != m_sigma_min:
                sigmas_kwargs["sigma_min"] = opts.sigma_min
                p.extra_generation_params["Schedule min sigma"] = opts.sigma_min
            if opts.sigma_max != 0 and opts.sigma_max != m_sigma_max:
                sigmas_kwargs["sigma_max"] = opts.sigma_max
                p.extra_generation_params["Schedule max sigma"] = opts.sigma_max

            if scheduler.default_rho != -1 and opts.rho != 0 and opts.rho != scheduler.default_rho:
                sigmas_kwargs["rho"] = opts.rho
                p.extra_generation_params["Schedule rho"] = opts.rho

            if scheduler.need_inner_model:
                sigmas_kwargs["inner_model"] = self.model_wrap

            if scheduler.need_width_height:
                sigmas_kwargs["width"] = p.width
                sigmas_kwargs["height"] = p.height

            if scheduler.label == "Beta":
                p.extra_generation_params["Beta schedule alpha"] = opts.beta_dist_alpha
                p.extra_generation_params["Beta schedule beta"] = opts.beta_dist_beta

            if scheduler.need_width_height:
                if p.is_hr_pass:
                    sigmas_kwargs["width"] = p.hr_upscale_to_x
                    sigmas_kwargs["height"] = p.hr_upscale_to_y
                else:
                    sigmas_kwargs["width"] = p.width
                    sigmas_kwargs["height"] = p.height

            sigmas = scheduler.function(n=steps, **sigmas_kwargs, device=devices.cpu)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas.cpu()

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(self.model_wrap.inner_model.forge_objects.unet, x=x)

        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

        sigmas = self.get_sigmas(p, steps).to(x.device)
        sigma_sched = sigmas[steps - t_enc - 1 :]

        x = x.to(noise)

        predictor = self.model_wrap.predictor
        is_flow = predictor.prediction_type == "const"

        # sigma_sched[0] is the first sigma in the sliced schedule. For all predictors
        # (EDM/eps AND flow/const), this is the noise level at which the sampler will
        # start. We use this directly so the initial mix xi is consistent with the
        # first sampler step. For flow/const, noise_scaling(sigma, noise, latent, False)
        # computes xi = sigma*noise + (1-sigma)*latent, which is the correct flow
        # interpolation at the schedule's starting noise level.
        # (Note: for shifted flow schedules, sigma_sched[0] != denoise_strength --
        # this is intentional, as the schedule's start position corresponds to the
        # model's noise level at the chosen denoise step, not the literal interp.)
        noise_sigma = sigma_sched[0]

        xi = predictor.noise_scaling(noise_sigma, noise, x, max_denoise=False)

        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            # For EDM/eps (xi = latent + sigma*noise), extra noise is equivalent to
            # bumping sigma by extra_noise, so add it as-is.
            # For flow/const (xi = sigma*noise + (1-sigma)*latent), the equivalent is
            # to add (1-sigma)*noise*extra_noise -- it increases the noise weight at
            # the expense of the latent weight, matching the EDM semantics.
            if is_flow:
                xi += noise * opts.img2img_extra_noise * (1.0 - noise_sigma)
            else:
                xi += noise * opts.img2img_extra_noise

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if "sigma_min" in parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs["sigma_min"] = sigma_sched[-2]
        if "sigma_max" in parameters:
            extra_params_kwargs["sigma_max"] = sigma_sched[0]
        if "n" in parameters:
            extra_params_kwargs["n"] = len(sigma_sched) - 1
        if "sigma_sched" in parameters:
            extra_params_kwargs["sigma_sched"] = sigma_sched
        if "sigmas" in parameters:
            extra_params_kwargs["sigmas"] = sigma_sched

        if self.config.options.get("brownian_noise", False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs["noise_sampler"] = noise_sampler

        if self.config.options.get("solver_type", None) == "heun":
            extra_params_kwargs["solver_type"] = "heun"

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        self.sampler_extra_args = {
            "cond": conditioning,
            "image_cond": image_conditioning,
            "uncond": unconditional_conditioning,
            "cond_scale": p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }
        if getattr(p, 'empty_c', None) is not None:
            # TraSCE: pass empty conditioning for the Direction = Empty + CFG*(Pos - Neg) formula.
            self.sampler_extra_args["cond_empty"] = p.empty_c
        if getattr(p, 'tmg_base_c', None) is not None:
            self.sampler_extra_args["cond_base"] = p.tmg_base_c

        p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"] = sigmas

        samples = self.launch_sampling(
            t_enc + 1,
            lambda: self.func(self.model_wrap_cfg, xi, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs),
        )

        samples = predictor.inverse_noise_scaling(sigma_sched[-1], samples)

        self.add_infotext(p)

        sampling_cleanup(unet_patcher)

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(self.model_wrap.inner_model.forge_objects.unet, x=x)

        steps = steps or p.steps

        sigmas = self.get_sigmas(p, steps).to(x.device)

        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True

        x = self.model_wrap.predictor.noise_scaling(sigmas[0], x, torch.zeros_like(x), max_denoise=opts.sgm_noise_multiplier)

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if "n" in parameters:
            extra_params_kwargs["n"] = steps

        if "sigma_min" in parameters:
            extra_params_kwargs["sigma_min"] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs["sigma_max"] = self.model_wrap.sigmas[-1].item()

        if "sigmas" in parameters:
            extra_params_kwargs["sigmas"] = sigmas

        if self.config.options.get("brownian_noise", False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs["noise_sampler"] = noise_sampler

        if self.config.options.get("solver_type", None) == "heun":
            extra_params_kwargs["solver_type"] = "heun"

        self.last_latent = x
        self.sampler_extra_args = {
            "cond": conditioning,
            "image_cond": image_conditioning,
            "uncond": unconditional_conditioning,
            "cond_scale": p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }
        if getattr(p, 'empty_c', None) is not None:
            # TraSCE: pass empty conditioning into extra_args so CFGDenoiser can apply
            # Direction = Empty + CFG*(Pos - Neg) instead of the legacy Neg + CFG*(Pos - Neg).
            self.sampler_extra_args["cond_empty"] = p.empty_c
        if getattr(p, 'tmg_base_c', None) is not None:
            self.sampler_extra_args["cond_base"] = p.tmg_base_c

        p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"] = sigmas

        samples = self.launch_sampling(
            steps,
            lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs),
        )

        samples = self.model_wrap.predictor.inverse_noise_scaling(sigmas[-1], samples)

        self.add_infotext(p)

        sampling_cleanup(unet_patcher)

        return samples
