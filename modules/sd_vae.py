import glob
import os.path
from copy import deepcopy

import torch

from backend import memory_management, utils
from modules import hashes, paths, sd_models, shared
try:
    from ldm_patched.modules import diffusers_convert
except ImportError:
    diffusers_convert = None

vae_path = os.path.abspath(os.path.join(paths.models_path, "VAE"))
vae_ignore_keys: set[str] = {"model_ema.decay", "model_ema.num_updates"}
vae_dict: dict[str, os.PathLike] = {}

base_vae: dict[str, torch.Tensor] = None
loaded_vae_file: os.PathLike = None
checkpoint_info: "sd_models.CheckpointInfo" = None


@torch.inference_mode()
def _load_vae_dict(model, vae_sd: dict):
    sd = {k: v for k, v in vae_sd.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    sd = _normalize_vae_state_dict(sd)

    # Strip bn.* keys — these are Flux2-style training artifacts that IntegratedAutoencoderKL
    # doesn't have a module for; ignore them silently rather than crashing on strict load.
    sd_to_load = {k: v for k, v in sd.items() if not k.startswith("bn.")}

    # Prefer updating the Forge VAE wrapper so encode/decode during sampling uses the new weights.
    forge_vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    target = getattr(forge_vae, "first_stage_model", None) if forge_vae is not None else None
    if target is None:
        target = model.first_stage_model

    try:
        target.load_state_dict(sd_to_load, strict=True)
    except RuntimeError:
        # Architecture mismatch (e.g. channel count difference) — fall back to non-strict.
        missing, unexpected = target.load_state_dict(sd_to_load, strict=False)
        if unexpected:
            print(f"VAE load (non-strict): unexpected keys {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Keep all Forge references in sync.
    if forge_vae is not None:
        forge_vae.first_stage_model = target
        if hasattr(model, "forge_objects_original") and getattr(model.forge_objects_original, "vae", None) is not None:
            model.forge_objects_original.vae.first_stage_model = target
        if hasattr(model, "forge_objects_after_applying_lora") and getattr(model.forge_objects_after_applying_lora, "vae", None) is not None:
            model.forge_objects_after_applying_lora.vae.first_stage_model = target
    model.first_stage_model = target


def _looks_like_vae_state_dict(sd: dict) -> bool:
    if not sd:
        return False
    return (
        "decoder.conv_in.weight" in sd
        or "encoder.conv_in.weight" in sd
        or "decoder.up_blocks.0.resnets.0.norm1.weight" in sd
        or any(k.startswith("decoder.") or k.startswith("encoder.") for k in sd)
    )


def _normalize_vae_state_dict(sd: dict) -> dict:
    """Strip common checkpoint prefixes and convert diffusers format if needed."""
    if not sd or _looks_like_vae_state_dict(sd):
        out = sd
    else:
        out = sd
        for prefix in ("first_stage_model.", "vae.", "model.first_stage_model.", "model.vae."):
            cand = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if _looks_like_vae_state_dict(cand):
                print(f"VAE load: stripped prefix '{prefix}'")
                out = cand
                break

    if diffusers_convert is not None and "decoder.up_blocks.0.resnets.0.norm1.weight" in out:
        out = diffusers_convert.convert_vae_state_dict(out)

    return out


def get_loaded_vae_name() -> str:
    if loaded_vae_file is None:
        return None
    return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash() -> str:
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, "vae")
    return sha256[0:10] if sha256 else None


# def get_base_vae(model):
#     if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
#         return base_vae
#     return None


def store_base_vae(model):
    global base_vae, checkpoint_info
    assert loaded_vae_file is None
    memory_management.logger.debug("Storing Original VAE...")
    base_vae = deepcopy(model.first_stage_model.state_dict())
    checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None
    memory_management.soft_empty_cache()


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is None:
        return
    memory_management.logger.debug("Restoring Original VAE...")
    _load_vae_dict(model, base_vae)
    loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath: os.PathLike) -> str:
    return os.path.basename(filepath)


def refresh_vae_list():
    vae_dict.clear()
    paths = []

    file_extensions = ("ckpt", "pt", "pth", "bin", "safetensors", "sft", "gguf")

    for ext in file_extensions:
        paths.append(os.path.join(sd_models.model_path, f"**/*.vae.{ext}"))
        paths.append(os.path.join(vae_path, f"**/*.{ext}"))

    for _dir in shared.cmd_opts.vae_dirs:
        for ext in file_extensions:
            paths.append(os.path.join(_dir, f"**/*.{ext}"))

    candidates = []
    for path in paths:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath

    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


# def find_vae_near_checkpoint(checkpoint_file):
#     checkpoint_path = os.path.basename(checkpoint_file).rsplit(".", 1)[0]
#     for vae_file in vae_dict.values():
#         if os.path.basename(vae_file).startswith(checkpoint_path):
#             return vae_file

#     return None


# @dataclass
# class VaeResolution:
#     vae: str = None
#     source: str = None
#     resolved: bool = True

#     def tuple(self):
#         return self.vae, self.source


# def is_automatic():
#     return shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config


# def resolve_vae_from_setting() -> VaeResolution:
#     if shared.opts.sd_vae == "None":
#         return VaeResolution()

#     vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
#     if vae_from_options is not None:
#         return VaeResolution(vae_from_options, "specified in settings")

#     if not is_automatic():
#         print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

#     return VaeResolution(resolved=False)


# def resolve_vae_from_user_metadata(checkpoint_file) -> VaeResolution:
#     metadata = extra_networks.get_user_metadata(checkpoint_file)
#     vae_metadata = metadata.get("vae", None)
#     if vae_metadata is not None and vae_metadata != "Automatic":
#         if vae_metadata == "None":
#             return VaeResolution()

#         vae_from_metadata = vae_dict.get(vae_metadata, None)
#         if vae_from_metadata is not None:
#             return VaeResolution(vae_from_metadata, "from user metadata")

#     return VaeResolution(resolved=False)


# def resolve_vae_near_checkpoint(checkpoint_file) -> VaeResolution:
#     vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
#     if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
#         return VaeResolution(vae_near_checkpoint, "found near the checkpoint")

#     return VaeResolution(resolved=False)


# def resolve_vae(checkpoint_file) -> VaeResolution:
#     if shared.cmd_opts.vae_path is not None:
#         return VaeResolution(shared.cmd_opts.vae_path, "from commandline argument")

#     if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
#         return resolve_vae_from_setting()

#     res = resolve_vae_from_user_metadata(checkpoint_file)
#     if res.resolved:
#         return res

#     res = resolve_vae_near_checkpoint(checkpoint_file)
#     if res.resolved:
#         return res

#     res = resolve_vae_from_setting()

#     return res


def reload_vae_weights(vae: str):
    if vae in (None, "None", "Automatic"):
        return

    store_base_vae(shared.sd_model)
    vae_sd = utils.load_torch_file(vae)
    _load_vae_dict(shared.sd_model, vae_sd)


def restore_vae_weights():
    restore_base_vae(shared.sd_model)
