"""
Orthogonal Conditioning Steering (OCS)
=====================================
Injects a quality steering vector into conditioning tensors to avoid subject bias
from quality tags. Uses mean-pooled quality deltas and orthogonal projection.
"""

import copy
import torch
import gradio as gr
import logging

from modules import scripts, script_callbacks, prompt_parser
from modules.ui_components import InputAccordion
from backend.logging import setup_logger

logger = logging.getLogger("OCS")
setup_logger(logger)

# ══════════════════════════════════════════════════════════════════════════════
# Encoding
# ══════════════════════════════════════════════════════════════════════════════

def _get_cond(d, keys):
    if not isinstance(d, dict): return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _unwrap_cond(c):
    """Extracts (cross [T, D], pool [D]) from Forge/Comfy dict/list/tensor."""
    cross, pool = None, None

    if isinstance(c, dict):
        cross = _get_cond(c, ("crossattn", "cross_attn", "c_crossattn"))
        pool  = _get_cond(c, ("vector", "pooled_output", "y"))

    elif isinstance(c, (list, tuple)) and c:
        first = c[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            tensor_part, extra_part = first
            if isinstance(tensor_part, torch.Tensor):
                cross = tensor_part
            if isinstance(extra_part, dict):
                pool = _get_cond(extra_part, ("pooled_output", "vector", "y"))
        elif isinstance(first, dict):
            cross = _get_cond(first, ("crossattn", "cross_attn", "c_crossattn"))
            pool  = _get_cond(first, ("vector", "pooled_output", "y"))
        elif isinstance(first, torch.Tensor):
            cross = first

    elif isinstance(c, torch.Tensor):
        cross = c

    # Normalize to [T, D] and [D]
    if isinstance(cross, torch.Tensor) and cross.ndim == 3 and cross.shape[0] == 1:
        cross = cross[0]
    if isinstance(pool, torch.Tensor) and pool.ndim == 2 and pool.shape[0] == 1:
        pool = pool[0]

    return cross, pool


def get_active_len(tensor, sd_model=None):
    """Detects content length by diffing against empty-string encoding or tail padding."""
    if tensor is None: return 77
    if sd_model:
        try:
            pad_cond = prompt_parser.get_learned_conditioning(sd_model, [""], 1)[0][0].cond
            pad, _ = _unwrap_cond(pad_cond)
            if pad is not None:
                T = min(tensor.shape[0], pad.shape[0])
                for i in range(T - 1, 0, -1):
                    if not torch.allclose(tensor[i], pad[i], atol=1e-4): return i + 1
        except: pass
    
    last = tensor[-1]
    for i in range(tensor.shape[0] - 1, 0, -1):
        if not torch.allclose(tensor[i], last, atol=1e-4): return i + 1
    return max(2, tensor.shape[0])


def _is_vpred(sd_model):
    """Check if the active model uses v-parameterization."""
    candidates = [sd_model]
    obj = sd_model
    for _ in range(4):
        inner = getattr(obj, "inner_model", None)
        if inner is None:
            break
        candidates.append(inner)
        obj = inner
    for m in candidates:
        if getattr(m, "parameterization", None) == "v":
            return True
        model_attr = getattr(m, "model", None)
        if model_attr and getattr(model_attr, "parameterization", None) == "v":
            return True
    return False





# ══════════════════════════════════════════════════════════════════════════════
# Vector Math
# ══════════════════════════════════════════════════════════════════════════════

def perp_reject(v, basis):
    """v_ortho = v - proj_basis(v)"""
    dot = (v * basis).sum()
    base_sq = (basis * basis).sum().clamp(min=1e-12)
    return v - (dot / base_sq) * basis


def compute_steering_vectors(pos_cross, neg_cross, pos_pool, neg_pool, pos_len, base_cross=None, base_pool=None, base_len=None, use_perp=False):
    """Computes mean quality direction [D] for cross-attn and raw delta [D] for pooled."""
    if pos_cross is None or neg_cross is None: return None, None
    T = min(pos_cross.shape[0], neg_cross.shape[0])
    
    v_raw = pos_cross[:T] - neg_cross[:T]
    active_end = min(pos_len, T)
    if active_end < 2: active_end = T
    v_mean = v_raw[1:active_end].mean(dim=0) # skip BOS

    if use_perp and base_cross is not None:
        b_end = min(base_len or T, base_cross.shape[0])
        if b_end > 1:
            v_mean = perp_reject(v_mean, base_cross[1:b_end].mean(dim=0))

    v_pool = None
    if pos_pool is not None and neg_pool is not None:
        v_pool = pos_pool - neg_pool
        if use_perp and base_pool is not None:
            basis_unit = base_pool / (torch.linalg.vector_norm(base_pool).clamp(min=1e-12))
            v_pool = perp_reject(v_pool, basis_unit)

    return v_mean, v_pool


# ══════════════════════════════════════════════════════════════════════════════
# Extension logic
# ══════════════════════════════════════════════════════════════════════════════

class OrthogonalConditioningSteering(scripts.Script):
    sorting_priority = 2028
    def title(self): return "Orthogonal Conditioning Steering"
    def show(self, is_img2img): return scripts.AlwaysVisible

    def ui(self, is_img2img):
        px = "img2img" if is_img2img else "txt2img"
        with InputAccordion(False, label="Orthogonal Conditioning Steering", elem_id=f"{px}_ocs_enable") as enable:
            gr.Markdown("Injects a quality steering vector into conditioning to avoid subject bias.")
            with gr.Row():
                pos_tags = gr.Textbox(label="Positive Quality Tags", value="masterpiece, best quality, highly detailed, score_9, score_8_up", elem_id=f"{px}_ocs_pos")
                neg_tags = gr.Textbox(label="Negative Quality Tags", value="worst quality, low quality, blurry, score_1, score_2, score_3", elem_id=f"{px}_ocs_neg")
            with gr.Row():
                w_cross = gr.Slider(0.0, 3.0, 0.75, step=0.05, label="Cross-Attention Weight", elem_id=f"{px}_ocs_cw")
                w_pool  = gr.Slider(0.0, 3.0, 1.0, step=0.05, label="Pooled Vector Weight", elem_id=f"{px}_ocs_pw")
            with gr.Row():
                use_perp = gr.Checkbox(True, label="Perpendicular Projection", info="Removes subject-parallel components.", elem_id=f"{px}_ocs_perp")
                steer_un = gr.Checkbox(False, label="Steer Unconditional (Push-Pull)", info="Applies opposite push to Uncond.", elem_id=f"{px}_ocs_un")

        self.infotext_fields = [(enable, "OCS Enable"), (pos_tags, "OCS Pos"), (neg_tags, "OCS Neg"), (w_cross, "OCS CW"), (w_pool, "OCS PW"), (use_perp, "OCS Perp"), (steer_un, "OCS Uncond")]
        return [enable, pos_tags, neg_tags, w_cross, w_pool, use_perp, steer_un]

    def process(self, p, enable, pos_tags, neg_tags, w_cross, w_pool, use_perp, steer_un):
        if not enable or (w_cross <= 0 and w_pool <= 0): return
        p.extra_generation_params.update({"OCS Enable": True, "OCS Pos": pos_tags, "OCS Neg": neg_tags, "OCS CW": w_cross, "OCS PW": w_pool, "OCS Perp": use_perp, "OCS Uncond": steer_un})

        # ── 1. Resolve Schedules & Encode ─────────────────────────────────────
        steps = p.steps
        base_prompt = p.all_prompts[0] if p.all_prompts else p.prompt
        
        try:
            conds = prompt_parser.get_learned_conditioning(p.sd_model, [pos_tags, neg_tags, base_prompt], steps)
            pos_cond_raw = conds[0][0].cond
            neg_cond_raw = conds[1][0].cond
            base_cond_raw = conds[2][0].cond
        except Exception as exc:
            logger.error(f"Encoding error: {exc}")
            return
            
        pos_cross, pos_pool = _unwrap_cond(pos_cond_raw)
        neg_cross, neg_pool = _unwrap_cond(neg_cond_raw)
        base_cross, base_pool = _unwrap_cond(base_cond_raw)
        
        # ── 2. Pre-calculate Velocity ─────────────────────────────────────────
        
        v_m, v_p, base_len = None, None, 77
        
        if pos_cross is not None and neg_cross is not None:
            plen = get_active_len(pos_cross, p.sd_model)
            base_len = get_active_len(base_cross, p.sd_model) if base_cross is not None else 77

            v_m, v_p = compute_steering_vectors(pos_cross, neg_cross, pos_pool, neg_pool, plen, base_cross, base_pool, base_len, use_perp)
            
            is_vp = _is_vpred(p.sd_model)
            vpred_pool_dampen = 0.3 if is_vp else 1.0
            if is_vp:
                logger.info("V-Pred model detected — dampening pooled vector push (×0.3) to prevent AdaLN saturation.")

            if v_p is not None:
                v_p = v_p * vpred_pool_dampen

        vm_norm = torch.linalg.vector_norm(v_m).item() if v_m is not None else 0.0
        vp_norm = torch.linalg.vector_norm(v_p).item() if v_p is not None else 0.0
        logger.debug(f"‖v_m‖={vm_norm:.4f}, ‖v_p‖={vp_norm:.4f}")
        p.ocs_payload = {"v_m": v_m.detach().cpu() if v_m is not None else None, "v_p": v_p.detach().cpu() if v_p is not None else None, "N": base_len, "wc": w_cross, "wp": w_pool, "sun": steer_un}

        def steering_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            payload = getattr(p, "ocs_payload", None)
            if not payload: return model, x, timestep, uncond, cond, cond_scale, model_options, seed
            
            if not getattr(p, "_ocs_fired", False):
                logger.info(f"Steering active: N={payload['N']}, wc={payload['wc']}, wp={payload['wp']}")
                p._ocs_fired = True

            v_m, v_p, N, wc, wp = payload["v_m"], payload["v_p"], payload["N"], payload["wc"], payload["wp"]

            def push(c_list, mult=1.0):
                out = []
                for c_dict in c_list:
                    nc = copy.copy(c_dict)
                    ca_key = next((k for k in ("crossattn", "cross_attn", "c_crossattn") if k in nc), None)
                    if ca_key and v_m is not None and wc > 0:
                        ca = nc[ca_key].clone()
                        target = v_m.to(ca)
                        end = min(N, ca.shape[1])
                        if end > 1: ca[:, 1:end] += target * (wc * mult)
                        nc[ca_key] = ca
                        _rebuild_conds(nc, ("c_crossattn", "crossattn", "cross_attn"), ca)

                    po_key = next((k for k in ("vector", "pooled_output", "y") if k in nc), None)
                    if po_key and v_p is not None and wp > 0:
                        pv = nc[po_key].clone()
                        target = v_p.to(pv)
                        if pv.ndim == 2: pv += (target.unsqueeze(0) * wp) * mult
                        else: pv += target * wp * mult
                        nc[po_key] = pv
                        _rebuild_conds(nc, ("y", "vector", "pooled_output"), pv)
                    out.append(nc)
                return out

            return model, x, timestep, push(uncond, -1.0) if payload["sun"] else uncond, push(cond, 1.0), cond_scale, model_options, seed
        
        steering_modifier.__name__ = "ocs_steering_modifier"

        unet = p.sd_model.forge_objects.unet
        if unet:
            _clear_mod(unet)
            unet.add_conditioning_modifier(steering_modifier, ensure_uniqueness=True)
            logger.debug(f"Modifier registered to UNet {id(unet):#x}")

    def postprocess(self, p, processed, *args):
        unet = getattr(getattr(p.sd_model, "forge_objects", None), "unet", None)
        if unet: _clear_mod(unet)

def _rebuild_conds(nc, keys, tensor):
    if "model_conds" not in nc: return
    mc = dict(nc["model_conds"])
    for k in keys:
        if k in mc:
            try: mc[k] = type(mc[k])(tensor)
            except:
                try: mc[k].cond = tensor
                except: pass
    nc["model_conds"] = mc

def _clear_mod(unet):
    if "conditioning_modifiers" in unet.model_options:
        unet.model_options["conditioning_modifiers"] = [m for m in unet.model_options["conditioning_modifiers"] if getattr(m, "__name__", "") != "ocs_steering_modifier"]

script_callbacks.on_script_unloaded(lambda: None)