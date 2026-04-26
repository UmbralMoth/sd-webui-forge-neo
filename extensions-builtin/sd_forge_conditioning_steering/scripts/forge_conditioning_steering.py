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

def _unwrap_cond(c):
    """Extracts (cross [T, D], pool [D]) from Forge/Comfy dict/list/tensor."""
    cross, pool = None, None

    if isinstance(c, dict):
        cross = c.get("crossattn") or c.get("cross_attn") or c.get("c_crossattn")
        pool  = c.get("vector")    or c.get("pooled_output") or c.get("y")

    elif isinstance(c, list) and c:
        first = c[0]
        if isinstance(first, dict):
            cross = first.get("crossattn") or first.get("cross_attn") or first.get("c_crossattn")
            pool  = first.get("vector")    or first.get("pooled_output") or first.get("y")
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


def batch_encode(sd_model, prompts):
    """Encodes list of prompts; handles batched dict/tensor/list returns."""
    if not prompts: return []
    with torch.no_grad():
        try:
            raw = sd_model.get_learned_conditioning(prompts)
        except Exception as exc:
            logger.error(f"Encoding error: {exc}")
            return [(None, None)] * len(prompts)

    results = []

    # Case 1: Batched Dict (Flux/SDXL)
    if isinstance(raw, dict):
        cross = raw.get("crossattn") or raw.get("cross_attn") or raw.get("c_crossattn")
        pool  = raw.get("vector")    or raw.get("pooled_output") or raw.get("y")

        if isinstance(cross, torch.Tensor) and cross.ndim == 3 and cross.shape[0] == len(prompts):
            for i in range(len(prompts)):
                p_i = pool[i] if (isinstance(pool, torch.Tensor) and pool.ndim >= 2 and pool.shape[0] == len(prompts)) else pool
                results.append((cross[i], p_i))
            return results
        results.append(_unwrap_cond(raw))
        return results

    # Case 2: Batched Tensor (Anima/Qwen)
    if isinstance(raw, torch.Tensor) and raw.ndim == 3 and raw.shape[0] == len(prompts):
        for i in range(len(prompts)):
            results.append((raw[i], None))
        return results

    # Case 3: List of conditionings (Legacy)
    if not isinstance(raw, list): raw = [raw]
    for item in raw:
        results.append(_unwrap_cond(item))
    return results


def get_active_len(tensor, sd_model=None):
    """Detects content length by diffing against empty-string encoding or tail padding."""
    if tensor is None: return 77
    if sd_model:
        try:
            pad_enc = batch_encode(sd_model, [""])
            if pad_enc and pad_enc[0][0] is not None:
                pad = pad_enc[0][0]
                T = min(tensor.shape[0], pad.shape[0])
                for i in range(T - 1, 0, -1):
                    if not torch.allclose(tensor[i], pad[i], atol=1e-4): return i + 1
        except: pass
    
    last = tensor[-1]
    for i in range(tensor.shape[0] - 1, 0, -1):
        if not torch.allclose(tensor[i], last, atol=1e-4): return i + 1
    return max(2, tensor.shape[0])


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
            v_pool = perp_reject(v_pool, base_pool)

    return v_mean, v_pool


def adaptive_norm(v, ref_norms, scale):
    """Rescales v to target magnitude: scale * mean(ref_norms)."""
    target = ref_norms.mean().item() * scale
    v_mag  = torch.linalg.vector_norm(v).clamp(min=1e-12)
    return v * (target / v_mag)


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
            with gr.Row():
                use_norm = gr.Checkbox(False, label="Adaptive Normalization", elem_id=f"{px}_ocs_norm")
                n_scale  = gr.Slider(0.01, 1.0, 0.15, step=0.01, label="Norm Scale", visible=False, elem_id=f"{px}_ocs_nscale")
            use_norm.change(fn=lambda x: gr.update(visible=x), inputs=[use_norm], outputs=[n_scale])

        self.infotext_fields = [(enable, "OCS Enable"), (pos_tags, "OCS Pos"), (neg_tags, "OCS Neg"), (w_cross, "OCS CW"), (w_pool, "OCS PW"), (use_perp, "OCS Perp"), (steer_un, "OCS Uncond"), (use_norm, "OCS Norm"), (n_scale, "OCS NScale")]
        return [enable, pos_tags, neg_tags, w_cross, w_pool, use_perp, steer_un, use_norm, n_scale]

    def process(self, p, enable, pos_tags, neg_tags, w_cross, w_pool, use_perp, steer_un, use_norm, n_scale):
        if not enable or (w_cross <= 0 and w_pool <= 0): return
        p.extra_generation_params.update({"OCS Enable": True, "OCS Pos": pos_tags, "OCS Neg": neg_tags, "OCS CW": w_cross, "OCS PW": w_pool, "OCS Perp": use_perp, "OCS Uncond": steer_un, "OCS Norm": use_norm, "OCS NScale": n_scale if use_norm else "—"})

        # ── 1. Resolve Schedules ──────────────────────────────────────────────
        steps = p.steps
        pos_sched  = prompt_parser.get_learned_conditioning_prompt_schedules([pos_tags], steps)[0]
        neg_sched  = prompt_parser.get_learned_conditioning_prompt_schedules([neg_tags], steps)[0]
        base_sched = prompt_parser.get_learned_conditioning_prompt_schedules([p.all_prompts[0] if p.all_prompts else p.prompt], steps)[0]

        # Collect unique texts for batch encoding
        unique_texts = set()
        for _, t in pos_sched:  unique_texts.add(t)
        for _, t in neg_sched:  unique_texts.add(t)
        if use_perp or use_norm:
            for _, t in base_sched: unique_texts.add(t)
        
        text_list = list(unique_texts)
        logger.debug(f"Encoding {len(text_list)} unique schedule segments...")
        enc_results = batch_encode(p.sd_model, text_list)
        text_map = {txt: res for txt, res in zip(text_list, enc_results)}

        # ── 2. Pre-calculate Velocity Schedule ────────────────────────────────
        v_schedule = []
        
        def get_at(sched, s):
            for end_step, txt in sched:
                if s <= end_step: return txt
            return sched[-1][1]

        for s in range(1, steps + 1):
            pt, nt, bt = get_at(pos_sched, s), get_at(neg_sched, s), get_at(base_sched, s)
            
            pc, pp = text_map.get(pt, (None, None))
            nc, np = text_map.get(nt, (None, None))
            bc, bp = text_map.get(bt, (None, None))
            
            if pc is None or nc is None:
                v_schedule.append(None)
                continue
                
            plen = get_active_len(pc, p.sd_model)
            blen = get_active_len(bc, p.sd_model) if bc is not None else 77

            vm, vp = compute_steering_vectors(pc, nc, pp, np, plen, bc, bp, blen, use_perp)
            
            if vm is not None and use_norm and bc is not None:
                vm = adaptive_norm(vm, torch.linalg.vector_norm(bc[1:blen], dim=-1), n_scale)
                if vp is not None and bp is not None:
                    vp = adaptive_norm(vp, torch.linalg.vector_norm(bp).unsqueeze(0), n_scale)

        logger.debug(f"‖v_m‖={torch.linalg.vector_norm(v_m).item():.4f}, ‖v_p‖={torch.linalg.vector_norm(v_p).item():.4f if v_p is not None else 0}")
        p.ocs_payload = {"v_m": v_m.detach().cpu(), "v_p": v_p.detach().cpu() if v_p is not None else None, "N": base_len, "wc": w_cross, "wp": w_pool, "sun": steer_un}

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
        unet.model_options["conditioning_modifiers"] = [m for m in unet.model_options["conditioning_modifiers"] if getattr(m, "__name__", "") != "steering_modifier"]

script_callbacks.on_script_unloaded(lambda: None)