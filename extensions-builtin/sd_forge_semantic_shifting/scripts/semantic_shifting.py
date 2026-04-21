import torch
import gradio as gr

from modules import scripts
from modules.ui_components import InputAccordion


def encode_string(sd_model, prompt):
    with torch.no_grad():
        try:
            conds = sd_model.get_learned_conditioning([prompt])
            if isinstance(conds, dict) and "crossattn" in conds:
                return conds["crossattn"]
            elif isinstance(conds, list) and len(conds) > 0 and isinstance(conds[0], dict) and "crossattn" in conds[0]:
                return conds[0]["crossattn"]
            elif isinstance(conds, torch.Tensor):
                return conds
            elif isinstance(conds, list) and isinstance(conds[0], torch.Tensor):
                return conds[0]
            return conds
        except Exception as e:
            print(f"[Modulation Guidance] Exception during encode: {e}")
            return None


class SemanticShiftingForForge(scripts.Script):
    sorting_priority = 2027

    def title(self):
        return "Global Semantic Shifting"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label="Global Semantic Shifting (Attention-Space)") as enable:
            with gr.Row():
                pos_prompt = gr.Textbox(
                    label="Positive Quality Prompt", 
                    value="absurdres, masterpiece, best quality, highly detailed, score_7, score_8, score_9"
                )
                neg_prompt = gr.Textbox(
                    label="Negative Quality Prompt", 
                    value="worst quality, low quality, messy, bad anatomy, score_1, score_2, score_3"
                )
            
            with gr.Row():
                shift_mode = gr.Dropdown(
                    choices=["Orthogonal (Clean & Preserved)", "Additive (Raw & Atmospheric)"],
                    value="Orthogonal (Clean & Preserved)",
                    label="Injection Mode",
                    info="Orthogonal protects character structure. Additive is stronger but can smudge."
                )

            weight = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=2.0,
                step=0.1,
                label="Guidance Weight (w)",
                info="Higher = Stronger Quality Steer. Additive mode requires lower weights."
            )
            
            rescale_phi = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Normalization Penalty (Rescale Phi)",
                info="1.0 = Strict scale preservation. Prevents math blowout."
            )

        return [enable, pos_prompt, neg_prompt, shift_mode, weight, rescale_phi]

    # --- NEW: Compute heavy embeddings ONCE per generation ---
    def process(self, p, enable: bool, pos_prompt: str, neg_prompt: str, shift_mode: str, weight: float, rescale_phi: float=0.5, *args, **kwargs):
        if not enable or weight <= 0:
            return

        clip = getattr(p.sd_model.forge_objects, "clip", None)
        if clip is None:
            print("[Semantic Shifting] Error: Model does not have a properly initialized Text Encoder.")
            return

        # Contextualized Delta Vector Implementation:
        # We append the user's base prompt to force the Text Encoder's 
        # self-attention layers to contextualize the quality tags against the actual subject.
        base_prompt = getattr(p, "prompt", "")

        def get_active_mean(tensor):
            t = tensor[0] if tensor.ndim == 3 else tensor
            
            if t.ndim != 2:
                if t.ndim == 1:
                    return t
                return t.view(-1, t.shape[-1]).mean(dim=0)
            
            norms = torch.linalg.vector_norm(t, dim=-1)
            is_active_t5 = norms > 1e-4
            
            last_token = t[-1]
            diffs = torch.linalg.vector_norm(t - last_token, dim=-1)
            valid_idx = torch.where(diffs > 1e-4)[0]
            
            active_len = valid_idx.max().item() + 2 if valid_idx.numel() > 0 else t.shape[0] // 2
            active_len = min(active_len, t.shape[0])
            
            active_mask = torch.zeros(t.shape[0], dtype=torch.bool, device=t.device)
            active_mask[:active_len] = True
            active_mask = active_mask & is_active_t5
            active_mask[0] = True 
            
            active_tokens = t[active_mask]
            return active_tokens.mean(dim=0) if active_tokens.shape[0] > 0 else t.mean(dim=0)

        from modules import prompt_parser
        steps = getattr(p, "steps", 20)
        
        try:
            schedules = prompt_parser.get_learned_conditioning_prompt_schedules([base_prompt], steps)[0]
        except Exception as e:
            print(f"[Semantic Shifting] Error parsing prompt schedule: {e}")
            schedules = [[steps, base_prompt]]

        quality_vec_schedule = []
        for end_step, prompt_str in schedules:
            # Reconstruct the contextualized prompts for each step in the schedule
            contextualized_pos_prompt = f"{pos_prompt}, {prompt_str}" if prompt_str else pos_prompt
            contextualized_neg_prompt = f"{neg_prompt}, {prompt_str}" if prompt_str else neg_prompt

            pos_tensor = encode_string(p.sd_model, contextualized_pos_prompt)
            neg_tensor = encode_string(p.sd_model, contextualized_neg_prompt)

            if pos_tensor is None or neg_tensor is None:
                continue

            pos_mean = get_active_mean(pos_tensor)
            neg_mean = get_active_mean(neg_tensor)
            
            quality_vec_schedule.append((end_step, (pos_mean - neg_mean).detach()))

        if not quality_vec_schedule:
            print("[Semantic Shifting] Error encoding prompts. Skipping modulation.")
            return

        # Cache the schedule of vectors on the processing object so the step-hook can grab them instantly
        p.semantic_shifting_quality_schedule = quality_vec_schedule


    # --- Wrapper Hook: Now lightweight and fast ---
    def process_before_every_sampling(self, p, enable: bool, pos_prompt: str, neg_prompt: str, shift_mode: str, weight: float, rescale_phi: float=0.5, *args, **kwargs):
        if not enable or weight <= 0:
            return

        quality_schedule = getattr(p, "semantic_shifting_quality_schedule", None)
        unet = p.sd_model.forge_objects.unet

        if not quality_schedule or unet is None:
            return

        def semantic_shifting_wrapper(model_function, kwargs):
            from modules import shared
            current_step = getattr(shared.state, "sampling_step", 0) + 1
            
            quality_vec = quality_schedule[-1][1]
            for end_step, q_vec in quality_schedule:
                if current_step <= end_step:
                    quality_vec = q_vec
                    break
            
            # Ensure vector is on the correct UNet device for this step
            quality_vec = quality_vec.to(unet.current_device)

            c_kwargs = kwargs.get("c", {}).copy()
            c_crossattn = c_kwargs.get("c_crossattn", None)

            if c_crossattn is not None:
                B, Seq, Dim = c_crossattn.shape
                c_crossattn_new = c_crossattn.clone()
                
                # FIX: Robust Batch Indexing
                cond_idx = B // 2 if B >= 2 else 0
                    
                target_chunk = c_crossattn_new[cond_idx:]
                shifted_target_chunk = target_chunk.clone()
                
                attn_mask = c_kwargs.get("c_crossattn_mask", c_kwargs.get("attention_mask", None))
                
                for b_idx in range(target_chunk.shape[0]):
                    batch_item = target_chunk[b_idx]
                    
                    if attn_mask is not None:
                        b_mask = attn_mask[cond_idx + b_idx] if cond_idx + b_idx < attn_mask.shape[0] else attn_mask[0]
                        active_mask = b_mask > 0.5
                        if active_mask.dim() == 2:
                            active_mask = active_mask.squeeze(-1)
                    else:
                        token_norms = torch.linalg.vector_norm(batch_item, dim=-1)
                        is_active_t5 = token_norms > 1e-4

                        last_token = batch_item[-1]
                        diffs = torch.linalg.vector_norm(batch_item - last_token, dim=-1)
                        valid_idx = torch.where(diffs > 1e-4)[0]
                        
                        if valid_idx.numel() > 0:
                            active_len = valid_idx.max().item() + 2
                        else:
                            active_len = Seq // 2
                            
                        active_len = min(active_len, Seq)
                        active_mask = torch.zeros(Seq, dtype=torch.bool, device=batch_item.device)
                        active_mask[:active_len] = True
                        active_mask = active_mask & is_active_t5
                    
                    active_mask[0] = True 
                    
                    if active_mask.sum() == 0:
                        active_mask[:] = True
                        
                    active_tokens = batch_item[active_mask]
                    
                    # Float32 & Device Matching Cast
                    active_f32 = active_tokens.to(torch.float32)
                    
                    if quality_vec.ndim > 1:
                        quality_f32 = quality_vec.view(-1).to(dtype=torch.float32, device=active_f32.device)
                    else:
                        quality_f32 = quality_vec.to(dtype=torch.float32, device=active_f32.device)
                    
                    orig_norm = torch.linalg.vector_norm(active_f32, dim=-1, keepdim=True)
                    avg_token_norm = orig_norm.mean()
                    shift_dir = torch.nn.functional.normalize(quality_f32, dim=-1, eps=1e-5)
                    
                    # Core Shift Application based on UI Mode
                    if "Orthogonal" in shift_mode:
                        scaled_shift = shift_dir * (avg_token_norm * 0.1) 
                        prompt_mean = active_f32.mean(dim=0, keepdim=True)
                        
                        dot_product = (scaled_shift * prompt_mean).sum(dim=-1, keepdim=True)
                        prompt_sq_norm = (prompt_mean * prompt_mean).sum(dim=-1, keepdim=True) + 1e-5
                        projection = (dot_product / prompt_sq_norm) * prompt_mean
                        
                        final_shift = scaled_shift - projection.squeeze(0)
                    else:
                        # Additive Mode: Higher base energy, no orthogonal projection
                        scaled_shift = shift_dir * (avg_token_norm * 0.3)
                        final_shift = scaled_shift
                        
                    shifted_active_f32 = active_f32 + (weight * final_shift)
                    
                    # Clamp & Rescale
                    new_norm = torch.linalg.vector_norm(shifted_active_f32, dim=-1, keepdim=True)
                    safe_rescale_factor = torch.clamp(orig_norm / (new_norm + 1e-5), min=0.5, max=2.5)
                    
                    rescaled_active_f32 = shifted_active_f32 * safe_rescale_factor
                    blended_active_f32 = rescaled_active_f32 * rescale_phi + shifted_active_f32 * (1.0 - rescale_phi)
                    
                    shifted_target_chunk[b_idx, active_mask] = blended_active_f32.to(target_chunk.dtype)

                c_crossattn_new[cond_idx:] = shifted_target_chunk
                c_kwargs["c_crossattn"] = c_crossattn_new

            # FIX: Updated Wrapper Return Signature
            kwargs["c"] = c_kwargs
            return model_function(kwargs.get("input"), kwargs.get("timestep"), **kwargs.get("c"))

        new_unet = unet.clone()
        new_unet.set_model_unet_function_wrapper(semantic_shifting_wrapper)
        p.sd_model.forge_objects.unet = new_unet
