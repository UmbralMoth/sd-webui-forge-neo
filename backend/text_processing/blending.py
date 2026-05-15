
import torch
import math

def pad_tensors(tensors):
    """
    Pads a list of tensors to match the maximum sequence length using 'repeat last' strategy.
    """
    if not tensors: return tensors
    if len(tensors) == 1: return tensors
    
    seq_dim = 1 if tensors[0].ndim == 3 else 0
    max_len = max(t.shape[seq_dim] for t in tensors)
    
    res = []
    for t in tensors:
        if t.shape[seq_dim] < max_len:
            last_vector = t.select(seq_dim, -1).unsqueeze(seq_dim)
            pad_shape = list(t.shape)
            pad_shape[seq_dim] = max_len - t.shape[seq_dim]
            t = torch.cat([t, last_vector.expand(pad_shape)], dim=seq_dim)
        res.append(t)
    return res

def n_way_blend(tensors, weights=None, alpha=0.0, preserve_magnitude=True, eps=1e-8):
    """
    Superior Semantic Blending (Magnitude-Renormalized).
    Optimized for memory efficiency to prevent OOMs on large visual tensors.
    """
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    count = len(tensors)
    if weights is None:
        weights = [1.0] * count
        
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Memory Optimization: Only upcast small tensors (embeddings).
    is_large = tensors[0].numel() > 1000000 
    calc_dtype = torch.float32 if not is_large else tensors[0].dtype
    orig_dtype = tensors[0].dtype
    
    # 1. Padding handling
    if not is_large:
        tensors = pad_tensors(tensors)
    
    # 2. Energy Calculation
    # We calculate norms in float32 for precision and to avoid overflow/underflow in fp16.
    norms = [torch.linalg.vector_norm(t.to(torch.float32), dim=-1, keepdim=True) for t in tensors]
    
    # 3. Magnitude Renormalization weights
    # Stable handling of zero-magnitude vectors (common in deltas)
    # Vectors with norm < eps are ignored in direction fusion but included in magnitude average.
    eps_t = torch.tensor(eps, dtype=torch.float32, device=tensors[0].device)
    valid_masks = [n > eps_t for n in norms]
    
    # Force weight math to float32 to prevent NaNs on small fp16 values
    n_weights = [torch.tensor(w, dtype=torch.float32, device=tensors[0].device) for w in normalized_weights]
    
    effective_weights = [nw / torch.maximum(n, eps_t) for nw, n in zip(n_weights, norms)]
    
    # Use torch.where instead of multiplication to avoid NaN when multiplying inf by 0.0
    zero_weights = torch.zeros_like(effective_weights[0])
    masked_eff = [torch.where(m, ew, zero_weights) for ew, m in zip(effective_weights, valid_masks)]
    total_eff = sum(masked_eff)
    
    # Safety: if all vectors are zero, default to arithmetic mean of zeros (0)
    total_eff = torch.maximum(total_eff, eps_t)
    renorm_weights = [torch.where(m, ew / total_eff, zero_weights) for ew, m in zip(effective_weights, valid_masks)]
    
    # 4. Arithmetic Weighted Sum (Fusion)
    # Convert weights back to calc_dtype for the main fusion to save memory/speed
    blended_sum = torch.zeros_like(tensors[0], dtype=calc_dtype)
    for i in range(count):
        blended_sum += renorm_weights[i].to(calc_dtype) * tensors[i].to(calc_dtype)
        
    # To save memory on large tensors, we compute the norm in float32 but avoid 
    # creating a full float32 copy of the results for the division.
    blended_sum_n = torch.linalg.vector_norm(blended_sum.to(torch.float32), dim=-1, keepdim=True).clamp(min=1e-12)
    
    # inv_n is a small tensor (1 value per token/feature), casting it to calc_dtype 
    # allows the massive multiplication to happen in fp16/bf16 without a large float32 copy.
    inv_n = (1.0 / blended_sum_n).to(calc_dtype)
    blended_sum_dir = blended_sum * inv_n
    
    # 5. Target Magnitude
    # ...
    if alpha > 0.0:
        log_gm = sum(nw * torch.log(torch.maximum(n, eps_t)) for nw, n in zip(n_weights, norms))
        gm = torch.exp(log_gm)
        mean_norm = sum(nw * n for nw, n in zip(n_weights, norms))
        target_norm = torch.lerp(mean_norm, gm, alpha)
    else:
        target_norm = sum(nw * n for nw, n in zip(n_weights, norms))
        
    # 6. Magnitude Normalization (The "Kick" Fix)
    if not preserve_magnitude:
        target_norm = target_norm * blended_sum_n
    
    # 7. Final Synthesis
    # target_norm is small, casting it for the final scaling keeps everything efficient.
    res = blended_sum_dir * target_norm.to(calc_dtype)
    return res.to(orig_dtype)

def blend_layered_conds(conds, weights=None, alpha=0.0, preserve_magnitude=True):
    """
    Recursively blend dictionaries or complex conditioning objects.
    """
    if not conds: return None
    if len(conds) == 1: return conds[0]
    
    first = conds[0]
    if isinstance(first, dict):
        res = {}
        all_keys = set().union(*(c.keys() for c in conds if isinstance(c, dict)))
        for k in all_keys:
            subset = [c[k] for c in conds if isinstance(c, dict) and k in c]
            if subset and isinstance(subset[0], torch.Tensor):
                res[k] = n_way_blend(subset, weights, alpha, preserve_magnitude=preserve_magnitude)
            else:
                res[k] = subset[0] if subset else None
        return res
    elif isinstance(first, torch.Tensor):
        return n_way_blend(conds, weights, alpha, preserve_magnitude=preserve_magnitude)
    
    return first
def surgical_delta_blend(baseline, variants, weights, slot_mask, alpha=0.0, eps=1e-8):
    """
    Surgical Delta Grafting.
    - Slot indices (mask=True): Direct high-intensity blending.
    - Static indices (mask=False): Orthogonal Contextual Delta Injection.
    """
    # For 1D/2D pooled vectors, delta grafting is inappropriate as there is no 'local' context.
    # We use direct n_way_blend for global style vectors. We MUST preserve magnitude to prevent SDXL AdaLN NaNs.
    if baseline.ndim < 3:
        return n_way_blend(variants, weights, alpha=alpha, preserve_magnitude=True, eps=eps)

    orig_dtype = baseline.dtype
    b_fp32 = baseline.to(torch.float32)
    v_fp32_list = [v.to(torch.float32) for v in variants]
    
    # 1. Generate Direct Blend
    # We disable preserve_magnitude to allow natural 'slump' (cancellation) for divergent concepts.
    direct_blend = n_way_blend(v_fp32_list, weights, alpha=alpha, preserve_magnitude=False, eps=eps)

    # 2. Generate Contextual Shift (The "Filtered Velocity")
    deltas = []
    for v_fp32 in v_fp32_list:
        v_dot_b = (v_fp32 * b_fp32).sum(dim=-1, keepdim=True)
        b_dot_b = (b_fp32 * b_fp32).sum(dim=-1, keepdim=True).clamp(min=eps)
        proj_scalar = v_dot_b / b_dot_b
        # Delta is the orthogonal component relative to baseline
        deltas.append(v_fp32 - (proj_scalar * b_fp32))

    # We blend deltas linearly-ish (preserve_magnitude=False) to ensure 
    # that 'no shift' (0) properly averages with 'full shift'.
    blended_delta = n_way_blend(deltas, weights, alpha=alpha, preserve_magnitude=False, eps=eps)
    contextual_graft = b_fp32 + blended_delta
    
    # 3. Surgical Recombination
    # We only use the slot mask if the tensor has a sequence dimension (3D: [B, S, D])
    # Pooled vectors (2D: [B, D]) get the global contextual graft.
    if b_fp32.ndim == 3 and slot_mask is not None:
        mask = slot_mask.to(device=b_fp32.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        
        # Ensure mask matches sequence length (important for multi-chunk prompts)
        if mask.shape[1] != b_fp32.shape[1]:
            # If the mask is shorter than the sequence, pad it with False
            new_mask = torch.zeros_like(b_fp32[:, :, :1], dtype=torch.bool)
            m_len = min(mask.shape[1], b_fp32.shape[1])
            new_mask[:, :m_len, :] = mask[:, :m_len, :]
            mask = new_mask
            
        # Ensure mask is on the same device as the tensors
        mask = mask.to(device=b_fp32.device)
            
        final_res = torch.where(mask, direct_blend, contextual_graft)
    else:
        # For 2D pooled vectors, the 'delta' is always global
        final_res = contextual_graft
    
    # 4. Magnitude Restoration (Velocity Matching)
    # We ensure the post-blend magnitude matches the weighted average of variants.
    # Simple arithmetic mean of norms is more stable here than n_way_blend.
    v_norms = [torch.linalg.vector_norm(v.to(torch.float32), dim=-1, keepdim=True) for v in v_fp32_list]
    total_w = sum(weights)
    target_norm = sum(((torch.tensor(w, dtype=torch.float32, device=b_fp32.device) / total_w) * n) for w, n in zip(weights, v_norms))
    
    res_n = torch.linalg.vector_norm(final_res.to(torch.float32), dim=-1, keepdim=True).clamp(min=eps)
    final_res = (final_res.to(torch.float32) / res_n) * target_norm.to(torch.float32)
    
    return final_res.to(orig_dtype)

def surgical_delta_blend_conds(baseline, variants, weights, slot_mask, alpha=0.0):
    """
    Recursively applies Surgical Delta Grafting.
    """
    if isinstance(baseline, dict):
        res = {}
        for k in baseline.keys():
            v_subset = [v[k] for v in variants if isinstance(v, dict) and k in v]
            if isinstance(baseline[k], torch.Tensor) and len(v_subset) == len(variants):
                res[k] = surgical_delta_blend(baseline[k], v_subset, weights, slot_mask, alpha)
            else:
                res[k] = baseline[k]
        return res
    elif isinstance(baseline, torch.Tensor):
        return surgical_delta_blend(baseline, variants, weights, slot_mask, alpha)
    return baseline
