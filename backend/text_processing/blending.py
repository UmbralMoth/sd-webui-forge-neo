
import torch
import math

class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)

    @property
    def shape(self):
        return self["crossattn"].shape

    def to(self, *args, **kwargs):
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                self[k] = self[k].to(*args, **kwargs)
        return self

    def advanced_indexing(self, item):
        result = {}
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                result[k] = self[k][item]
        return DictWithShape(result)


class LayeredConditioning:
    def __init__(self, in_mid_cond, out_cond):
        self.in_mid_cond = in_mid_cond
        self.out_cond = out_cond

    @property
    def shape(self):
        if isinstance(self.in_mid_cond, dict):
            return self.in_mid_cond["crossattn"].shape
        return self.in_mid_cond.shape

    @property
    def dtype(self):
        def _dtype(c):
            if isinstance(c, dict):
                # Return the dtype of the first tensor found in the dict
                for v in c.values():
                    if isinstance(v, torch.Tensor):
                        return v.dtype
                return torch.float32
            elif isinstance(c, torch.Tensor):
                return c.dtype
            return torch.float32
        return _dtype(self.in_mid_cond)

    @property
    def device(self):
        def _device(c):
            if isinstance(c, dict):
                # Return the device of the first tensor found in the dict
                for v in c.values():
                    if isinstance(v, torch.Tensor):
                        return v.device
                return torch.device("cpu")
            elif isinstance(c, torch.Tensor):
                return c.device
            return torch.device("cpu")
        return _device(self.in_mid_cond)

    def to(self, *args, **kwargs):
        def _to(c):
            if isinstance(c, dict):
                return DictWithShape({k: v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v for k, v in c.items()}, shape=getattr(c, 'shape', None))
            elif isinstance(c, torch.Tensor):
                return c.to(*args, **kwargs)
            return c
        return LayeredConditioning(_to(self.in_mid_cond), _to(self.out_cond))

    def __getitem__(self, item):
        if isinstance(item, str):
            def _get_val(c):
                if isinstance(c, dict):
                    return c.get(item, None)
                return None
            val_in_mid = _get_val(self.in_mid_cond)
            val_out = _get_val(self.out_cond)
            
            if val_in_mid is None and val_out is None:
                return None
            return LayeredConditioning(val_in_mid, val_out)

        def _get(c):
            if isinstance(c, dict):
                res = {k: v[item] if isinstance(v, torch.Tensor) else v for k, v in c.items()}
                return DictWithShape(res, shape=getattr(c, 'shape', None))
            elif isinstance(c, torch.Tensor):
                return c[item]
            return c
        return LayeredConditioning(_get(self.in_mid_cond), _get(self.out_cond))

    def advanced_indexing(self, item):
        return self.__getitem__(item)

    def repeat(self, *args, **kwargs):
        def _repeat(c):
            if isinstance(c, dict):
                res = {k: v.repeat(*args, **kwargs) if isinstance(v, torch.Tensor) else v for k, v in c.items()}
                return DictWithShape(res, shape=getattr(c, 'shape', None))
            elif isinstance(c, torch.Tensor):
                return c.repeat(*args, **kwargs)
            return c
        return LayeredConditioning(_repeat(self.in_mid_cond), _repeat(self.out_cond))

    def chunk(self, *args, **kwargs):
        def _chunk(c):
            if isinstance(c, dict):
                chunks = [{} for _ in range(args[0] if args else 1)]
                for k, v in c.items():
                    if isinstance(v, torch.Tensor):
                        for i, chunk_v in enumerate(v.chunk(*args, **kwargs)):
                            chunks[i][k] = chunk_v
                    else:
                        for i in range(len(chunks)):
                            chunks[i][k] = v
                return [DictWithShape(ch, shape=getattr(c, 'shape', None)) for ch in chunks]
            elif isinstance(c, torch.Tensor):
                return c.chunk(*args, **kwargs)
            return [c]
        
        in_mid_chunks = _chunk(self.in_mid_cond)
        out_chunks = _chunk(self.out_cond)
        return [LayeredConditioning(i, o) for i, o in zip(in_mid_chunks, out_chunks)]

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
    if isinstance(first, LayeredConditioning):
        # Recursive support for Layered Conditioning
        in_mids = [c.in_mid_cond if isinstance(c, LayeredConditioning) else c for c in conds]
        outs = [c.out_cond if isinstance(c, LayeredConditioning) else c for c in conds]
        return LayeredConditioning(
            blend_layered_conds(in_mids, weights, alpha, preserve_magnitude),
            blend_layered_conds(outs, weights, alpha, preserve_magnitude)
        )
    elif isinstance(first, dict):
        res = {}
        all_keys = set().union(*(c.keys() for c in conds if isinstance(c, dict)))
        for k in all_keys:
            subset = [c[k] for c in conds if isinstance(c, dict) and k in c]
            if subset and isinstance(subset[0], torch.Tensor):
                res[k] = n_way_blend(subset, weights, alpha, preserve_magnitude=preserve_magnitude)
            else:
                res[k] = subset[0] if subset else None
        return DictWithShape(res) if isinstance(first, DictWithShape) else res
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
    # We only use the slot mask if the tensor has a sequence dimension (2D: [S, D] or 3D: [B, S, D])
    # Pooled vectors (1D: [D] or 2D: [B, D]) get the global contextual graft.
    if b_fp32.ndim >= 2 and slot_mask is not None:
        mask = slot_mask.to(device=b_fp32.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if mask.ndim == 3 and b_fp32.ndim == 2:
            mask = mask.squeeze(0)  # Strip batch dim from mask to match b_fp32
        
        # Ensure mask matches sequence length (important for multi-chunk prompts)
        seq_dim = 1 if b_fp32.ndim == 3 else 0
        if mask.shape[seq_dim] != b_fp32.shape[seq_dim]:
            # If the mask is shorter than the sequence, pad it with False
            if b_fp32.ndim == 3:
                new_mask = torch.zeros_like(b_fp32[:, :1, :1], dtype=torch.bool).expand(-1, b_fp32.shape[1], -1).clone()
                m_len = min(mask.shape[1], b_fp32.shape[1])
                new_mask[:, :m_len, :] = mask[:, :m_len, :]
            else:
                new_mask = torch.zeros_like(b_fp32[:1, :1], dtype=torch.bool).expand(b_fp32.shape[0], -1).clone()
                m_len = min(mask.shape[0], b_fp32.shape[0])
                new_mask[:m_len, :] = mask[:m_len, :]
            mask = new_mask
            
        # Ensure mask is on the same device as the tensors
        mask = mask.to(device=b_fp32.device)
            
        final_res = torch.where(mask, direct_blend, contextual_graft)
    else:
        # For pooled vectors, the 'delta' is always global
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
    if isinstance(baseline, LayeredConditioning):
        in_mids = [v.in_mid_cond if isinstance(v, LayeredConditioning) else v for v in variants]
        outs = [v.out_cond if isinstance(v, LayeredConditioning) else v for v in variants]
        return LayeredConditioning(
            surgical_delta_blend_conds(baseline.in_mid_cond, in_mids, weights, slot_mask, alpha),
            surgical_delta_blend_conds(baseline.out_cond, outs, weights, slot_mask, alpha)
        )
    elif isinstance(baseline, dict):
        res = {}
        for k in baseline.keys():
            v_subset = [v[k] for v in variants if isinstance(v, dict) and k in v]
            if isinstance(baseline[k], torch.Tensor) and len(v_subset) == len(variants):
                res[k] = surgical_delta_blend(baseline[k], v_subset, weights, slot_mask, alpha)
            else:
                res[k] = baseline[k]
        return DictWithShape(res) if isinstance(baseline, DictWithShape) else res
    elif isinstance(baseline, torch.Tensor):
        return surgical_delta_blend(baseline, variants, weights, slot_mask, alpha)
    return baseline


def blend_chunk_attention(tensors, eps=1e-8):
    """
    Blends independent chunk attention outputs using norm-weighted magnitude renormalization.
    This prevents style-erasure and prompt strength dilution by ensuring padding/noise chunks
    do not dilute the direction or magnitude of active prompt chunks.
    """
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    calc_dtype = tensors[0].dtype
    
    # Calculate vector norms of each tensor along the channel dimension (last dimension)
    norms = [torch.linalg.vector_norm(t.to(torch.float32), dim=-1, keepdim=True) for t in tensors]
    
    # Sum of norms (clamped to avoid division by zero)
    total_norm = sum(norms)
    total_norm_clamped = torch.clamp(total_norm, min=eps)
    
    # Sum of tensors
    sum_tensors = sum(t.to(torch.float32) for t in tensors)
    sum_tensors_n = torch.linalg.vector_norm(sum_tensors, dim=-1, keepdim=True).clamp(min=eps)
    blended_sum_dir = sum_tensors / sum_tensors_n
    
    # Norm-weighted average of norms: sum(norm_i^2) / total_norm
    sum_sq_norms = sum(n * n for n in norms)
    target_norm = sum_sq_norms / total_norm_clamped
    
    # Scale the blended direction by the target norm
    res = blended_sum_dir * target_norm
    return res.to(calc_dtype)

