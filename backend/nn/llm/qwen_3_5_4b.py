import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from backend.attention import attention_function


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight.to(device=x.device, dtype=torch.float32) * x_normed).to(orig_dtype)


class ExpRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (torch.exp(self.weight.to(device=x.device, dtype=torch.float32)) * x_normed).to(orig_dtype)


class SSMBlock(nn.Module):
    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32,
                 d_gate=4096, conv_kernel=4, norm_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_inner = d_inner
        self.n_groups = n_groups
        self.d_ssm = d_gate
        self.head_dim = d_gate // n_groups
        self.d_state = (d_inner - d_gate) // (2 * n_groups)

        self.in_proj_qkv = nn.Linear(hidden_size, d_inner, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, d_gate, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, n_groups, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, n_groups, bias=False)

        self.conv1d = nn.Conv1d(
            d_inner, d_inner, conv_kernel, groups=d_inner,
            padding=conv_kernel - 1, bias=False
        )

        self.out_proj = nn.Linear(d_gate, hidden_size, bias=False)
        self.norm = RMSNorm(norm_dim)

        self.A_log = nn.Parameter(torch.zeros(n_groups))
        self.dt_bias = nn.Parameter(torch.zeros(n_groups))

    def _ssm_scan(self, x, B_state, C_state, dt_input, D_input):
        batch, seq_len, nheads, head_dim = x.shape
        d_state = B_state.shape[-1]
        device = x.device
        compute_dtype = torch.float32

        A = -torch.exp(self.A_log.to(device=device).float())
        dt_bias = self.dt_bias.to(device=device).float()

        h = torch.zeros(batch, nheads, head_dim, d_state, device=device, dtype=compute_dtype)
        outputs = []

        x_f = x.float()
        B_f = B_state.float()
        C_f = C_state.float()
        dt_f = dt_input.float()
        D_f = D_input.float()

        for t in range(seq_len):
            x_t = x_f[:, t]
            B_t = B_f[:, t]
            C_t = C_f[:, t]
            dt_t = F.softplus(dt_f[:, t] + dt_bias)
            dA_t = torch.exp(dt_t * A.unsqueeze(0))
            dt_expanded = dt_t.unsqueeze(-1).unsqueeze(-1)
            dBx = dt_expanded * torch.einsum('bnh,bns->bnhs', x_t, B_t)
            h = dA_t.unsqueeze(-1).unsqueeze(-1) * h + dBx
            y_t = torch.einsum('bnhs,bns->bnh', h, C_t)
            y_t = y_t + D_f[:, t].unsqueeze(-1) * x_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(x.dtype)

    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape
        z = self.in_proj_z(hidden_states)
        xBC = self.in_proj_qkv(hidden_states)
        dt_input = self.in_proj_b(hidden_states)
        D_input = self.in_proj_a(hidden_states)

        xBC_conv = xBC.transpose(1, 2)
        xBC_conv = self.conv1d(xBC_conv)[..., :seq_len]
        xBC_conv = F.silu(xBC_conv.transpose(1, 2))

        x, B_conv, C_conv = torch.split(
            xBC_conv,
            [self.d_ssm, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1
        )

        x = x.reshape(batch, seq_len, self.n_groups, self.head_dim)
        B_state = B_conv.reshape(batch, seq_len, self.n_groups, self.d_state)
        C_state = C_conv.reshape(batch, seq_len, self.n_groups, self.d_state)

        y = self._ssm_scan(x, B_state, C_state, dt_input, D_input)
        y = self.norm(y)
        y = y.reshape(batch, seq_len, -1)
        y = y * F.silu(z)
        return self.out_proj(y)


class GatedSelfAttention(nn.Module):
    def __init__(self, hidden_size=2560, num_heads=16, num_kv_heads=4,
                 head_dim=256, rope_theta=1000000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_heads // num_kv_heads
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, 2 * self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        B, L, _ = hidden_states.shape
        qg = self.q_proj(hidden_states)
        q, gate = qg.chunk(2, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = self._apply_rotary_emb(q, cos, sin)
            k = self._apply_rotary_emb(k, cos, sin)

        k = k.repeat_interleave(self.gqa_ratio, dim=1)
        v = v.repeat_interleave(self.gqa_ratio, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None)
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.inner_dim)
        attn_out = attn_out * F.silu(gate)
        return self.o_proj(attn_out)

    def _apply_rotary_emb(self, x, cos, sin):
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


class MLP(nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=9216):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 use_ssm=True, has_mlp=True):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp
        self.input_layernorm = RMSNorm(hidden_size)

        if use_ssm:
            self.linear_attn = SSMBlock(hidden_size=hidden_size)
        else:
            self.self_attn = GatedSelfAttention(hidden_size=hidden_size)

        if has_mlp:
            self.post_attention_layernorm = RMSNorm(hidden_size)
            self.mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, x, attention_mask=None, freqs_cis=None):
        residual = x
        x_norm = self.input_layernorm(x)

        if self.use_ssm:
            x = residual + self.linear_attn(x_norm)
        else:
            x = residual + self.self_attn(x_norm, attention_mask=attention_mask, freqs_cis=freqs_cis)

        if self.has_mlp:
            residual = x
            x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, max_grid_size):
        seq = torch.arange(max_grid_size, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def _rotate_half_vision(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half_vision(q) * sin)
    k_embed = (k * cos) + (_rotate_half_vision(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class ViTAttention(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states, position_embeddings=None):
        if hidden_states.dim() == 2:
            seq_len, _ = hidden_states.shape
            is_flat = True
        else:
            is_flat = False
            B, seq_len, _ = hidden_states.shape

        qkv = self.qkv(hidden_states)
        if is_flat:
            qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        else:
            qkv = qkv.reshape(B, seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        if is_flat:
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        if is_flat:
            attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        else:
            attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, -1)
        return self.proj(attn_out)


class ViTMLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class ViTBlock(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096, num_heads=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = ViTAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.mlp = ViTMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states, position_embeddings=None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), position_embeddings=position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class ViTPatchMerger(nn.Module):
    def __init__(self, hidden_size=1024, out_hidden_size=2560):
        super().__init__()
        self.hidden_size = hidden_size * 4
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, self.hidden_size)
        x = self.linear_fc2(self.act(self.linear_fc1(x)))
        return x


class Qwen35ViT(nn.Module):
    NUM_BLOCKS = 24
    HIDDEN_SIZE = 1024
    INTERMEDIATE_SIZE = 4096
    NUM_HEADS = 16
    HEAD_DIM = 64
    PATCH_SIZE = 16
    TEMPORAL_PATCH_SIZE = 2
    SPATIAL_MERGE_SIZE = 2
    OUT_HIDDEN_SIZE = 2560
    NUM_GRID_PER_SIDE = 48
    NUM_POSITION_EMBEDDINGS = 2304
    ROPE_THETA = 10000.0
    IN_CHANNELS = 3

    def __init__(self):
        super().__init__()
        self.patch_embed_proj = nn.Conv3d(
            self.IN_CHANNELS, self.HIDDEN_SIZE,
            kernel_size=(self.TEMPORAL_PATCH_SIZE, self.PATCH_SIZE, self.PATCH_SIZE),
            stride=(self.TEMPORAL_PATCH_SIZE, self.PATCH_SIZE, self.PATCH_SIZE),
            bias=True
        )

        self.pos_embed = nn.Embedding(self.NUM_POSITION_EMBEDDINGS, self.HIDDEN_SIZE)
        self.rotary_pos_emb = VisionRotaryEmbedding(self.HEAD_DIM // 2, theta=self.ROPE_THETA)

        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                num_heads=self.NUM_HEADS
            )
            for _ in range(self.NUM_BLOCKS)
        ])

        self.merger = ViTPatchMerger(
            hidden_size=self.HIDDEN_SIZE,
            out_hidden_size=self.OUT_HIDDEN_SIZE
        )

    def _compute_position_embeddings(self, h_patches, w_patches, device, dtype):
        merge = self.SPATIAL_MERGE_SIZE
        h_blocks = h_patches // merge
        w_blocks = w_patches // merge

        h_idxs = torch.linspace(0, self.NUM_GRID_PER_SIDE - 1, h_patches, device=device)
        w_idxs = torch.linspace(0, self.NUM_GRID_PER_SIDE - 1, w_patches, device=device)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clamp(max=self.NUM_GRID_PER_SIDE - 1)
        w_ceil = (w_floor + 1).clamp(max=self.NUM_GRID_PER_SIDE - 1)

        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()

        base_h = h_floor * self.NUM_GRID_PER_SIDE
        base_h_ceil = h_ceil * self.NUM_GRID_PER_SIDE

        idx_00 = (base_h[None].T + w_floor[None]).flatten()
        idx_01 = (base_h[None].T + w_ceil[None]).flatten()
        idx_10 = (base_h_ceil[None].T + w_floor[None]).flatten()
        idx_11 = (base_h_ceil[None].T + w_ceil[None]).flatten()

        w_00 = ((1 - dh)[None].T * (1 - dw)[None]).flatten()
        w_01 = ((1 - dh)[None].T * dw[None]).flatten()
        w_10 = (dh[None].T * (1 - dw)[None]).flatten()
        w_11 = (dh[None].T * dw[None]).flatten()

        idx_all = torch.stack([idx_00, idx_01, idx_10, idx_11]).long()
        w_all = torch.stack([w_00, w_01, w_10, w_11]).to(dtype=dtype)

        all_embeds = self.pos_embed(idx_all.to(device))
        pos_embed = (all_embeds * w_all.unsqueeze(-1)).sum(0)

        pos_embed = pos_embed.view(h_patches, w_patches, -1)
        pos_embed = pos_embed.view(h_blocks, merge, w_blocks, merge, -1)
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4)
        pos_embed = pos_embed.reshape(-1, self.HIDDEN_SIZE)

        max_hw = max(h_patches, w_patches)
        freq_table = self.rotary_pos_emb(max_hw)

        block_rows = torch.arange(h_blocks, device=device)
        block_cols = torch.arange(w_blocks, device=device)
        intra_row = torch.arange(merge, device=device)
        intra_col = torch.arange(merge, device=device)

        row_idx = block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge + intra_col[None, None, None, :]

        row_idx = row_idx.expand(h_blocks, w_blocks, merge, merge).reshape(-1)
        col_idx = col_idx.expand(h_blocks, w_blocks, merge, merge).reshape(-1)

        row_freqs = freq_table[row_idx]
        col_freqs = freq_table[col_idx]
        emb = torch.cat([row_freqs, col_freqs], dim=-1)
        emb = torch.cat([emb, emb], dim=-1)

        rope_cos = emb.cos().to(dtype=dtype)
        rope_sin = emb.sin().to(dtype=dtype)

        return pos_embed, (rope_cos, rope_sin)

    def _preprocess_image(self, image, target_size=None):
        x = image.permute(0, 3, 1, 2).float()
        B = x.shape[0]

        if target_size is not None:
            th, tw = target_size
        else:
            th = max(32, (x.shape[2] + 15) // 32 * 32)
            tw = max(32, (x.shape[3] + 15) // 32 * 32)

        if x.shape[2] != th or x.shape[3] != tw:
            x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False)
            x = x.clamp(0, 1)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        h_patches = th // self.PATCH_SIZE
        w_patches = tw // self.PATCH_SIZE
        merge = self.SPATIAL_MERGE_SIZE
        h_blocks = h_patches // merge
        w_blocks = w_patches // merge

        x = x.view(B, 3, 2, h_patches, self.PATCH_SIZE, w_patches, self.PATCH_SIZE)
        x = x.view(B, 3, 2, h_blocks, merge, self.PATCH_SIZE, w_blocks, merge, self.PATCH_SIZE)
        x = x.permute(0, 3, 6, 4, 7, 1, 2, 5, 8)
        patches = x.reshape(-1, 3, 2, self.PATCH_SIZE, self.PATCH_SIZE)
        return patches, h_patches, w_patches, B

    def forward(self, image, target_size=None):
        patches, h_patches, w_patches, B = self._preprocess_image(image, target_size)
        target_dtype = self.patch_embed_proj.weight.dtype
        patches = patches.to(dtype=target_dtype)
        hidden_states = self.patch_embed_proj(patches).view(-1, self.HIDDEN_SIZE)
        pos_embeds, rope_cos_sin = self._compute_position_embeddings(
            h_patches, w_patches, hidden_states.device, hidden_states.dtype
        )
        n_patches = h_patches * w_patches
        pos_embeds = pos_embeds.repeat(B, 1)
        hidden_states = hidden_states + pos_embeds
        cos, sin = rope_cos_sin
        cos = cos.repeat(B, 1)
        sin = sin.repeat(B, 1)
        for block in self.blocks:
            hidden_states = block(hidden_states, position_embeddings=(cos, sin))
        merged = self.merger(hidden_states)
        n_merged = n_patches // (self.SPATIAL_MERGE_SIZE ** 2)
        merged = merged.view(B, n_merged, self.OUT_HIDDEN_SIZE)
        return merged


class Qwen35HybridModel(nn.Module):
    SELF_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE = 248320
    OUTPUT_DIM = 1024
    HEAD_DIM = 256
    ROPE_THETA = 1000000.0

    def __init__(self, config_dict=None):
        super().__init__()
        self.num_layers = self.NUM_LAYERS
        self.embed_tokens = nn.Embedding(self.VOCAB_SIZE, self.HIDDEN_SIZE)
        self.layers = nn.ModuleList()
        for i in range(self.NUM_LAYERS):
            use_ssm = (i not in self.SELF_ATTN_LAYERS)
            has_mlp = (i != 31)
            self.layers.append(HybridBlock(
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                use_ssm=use_ssm,
                has_mlp=has_mlp
            ))

        self.norm = nn.Sequential(
            nn.Linear(self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True),
            ExpRMSNorm(self.OUTPUT_DIM),
            nn.SiLU(),
            nn.Linear(self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True),
        )

        self._output_scale = 1.0
        self._alignment_strength = 1.0

        self._calibration_scale = None
        self._calibration_bias = None
        self._rotation_matrix = None
        self._rotation_mean_4b = None
        self._rotation_mean_06b = None

        self._pending_visual_embeds = None
        self._pending_vision_weight = 1.0
        self._pending_vision_mode = "add"
        self.visual = Qwen35ViT()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                **kwargs):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(input_ids)

        seq_len = x.shape[1]
        freqs_cis = self._precompute_freqs_cis(self.HEAD_DIM, seq_len, theta=self.ROPE_THETA, device=x.device, dtype=x.dtype)

        attn_mask = None
        if attention_mask is not None:
            mask_fill = torch.finfo(x.dtype).min / 4
            causal = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(mask_fill).triu_(1)
            pad_mask = 1.0 - attention_mask.to(x.dtype).reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1]).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            pad_mask = pad_mask.masked_fill(pad_mask.to(torch.bool), mask_fill)
            attn_mask = causal + pad_mask
        elif seq_len > 1:
            mask_fill = torch.finfo(x.dtype).min / 4
            attn_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(mask_fill).triu_(1)

        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=attn_mask, freqs_cis=freqs_cis)
            if intermediate_output is not None:
                if isinstance(intermediate_output, int):
                    if i == intermediate_output or i == (len(self.layers) + intermediate_output):
                        intermediate = x.clone()
                elif isinstance(intermediate_output, list) and i in intermediate_output:
                    if intermediate is None:
                        intermediate = {}
                    intermediate[i] = x.clone()

        n_visual = 0
        _vis_for_post_norm = None
        if self._pending_visual_embeds is not None:
            visual = self._pending_visual_embeds.to(device=x.device, dtype=x.dtype)
            n_visual = visual.shape[1]
            mode = self._pending_vision_mode
            weight = self._pending_vision_weight
            if mode == "add":
                style_vec_2560 = visual.mean(dim=1, keepdim=True)
                text_scale = x.norm(dim=-1).mean().clamp(min=1e-6)
                style_scale = style_vec_2560.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                style_vec_2560 = style_vec_2560 * (text_scale / style_scale)
                x = x + weight * style_vec_2560
                n_visual = 0
            else:
                _vis_for_post_norm = (visual, n_visual, mode, weight)

        x = self.norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            if isinstance(intermediate, dict):
                intermediate = {k: self.norm(v) for k, v in intermediate.items()}
            else:
                intermediate = self.norm(intermediate)

        from modules.shared import opts
        use_alignment = getattr(opts, "anima_qwen35_use_alignment", False)
        use_calibration = getattr(opts, "anima_qwen35_use_calibration", True)
        alpha = getattr(opts, "anima_qwen35_alignment_strength", 0.0)
        output_scale = getattr(opts, "anima_qwen35_output_scale", 1.0)

        # FP32 math for alignment/calibration
        x_f32 = x.float()

        applied_alignment = False
        if use_alignment and self._rotation_matrix is not None:
            R = self._rotation_matrix.to(device=x.device, dtype=torch.float32)
            m4b = self._rotation_mean_4b.to(device=x.device, dtype=torch.float32)
            m06b = self._rotation_mean_06b.to(device=x.device, dtype=torch.float32)
            
            # Always rotate relative to m4b (the source distribution center)
            # We subtract m4b to center it at 0, rotate, then add a mean back.
            x_f32 = torch.einsum('ij,...j->...i', R, x_f32 - m4b)
            
            if use_calibration:
                # If we are going to calibrate, we stay at m4b center.
                # Calibration expects m4b-centered input to move it to m06b.
                x_f32 = x_f32 + m4b
            else:
                # Standard Rotate + Blend Mean (if no calibration)
                x_f32 = x_f32 + (1.0 - alpha) * m4b + alpha * m06b
            applied_alignment = True

        # Calibration Logic
        applied_calibration = False
        if use_calibration and self._calibration_scale is not None:
            cal_scale = self._calibration_scale.to(device=x.device, dtype=torch.float32)
            cal_bias = self._calibration_bias.to(device=x.device, dtype=torch.float32)
            x_f32 = x_f32 * cal_scale + cal_bias
            applied_calibration = True

        if output_scale != 1.0:
            x_f32 = x_f32 * output_scale

        x = x_f32.to(x.dtype)

        if _vis_for_post_norm is not None:
            visual, n_visual, mode, weight = _vis_for_post_norm
            visual_projected = self.norm(visual)
            if applied_alignment:
                R = self._rotation_matrix.to(device=visual_projected.device, dtype=visual_projected.dtype)
                m4b = self._rotation_mean_4b.to(device=visual_projected.device, dtype=visual_projected.dtype)
                m06b = self._rotation_mean_06b.to(device=visual_projected.device, dtype=visual_projected.dtype)
                vp_rotated = torch.einsum('ij,...j->...i', R, visual_projected - m4b)
                visual_projected = vp_rotated + (1.0 - alpha) * m4b + alpha * m06b
            if mode == "concat":
                if weight != 1.0:
                    visual_projected = visual_projected * weight
                x = torch.cat([visual_projected, x], dim=1)
            elif mode == "replace_padding":
                B, T, D = x.shape
                tok_norms = x.norm(dim=-1)
                non_pad = (tok_norms[0] > 1.0).nonzero(as_tuple=True)[0]
                first_pad = (non_pad[-1].item() + 1) if len(non_pad) > 0 else 0
                n_pad_slots = T - first_pad
                if n_pad_slots > 0:
                    if weight != 1.0:
                        visual_projected = visual_projected * weight
                    if n_visual <= n_pad_slots:
                        x[:, first_pad:first_pad + n_visual, :] = visual_projected
                    else:
                        chunk_size = n_visual // n_pad_slots
                        for s in range(n_pad_slots):
                            start = s * chunk_size
                            end = min(start + chunk_size, n_visual) if s < n_pad_slots - 1 else n_visual
                            x[:, first_pad + s, :] = visual_projected[:, start:end, :].mean(dim=1)
                n_visual = 0
        self._last_n_visual = n_visual
        return x, intermediate

    def _precompute_freqs_cis(self, head_dim, max_seq_len, theta=1000000.0, device=None, dtype=None):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin

    def load_extra_params(self, model_path: str):
        import os
        import safetensors.torch as safetensors_torch
        dir_name = os.path.dirname(model_path)
        cal_path = os.path.join(dir_name, "calibration_params.safetensors")
        if os.path.exists(cal_path):
            try:
                cal = safetensors_torch.load_file(cal_path, device="cpu")
                self._calibration_scale = cal["scale"].float()
                self._calibration_bias = cal["bias"].float()
                print(f"[Qwen3.5-Anima] Loaded calibration from {cal_path}")
            except Exception as e:
                print(f"[Qwen3.5-Anima] Failed to load calibration: {e}")
        rot_path = os.path.join(dir_name, "rotation_matrix.safetensors")
        if os.path.exists(rot_path):
            try:
                data = safetensors_torch.load_file(rot_path, device="cpu")
                self._rotation_matrix = data["rotation"].float()
                self._rotation_mean_4b = data["mean_4b"].float()
                self._rotation_mean_06b = data["mean_06b"].float()
                print(f"[Qwen3.5-Anima] Loaded alignment from {rot_path}")
            except Exception as e:
                print(f"[Qwen3.5-Anima] Failed to load alignment: {e}")
