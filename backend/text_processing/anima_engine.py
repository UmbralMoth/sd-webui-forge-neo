from typing import TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from backend.nn.llm.llama import Qwen3_06B
import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.qwen_tokens = []
        self.qwen_multipliers = []
        self.t5_tokens = []
        self.t5_multipliers = []


class AnimaTextProcessingEngine:
    def __init__(self, text_encoder, qwen_tokenizer, t5_tokenizer, unet=None):
        super().__init__()

        self.text_encoder = text_encoder
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer

        is_qwen35 = False
        if hasattr(text_encoder, "model") and "Qwen35HybridModel" in type(text_encoder.model).__name__:
            is_qwen35 = True

        self.id_pad = qwen_tokenizer.pad_token_id if hasattr(qwen_tokenizer, "pad_token_id") and qwen_tokenizer.pad_token_id is not None else 151643
        self.id_end = t5_tokenizer.eos_token_id if hasattr(t5_tokenizer, "eos_token_id") and t5_tokenizer.eos_token_id is not None else 1

        self.golden_vectors = {}
        if is_qwen35:
            self._load_golden_vectors()

    def _load_golden_vectors(self):
        import safetensors.torch as st
        from modules.shared import cmd_opts
        import os
        
        for fname in ["meta_embeddings_06b.safetensors", "artist_embeddings_06b.safetensors"]:
            path = os.path.join(cmd_opts.embeddings_dir, fname)
            if os.path.exists(path):
                try:
                    data = st.load_file(path, device="cpu")
                    self.golden_vectors.update(data)
                except Exception as e:
                    print(f"[Anima] Failed to load {fname}: {e}")

    def tokenize(self, texts):
        qwen_batch = []
        t5_batch = []
        
        for text in texts:
            qwen_tokens = []
            t5_tokens = []
            
            parts = text.split(',')
            for i, p in enumerate(parts):
                p_stripped = p.strip()
                if p_stripped and p_stripped in self.golden_vectors:
                    # Inject golden vector into Qwen stream
                    qwen_tokens.append({'type': 'anima_golden', 'vector': self.golden_vectors[p_stripped]})
                    # T5 processes the raw text
                    t5_tokens.extend(self.t5_tokenizer(p, truncation=False, add_special_tokens=False)["input_ids"])
                else:
                    if p:
                        qwen_tokens.extend(self.qwen_tokenizer(p, truncation=False, add_special_tokens=False)["input_ids"])
                        t5_tokens.extend(self.t5_tokenizer(p, truncation=False, add_special_tokens=False)["input_ids"])
                
                if i < len(parts) - 1:
                    qwen_tokens.extend(self.qwen_tokenizer(",", truncation=False, add_special_tokens=False)["input_ids"])
                    t5_tokens.extend(self.t5_tokenizer(",", truncation=False, add_special_tokens=False)["input_ids"])
            
            qwen_batch.append(qwen_tokens)
            t5_batch.append(t5_tokens)
            
        return qwen_batch, t5_batch

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        qwen_tokenized, t5_tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            if not chunk.qwen_tokens:
                chunk.qwen_tokens.append(self.id_pad)
                chunk.qwen_multipliers.append(1.0)

            chunk.t5_tokens.append(self.id_end)
            chunk.t5_multipliers.append(1.0)

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(qwen_tokenized, parsed):
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.qwen_tokens.append(token)
                chunk.qwen_multipliers.append(weight)
                position += 1

        for tokens, (text, weight) in zip(t5_tokenized, parsed):
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.t5_tokens.append(token)
                chunk.t5_multipliers.append(weight)
                position += 1

        if not chunks:
            next_chunk()

        return chunks

    def __call__(self, texts):
        zs, zm, ti, tw = [], [], [], []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                z, qwen_mask, chunk = cache[line]
            else:
                tokens_qwen = None
                tokens_t5 = None
                if hasattr(line, "aligned_tokens_dict") and line.aligned_tokens_dict is not None:
                    tokens_qwen = line.aligned_tokens_dict.get("qwen3_5_4b", None) or line.aligned_tokens_dict.get("qwen3_06b", None) or line.aligned_tokens_dict.get("qwen", None)
                    tokens_t5 = line.aligned_tokens_dict.get("t5xxl", None) or line.aligned_tokens_dict.get("t5", None)

                if tokens_qwen is not None and tokens_t5 is not None:
                    chunk = PromptChunk()
                    chunk.qwen_tokens = tokens_qwen if tokens_qwen else [self.id_pad]
                    chunk.qwen_multipliers = [1.0] * len(chunk.qwen_tokens)
                    chunk.t5_tokens = tokens_t5 + [self.id_end]
                    chunk.t5_multipliers = [1.0] * len(chunk.t5_tokens)
                    chunks = [chunk]
                else:
                    chunks: list[PromptChunk] = self.tokenize_line(line)

                assert len(chunks) == 1

                for chunk in chunks:
                    tokens = chunk.qwen_tokens
                    multipliers = chunk.qwen_multipliers
                    
                    if len(tokens) == 0:
                        tokens = [self.id_pad]
                        multipliers = [1.0]

                    z_batch, qwen_mask_batch = self.process_tokens([tokens], [multipliers])
                    z = z_batch[0]
                    qwen_mask = qwen_mask_batch[0]

                cache[line] = (z, qwen_mask, chunk)

            zs.append(z)
            zm.append(qwen_mask)
            ti.append(torch.tensor(chunk.t5_tokens, dtype=torch.int))
            tw.append(torch.tensor(chunk.t5_multipliers))

        def stack_with_padding(tensors, pad_value=0):
            max_len = max([t.shape[0] for t in tensors])
            out = []
            for t in tensors:
                if t.shape[0] < max_len:
                    if t.ndim == 1:
                        t = torch.cat([t, torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype, device=t.device)])
                    else:
                        t = torch.cat([t, torch.full((max_len - t.shape[0], *t.shape[1:]), pad_value, dtype=t.dtype, device=t.device)])
                out.append(t)
            return torch.stack(out)

        qwen_cond = stack_with_padding(zs)
        qwen_masks = stack_with_padding(zm, pad_value=0)
        t5_ids = stack_with_padding(ti, pad_value=0)
        t5_weights = stack_with_padding(tw, pad_value=1.0)
        t5_masks = (t5_ids != 0).long()
        
        device = memory_management.text_encoder_device()
        cross_attn = self.text_encoder.preprocess_text_embeds(
            qwen_cond.to(device=device), 
            t5_ids.to(device=device),
            target_attention_mask=t5_masks.to(device=device),
            source_attention_mask=qwen_masks.to(device=device),
        )
        if t5_weights is not None:
            if cross_attn.shape[1] == t5_weights.shape[1]:
                 cross_attn = cross_attn * t5_weights.unsqueeze(-1).to(cross_attn)

        # Select the mask that matches the cross_attn sequence length
        if cross_attn.shape[1] == qwen_masks.shape[1]:
            final_masks = qwen_masks
        else:
            final_masks = t5_masks

        if cross_attn.shape[1] < 512:
            cross_attn = torch.nn.functional.pad(cross_attn, (0, 0, 0, 512 - cross_attn.shape[1]))
            final_masks = torch.nn.functional.pad(final_masks, (0, 512 - final_masks.shape[1]))

        from backend.text_processing.blending import DictWithShape
        return DictWithShape({
            "crossattn": cross_attn.to(torch.float32),
            "crossattn_mask": final_masks.unsqueeze(-1).to(device=cross_attn.device, dtype=torch.float32),
            "vector": torch.zeros((cross_attn.shape[0], 1), dtype=torch.float32, device=cross_attn.device)
        }, shape=cross_attn.shape)

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()

        embeds_out = []
        attention_masks = []
        num_tokens = []
        embeds_info = []
        
        max_len = max(len(tokens) for tokens in batch_tokens) if len(batch_tokens) > 0 else 0

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0

            for t in tokens:
                try:
                    token = int(t)
                    attention_mask.append(0 if eos else 1)
                    tokens_temp += [token]
                    if not eos and token == self.id_pad:
                        eos = True
                except TypeError:
                    other_embeds.append((index, t))
                index += 1
            
            # Padding for variable length
            if len(tokens_temp) < max_len:
                pad_n = max_len - len(tokens_temp)
                tokens_temp += [self.id_pad] * pad_n
                attention_mask += [0] * pad_n

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            index = 0
            for o in other_embeds:
                emb, extra = self.text_encoder.preprocess_embed(o[1], device=device)
                if emb is None:
                    index += -1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]

                assert emb.shape[-1] == tokens_embed.shape[-1]
                tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                index += emb_shape - 1
                emb_type = o[1].get("type", None)
                embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count, info = self.process_embeds(batch_tokens)
        
        seq_len = embeds.size(1)
        padded_multipliers = []
        for multipliers in batch_multipliers:
            if len(multipliers) < seq_len:
                multipliers = multipliers + [1.0] * (seq_len - len(multipliers))
            else:
                multipliers = multipliers[:seq_len]
            padded_multipliers.append(multipliers)

        if len(padded_multipliers) > 0 and embeds.size(1) == len(padded_multipliers[0]):
            self.emphasis.tokens = batch_tokens
            self.emphasis.multipliers = torch.as_tensor(padded_multipliers).to(embeds)
            self.emphasis.z = embeds
            self.emphasis.after_transformers()
            embeds = self.emphasis.z

        z, _ = self.text_encoder(
            input_ids=None,
            embeds=embeds,
            attention_mask=mask,
            num_tokens=count,
            embeds_info=info
        )
        return z, mask
