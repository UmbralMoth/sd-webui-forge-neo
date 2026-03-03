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

        self.text_encoder: "Qwen3_06B" = text_encoder
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer

        self.id_pad = 151643
        self.id_end = 1

    def tokenize(self, texts):
        return (
            self.qwen_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
            self.t5_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
        )

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

        for tokens in qwen_tokenized:
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.qwen_tokens.append(token)
                chunk.qwen_multipliers.append(1.0)
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
        zs, ti, tw = [], [], []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                z, chunk = cache[line]
            else:
                chunks: list[PromptChunk] = self.tokenize_line(line)
                assert len(chunks) == 1

                for chunk in chunks:
                    tokens = chunk.qwen_tokens
                    multipliers = chunk.qwen_multipliers
                    
                    if len(tokens) == 0:
                        tokens = [self.id_pad]
                        multipliers = [1.0]
                    
                    z: torch.Tensor = self.process_tokens([tokens], [multipliers])[0]

                cache[line] = (z, chunk)

            zs.append(z)
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

        z = {
            "qwen_cond": stack_with_padding(zs),
            "t5_ids": stack_with_padding(ti, pad_value=0),
            "t5_weights": stack_with_padding(tw, pad_value=1.0),
        }

        return z

    def anima_preprocess(self, cross_attn: torch.Tensor, t5xxl_ids: torch.Tensor, t5xxl_weights: torch.Tensor) -> torch.Tensor:
        device = memory_management.text_encoder_device()

        cross_attn = cross_attn.unsqueeze(0).to(device=device)
        t5xxl_ids = t5xxl_ids.unsqueeze(0).to(device=device)

        cross_attn = self.text_encoder.preprocess_text_embeds(cross_attn, t5xxl_ids)
        if t5xxl_weights is not None:
            cross_attn *= t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(cross_attn)

        if cross_attn.shape[1] < 512:
            cross_attn = torch.nn.functional.pad(cross_attn, (0, 0, 0, 512 - cross_attn.shape[1]))

        return cross_attn.squeeze(0)

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
        return z
