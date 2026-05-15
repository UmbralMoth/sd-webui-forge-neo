import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class T5TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, min_length: int = 256, min_padding: int = -1):
        super().__init__()

        self.text_encoder = text_encoder.transformer
        self.tokenizer = tokenizer

        self.min_length = min_length
        self.min_padding = min_padding
        self.id_end = 1
        self.id_pad = 0
        self.break_tokens = self.tokenizer(["BREAK"], add_special_tokens=False)["input_ids"][0]

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def encode_with_transformers(self, tokens):
        device = memory_management.text_encoder_device()
        tokens = tokens.to(device)
        return self.text_encoder(input_ids=tokens)

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            chunk.tokens = chunk.tokens + [self.id_end]
            chunk.multipliers = chunk.multipliers + [1.0]

            if self.min_padding > 0:
                chunk.tokens += [self.id_pad] * self.min_padding
                chunk.multipliers += [1.0] * self.min_padding

            current_chunk_length = len(chunk.tokens)

            token_count += current_chunk_length
            remaining_count = self.min_length - current_chunk_length

            if self.min_length > 0 and remaining_count > 0:
                chunk.tokens += [self.id_pad] * remaining_count
                chunk.multipliers += [1.0] * remaining_count

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count

    def __call__(self, texts):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                tokens = None
                if hasattr(line, "aligned_tokens_dict") and line.aligned_tokens_dict is not None:
                    tokens = line.aligned_tokens_dict.get("t5xxl", None)
                elif hasattr(line, "aligned_tokens") and line.aligned_tokens is not None:
                    tokens = line.aligned_tokens
                
                if tokens is not None:
                    chunks = []
                    current_chunk = PromptChunk()
                    
                    def pad_chunk(c):
                        c.tokens = c.tokens + [self.id_end]
                        c.multipliers = c.multipliers + [1.0]

                        if self.min_padding > 0:
                            c.tokens += [self.id_pad] * self.min_padding
                            c.multipliers += [1.0] * self.min_padding

                        current_chunk_length = len(c.tokens)
                        remaining_count = self.min_length - current_chunk_length

                        if self.min_length > 0 and remaining_count > 0:
                            c.tokens += [self.id_pad] * remaining_count
                            c.multipliers += [1.0] * remaining_count
                        return c

                    pos = 0
                    while pos < len(tokens):
                        if self.break_tokens and tokens[pos : pos + len(self.break_tokens)] == self.break_tokens:
                            chunks.append(pad_chunk(current_chunk))
                            current_chunk = PromptChunk()
                            pos += len(self.break_tokens)
                            continue
                        current_chunk.tokens.append(tokens[pos])
                        current_chunk.multipliers.append(1.0)
                        pos += 1
                    chunks.append(pad_chunk(current_chunk))
                    token_count = sum(len(c.tokens) for c in chunks)
                else:
                    chunks, token_count = self.tokenize_line(line)
                
                line_z_values = []

                # pad all chunks to length of longest chunk
                max_tokens = 0
                for chunk in chunks:
                    max_tokens = max(len(chunk.tokens), max_tokens)

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    remaining_count = max_tokens - len(tokens)
                    if remaining_count > 0:
                        tokens += [self.id_pad] * remaining_count
                        multipliers += [1.0] * remaining_count

                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return torch.stack(zs)

    def process_tokens(self, batch_tokens, batch_multipliers):
        tokens = torch.asarray(batch_tokens)

        z = self.encode_with_transformers(tokens)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
