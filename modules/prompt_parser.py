from __future__ import annotations

import re
import math
from collections import namedtuple

import lark
import torch

from modules import shared

from backend.logging import setup_logger
import logging
logger = logging.getLogger("prompt_parser")
setup_logger(logger)

logger.info("[Actias] Prompt Parser Extension Loaded (Anchored Blend + Token Alignment)")

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75] [in foreground::0.6] [:in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25,  'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50,  'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60,  'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75,  'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(
    r"""
!start: (prompt | /[][():|~]/+)*
?prompt: (scheduled | alternate | blended | emphasized | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate.2: "[" prompt_item ("|" [prompt_item])+ "]"
blended.2: "[" prompt_item ("~" [prompt_item])+ "]"
prompt_item: (scheduled | alternate | blended | emphasized | plain_item | WHITESPACE)*
WHITESPACE: /\s+/
plain: /([^\\\[\]():|~]|\\.)+/
plain_item: /([^\\\[\]()|~]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""
)


from backend.text_processing.blending import (
    n_way_blend, blend_layered_conds, pad_tensors, 
    DictWithShape, LayeredConditioning, surgical_delta_blend_conds
)

def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False):
    logger.info(f"Conditioning: Processing {len(prompts)} prompt schedules (Steps: {base_steps})")
    r"""
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    >>> g("[fe|]male")
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("[fe|||]male")
    [[1, 'female'], [2, 'male'], [3, 'male'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'male'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10)[0]
    >>> g("a [b:.5] c")
    [[10, 'a b c']]
    >>> g("a [b:1.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    """

    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps, tree):
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                if use_old_scheduling:
                    v = v * steps if v < 1 else v
                else:
                    if "." in s:
                        v = (v - flt_offset) * steps
                    else:
                        v = v - int_offset
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after

            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]

            def blended(self, args):
                args = ["" if not arg else arg for arg in args if arg is not None and (not isinstance(arg, lark.Token) or arg.type != 'NUMBER')]
                yield args[0] if args else ""

            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)

                return "".join(flatten(args))

            def plain(self, args):
                yield args[0].value
            
            def plain_item(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError as e:
            logger.info(f"[Actias] Parse failed for prompt: {prompt[:100]}... Error: {str(e)[:100]}")
            return [[steps, prompt]]

        use_legacy_alternation = getattr(shared.opts, "use_legacy_alternation_behavior", False)
        if use_legacy_alternation:
            return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

        def eval_topdown(node, step, state):
            if isinstance(node, lark.Token):
                return str(node.value)
                
            if not isinstance(node, lark.Tree):
                return str(node)
                
            if getattr(node, 'data', None) == 'alternate':
                node_id = id(node)
                if node_id not in state:
                    state[node_id] = 0
                else:
                    state[node_id] += 1
                
                idx = state[node_id] % len(node.children)
                child = node.children[idx]
                if child is None:
                    return ""
                res = eval_topdown(child, step, state)
                return res if res else ""
                
            elif getattr(node, 'data', None) == 'scheduled':
                before, after, _colon, when, _ws = node.children
                if step <= int(when):
                    return eval_topdown(before, step, state) if before else ""
                else:
                    return eval_topdown(after, step, state) if after else ""
                    
            elif getattr(node, 'data', None) == 'blended':
                prompts = [c for c in node.children if c is not None]
                if len(prompts) < 2:
                    return eval_topdown(prompts[0], step, state) if prompts else ""
                    
                evaluated_prompts = [eval_topdown(p, step, state) if p else "" for p in prompts]
                
                parsed_items = []
                for p in evaluated_prompts:
                    p_str = str(p).strip()
                    # Match weight suffix like ":1.2"
                    match = re.search(r"^\s*(.*?)\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+))\s*$", p_str)
                    
                    if match:
                        text, weight_str = match.groups()
                        weight = float(weight_str)
                        if isinstance(p, LatentBlendNode):
                            # It's a nested blend with a weight suffix
                            suffix = p_str[len(text):].strip()
                            
                            def _strip(n):
                                if isinstance(n, LatentBlendNode):
                                    return LatentBlendNode([[_strip(c), w] for c, w in n.items])
                                s = str(n)
                                if s.endswith(suffix):
                                    return s[:-len(suffix)]
                                return s
                            
                            p = _strip(p)
                            parsed_items.append([p, weight])
                        else:
                            parsed_items.append([text.strip(), weight])
                    else:
                        parsed_items.append([p, None])
                        
                explicit_sum = sum(w for _, w in parsed_items if w is not None)
                unweighted = [item for item in parsed_items if item[1] is None]
                
                if len(unweighted) > 0:
                    remainder = max(0.0, 1.0 - explicit_sum)
                    fill_weight = remainder / len(unweighted)
                else:
                    fill_weight = 1.0
                    
                items = []
                for text, w in parsed_items:
                    final_w = w if w is not None else fill_weight
                    items.append([text, final_w])
                
                # Clean Logging: show the user the resolved weights
                weight_report = ", ".join([f"{str(t)[:50]}: {w:.2f}" for t, w in items])
                logger.info(f"Prompt Blend: Resolved [{weight_report}]")
                    
                return LatentBlendNode(items)
                
            elif getattr(node, 'data', None) == 'plain':
                return str(node.children[0].value)
                
            else:
                def combine(items):
                    items = [x for x in items if x != () and x is not None]
                    if len(items) == 0:
                        return ""
                    if len(items) == 1:
                        return items[0]
                        
                    first = items[0]
                    rest = combine(items[1:])
                    
                    if isinstance(first, LatentBlendNode):
                        new_items = []
                        for child, weight in first.items:
                            new_items.append((combine([child, rest]), weight))
                        return LatentBlendNode(new_items)
                    elif isinstance(rest, LatentBlendNode):
                        new_items = []
                        for child, weight in rest.items:
                            new_items.append((combine([first, child]), weight))
                        return LatentBlendNode(new_items)
                    else:
                        return str(first) + str(rest)
                        
                res = [eval_topdown(c, step, state) for c in node.children if c is not None]
                return combine(res)

        ts = collect_steps(steps, tree)
        state_dict = {}
        schedule = []
        for t in ts:
            schedule.append([t, eval_topdown(tree, t, state_dict)])
        return schedule

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]





def align_token_ids(token_ids_list, pad_id):
    if not token_ids_list:
        return token_ids_list
    
    # 1. Find common prefix
    prefix_len = 0
    min_len = min(len(ids) for ids in token_ids_list)
    for i in range(min_len):
        if all(ids[i] == token_ids_list[0][i] for ids in token_ids_list):
            prefix_len += 1
        else:
            break
            
    # 2. Find common suffix
    suffix_len = 0
    for i in range(1, min_len - prefix_len + 1):
        if all(ids[-i] == token_ids_list[0][-i] for ids in token_ids_list):
            suffix_len += 1
        else:
            break
            
    # 3. Differing middle parts
    max_mid_len = max(len(ids) - prefix_len - suffix_len for ids in token_ids_list)
    
    aligned = []
    for ids in token_ids_list:
        prefix = ids[:prefix_len]
        suffix = ids[len(ids)-suffix_len:] if suffix_len > 0 else []
        mid = ids[prefix_len : len(ids)-suffix_len] if suffix_len > 0 else ids[prefix_len:]
        
        # Pad mid part at the end
        padded_mid = list(mid) + [pad_id] * (max_mid_len - len(mid))
        aligned.append(prefix + padded_mid + suffix)
        
    return aligned


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """

    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None, distilled_cfg_scale=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, "is_negative_prompt", False)
        self.width = width or getattr(copy_from, "width", None)
        self.height = height or getattr(copy_from, "height", None)
        self.distilled_cfg_scale = distilled_cfg_scale or getattr(copy_from, "distilled_cfg_scale", None)


class PrealignedString(str):
    def __new__(cls, text, aligned_tokens_dict):
        obj = str.__new__(cls, text)
        obj.aligned_tokens_dict = aligned_tokens_dict
        return obj


def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        # 1. Expand the tree to find every unique prompt permutation string
        all_unique_texts = set()
        
        def find_all_texts(n):
            if isinstance(n, LatentBlendNode):
                for child, _ in n.items:
                    find_all_texts(child)
            else:
                all_unique_texts.add(str(n))
        
        for _, text_or_blend in prompt_schedule:
            find_all_texts(text_or_blend)
            
        unique_texts_list = list(all_unique_texts)

        # 2. Global Token Alignment pass
        tokenizers = {}
        if hasattr(model, "forge_objects"):
            tokenizers = getattr(model.forge_objects.clip.tokenizer, "__dict__", {})
            
        aligned_token_ids_map = {} # (text, tokenizer_key) -> ids

        for k, tokenizer in tokenizers.items():
            if tokenizer is None or k.startswith("__"): continue
            try:
                token_ids_list = [tokenizer([t], add_special_tokens=False)["input_ids"][0] for t in unique_texts_list]
                pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
                aligned_ids = align_token_ids(token_ids_list, pad_id)
                
                for text, ids in zip(unique_texts_list, aligned_ids):
                    aligned_token_ids_map[(text, k)] = ids
            except Exception as e:
                logger.error(f"[Actias] Global Alignment failed for {k}: {e}")

        # 3. Create PrealignedString objects for encoding
        prealigned_objects = {}
        texts_to_encode = []
        for text in unique_texts_list:
            token_dict = {tk: aligned_token_ids_map[(text, tk)] for tk in tokenizers.keys() if (text, tk) in aligned_token_ids_map}
            obj = PrealignedString(text, token_dict)
            prealigned_objects[text] = obj
            texts_to_encode.append(obj)

        if not texts_to_encode:
            texts_to_encode = [prompt]
            
        texts_obj = SdConditioning(texts_to_encode, copy_from=prompts)
        encoded_conds = model.get_learned_conditioning(texts_obj)
        
        encoded_list = []
        for i in range(len(texts_to_encode)):
            if isinstance(encoded_conds, dict):
                encoded_list.append({k: v[i] if isinstance(v, torch.Tensor) else v[i] for k, v in encoded_conds.items()})
            else:
                if isinstance(encoded_conds, LayeredConditioning) and isinstance(encoded_conds.in_mid_cond, torch.Tensor):
                    encoded_list.append(encoded_conds[i])
                else:
                    encoded_list.append(encoded_conds[i] if isinstance(encoded_conds, torch.Tensor) else encoded_conds[i])
                
        cond_dict = dict(zip(texts_to_encode, encoded_list))
        
        # 4. Final Recursive Resolution (Surgical Delta Grafting)
        def build_cond(node):
            if isinstance(node, LatentBlendNode):
                # 4a. Identify the "Global Baseline" string and generate a Slot Mask
                # The mask tracks which token indices belong to a blend slot (True) vs static (False)
                tokenizers = {}
                if hasattr(model, "forge_objects"):
                    tokenizers = getattr(model.forge_objects.clip.tokenizer, "__dict__", {})
                
                # We'll use the tokenizer corresponding to the sequence-based crossattn tensor
                # For Qwen 3.5 4B (which bypasses T5/LLM adapter), we must use the qwen3_5_4b tokenizer.
                if hasattr(model, "forge_objects") and model.forge_objects is not None and hasattr(model.forge_objects, "clip") and model.forge_objects.clip is not None:
                    cond_stage = model.forge_objects.clip.cond_stage_model
                    if "qwen3_5_4b" in cond_stage and "qwen3_5_4b" in tokenizers:
                        tokenizer_key = "qwen3_5_4b"
                    elif "t5xxl" in tokenizers:
                        tokenizer_key = "t5xxl"
                    elif "umt5xxl" in tokenizers:
                        tokenizer_key = "umt5xxl"
                    else:
                        tokenizer_key = next((k for k in tokenizers.keys() if not k.startswith("__")), "default")
                else:
                    if "t5xxl" in tokenizers:
                        tokenizer_key = "t5xxl"
                    elif "umt5xxl" in tokenizers:
                        tokenizer_key = "umt5xxl"
                    else:
                        tokenizer_key = next((k for k in tokenizers.keys() if not k.startswith("__")), "default")
                
                def get_metadata(n):
                    if isinstance(n, LatentBlendNode):
                        # Find the baseline child marker
                        base_child = next((c for c, w in n.items if w < 0.0), None)
                        if base_child is None:
                            base_child = n.items[0][0]
                            
                        # Recursively find the baseline text and sub-masks
                        base_text, sub_mask = get_metadata(base_child)
                        
                        ids_base = aligned_token_ids_map.get((base_text, tokenizer_key), [])
                        
                        # A slot index is any index where any variant's token IDs differ from the baseline's token IDs
                        node_mask = []
                        for child, _ in n.items:
                            child_text, _ = get_metadata(child)
                            if child_text == base_text:
                                continue
                            ids_child = aligned_token_ids_map.get((child_text, tokenizer_key), [])
                            temp_mask = [i != j for i, j in zip(ids_child, ids_base)]
                            if not node_mask:
                                node_mask = temp_mask
                            else:
                                mlen = max(len(node_mask), len(temp_mask))
                                merged = []
                                for idx in range(mlen):
                                    v1 = node_mask[idx] if idx < len(node_mask) else False
                                    v2 = temp_mask[idx] if idx < len(temp_mask) else False
                                    merged.append(v1 or v2)
                                node_mask = merged
                                
                        if not node_mask:
                            node_mask = [False] * len(ids_base)
                        
                        # Merge sub_mask into node_mask (OR logic)
                        if not sub_mask:
                            return base_text, node_mask
                            
                        mlen = max(len(node_mask), len(sub_mask))
                        merged = []
                        for i in range(mlen):
                            v1 = node_mask[i] if i < len(node_mask) else False
                            v2 = sub_mask[i] if i < len(sub_mask) else False
                            merged.append(v1 or v2)
                        return base_text, merged
                    return str(n), []

                baseline_text, raw_mask = get_metadata(node)
                baseline_obj = cond_dict.get(prealigned_objects.get(baseline_text), None)
                
                # Robustly find a tensor to get sequence length from
                tensor_for_shape = baseline_obj
                while not isinstance(tensor_for_shape, torch.Tensor):
                    if isinstance(tensor_for_shape, dict):
                        tensor_for_shape = tensor_for_shape.get("crossattn", next(iter(tensor_for_shape.values())))
                    elif hasattr(tensor_for_shape, "in_mid_cond"):
                        tensor_for_shape = tensor_for_shape.in_mid_cond
                    else:
                        break
                        
                seq_len = tensor_for_shape.shape[1] if tensor_for_shape.ndim >= 3 else tensor_for_shape.shape[0]
                full_mask = torch.zeros((1, seq_len, 1), dtype=torch.bool)
                
                # If it's a legacy model (SD1.5, SDXL), apply index mapping accounting for BOS/EOS tokens at every 77th position.
                # Otherwise (Flux, Anima, Wan), map tokens 1-to-1.
                is_legacy = getattr(model, "is_webui_legacy_model", lambda: True)()
                for i, is_unique in enumerate(raw_mask):
                    if is_legacy:
                        idx = 1 + i + 2 * (i // 75)
                    else:
                        idx = i
                    if idx < seq_len:
                        full_mask[0, idx, 0] = is_unique

                # 4b. Flatten and blend aligned permutations
                flat_perms = []
                def _expand(n, w_acc=1.0, path=""):
                    if isinstance(n, LatentBlendNode):
                        for child, w in n.items:
                            if w < 0.0: continue # Skip baseline marker
                            child_text = str(child)
                            new_path = f"{path} + {child_text[:15]}" if path else child_text[:15]
                            _expand(child, w_acc * w, new_path)
                    else:
                        text = str(n)
                        obj = prealigned_objects.get(text, n)
                        flat_perms.append((obj, w_acc, path))
                
                _expand(node)
                
                # Consolidate weights
                consolidated = {}
                for obj, weight, path in flat_perms:
                    curr_w, curr_p = consolidated.get(obj, (0.0, path))
                    consolidated[obj] = (curr_w + weight, curr_p)
                
                final_items = []
                for obj, (w, p) in consolidated.items():
                    final_items.append((obj, w, p))
                final_items.sort(key=lambda x: str(x[0]))
                
                variants = [cond_dict[item[0]] for item in final_items]
                weights = [item[1] for item in final_items]
                
                return surgical_delta_blend_conds(baseline_obj, variants, weights, full_mask, alpha=0.0)
            else:
                # Leaf node
                text = str(node)
                obj = prealigned_objects.get(text, node)
                return cond_dict.get(obj, None)
                
        cond_schedule = []
        for end_at_step, text_or_blend in prompt_schedule:
            cond = build_cond(text_or_blend)
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res


re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")


def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    res_indexes = []

    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        subprompts = re_AND.split(prompt)

        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)

            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight


class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch


def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False) -> MulticondLearnedConditioning:
    """
    same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.
    """

    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, hires_steps, use_old_scheduling)

    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)





class LatentBlendNode(str):
    def __new__(cls, items):
        # We still use the longest string for compatibility with length-checks in SD
        longest = str(max((item[0] for item in items), key=lambda x: len(str(x))))
        obj = str.__new__(cls, longest)
        obj.items = items
        return obj

    def __repr__(self):
        return f"[LatentBlendNode: {str(self)}]"


def _pad_seq(t1, t2):
    res = pad_tensors([t1, t2])
    return res[0], res[1]

def slerp_tensor(val: float, low: torch.Tensor, high: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    low_n = torch.linalg.vector_norm(low, dim=dim, keepdim=True)
    high_n = torch.linalg.vector_norm(high, dim=dim, keepdim=True)
    
    low_norm = low / low_n.clamp(min=eps)
    high_norm = high / high_n.clamp(min=eps)
    
    dot = (low_norm * high_norm).sum(dim=dim, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    mask = so.abs() < eps
    so = torch.where(mask, torch.ones_like(so), so)
    
    res_dir = (torch.sin((1.0 - val) * omega) / so) * low_norm + (torch.sin(val * omega) / so) * high_norm
    res_dir = torch.where(mask, (1.0 - val) * low_norm + val * high_norm, res_dir)
    
    target_n = low_n * (1.0 - val) + high_n * val
    return res_dir * target_n

def n_way_blend_conds(conds, weights, alpha=1.0):
    total_weight = sum(weights)
    if total_weight <= 0.001:
        return conds[0]
        
    any_layered = any(isinstance(c, LayeredConditioning) for c in conds)
    if not any_layered:
        # Standard model or recursive layer pass - apply provided alpha policy
        return blend_layered_conds(conds, weights, alpha)

    # Top-level Layered Blending: Dual-Alpha Routing
    in_mid_inputs = [c.in_mid_cond if isinstance(c, LayeredConditioning) else c for c in conds]
    out_inputs = [c.out_cond if isinstance(c, LayeredConditioning) else c for c in conds]

    comp_alpha = alpha if alpha < 0.5 else 0.8
    style_alpha = alpha if alpha < 0.5 else 0.0
    
    return LayeredConditioning(
        blend_layered_conds(in_mid_inputs, weights, comp_alpha),
        blend_layered_conds(out_inputs, weights, style_alpha)
    )


def blend_conds(cond1, cond2, weight, alpha=1.0):
    return n_way_blend_conds([cond1, cond2], [1.0 - weight, weight], alpha=alpha)



def get_continuous_cond(schedules, step_float, alpha=1.0):
    if len(schedules) == 1:
        return schedules[0].cond
    
    # Are we in an alternating schedule? (all step diffs == 1)
    is_alternating = True
    for i in range(1, len(schedules)):
        if schedules[i].end_at_step - schedules[i-1].end_at_step != 1:
            is_alternating = False
            break
            
    if is_alternating:
        # wave function indexing
        idx = int(step_float)
        idx = min(idx, len(schedules) - 1)
        next_idx = min(idx + 1, len(schedules) - 1)
        
        cond1 = schedules[idx].cond
        cond2 = schedules[next_idx].cond
        
        w = math.sin(math.pi / 2.0 * (step_float - idx)) ** 2
        return blend_conds(cond1, cond2, w, alpha=alpha)
    else:
        # Sigmoid Hand-off Edit sequence
        current_cond = schedules[0].cond
        for i in range(1, len(schedules)):
            cond_next = schedules[i].cond
            swap_step = schedules[i-1].end_at_step
            # c = 2.0 creates a ~4 step blend window
            w = torch.sigmoid(torch.tensor(2.0 * (step_float - swap_step))).item()
            current_cond = blend_conds(current_cond, cond_next, w, alpha=alpha)
        return current_cond


def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step, step_float=None, alpha=1.0):
    if step_float is None:
        step_float = float(current_step)

    param = c[0][0].cond
    if isinstance(param, LayeredConditioning):
        comp_alpha = alpha if alpha < 0.5 else 0.8
        style_alpha = alpha if alpha < 0.5 else 0.0
        in_mid_batch = reconstruct_cond_batch([[ScheduledPromptConditioning(s.end_at_step, s.cond.in_mid_cond) for s in sched] for sched in c], current_step, step_float, alpha=comp_alpha)
        out_batch = reconstruct_cond_batch([[ScheduledPromptConditioning(s.end_at_step, s.cond.out_cond) for s in sched] for sched in c], current_step, step_float, alpha=style_alpha)
        return LayeredConditioning(in_mid_batch, out_batch)

    is_dict = isinstance(param, dict)

    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype) for k, param in dict_cond.items()}
        res = DictWithShape(res, shape=getattr(dict_cond, 'shape', None))
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)

    for i, cond_schedule in enumerate(c):
        cond_val = get_continuous_cond(cond_schedule, step_float, alpha=alpha)

        if is_dict:
            for k, param_val in cond_val.items():
                res[k][i] = param_val
        else:
            res[i] = cond_val

    return res


def stack_conds(tensors):
    try:
        result = torch.stack(tensors)
    except:
        # if prompts have wildly different lengths above the limit we'll get tensors of different shapes
        # and won't be able to torch.stack them. So this fixes that.
        token_count = max([x.shape[0] for x in tensors])
        for i in range(len(tensors)):
            if tensors[i].shape[0] != token_count:
                last_vector = tensors[i][-1:]
                last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
                tensors[i] = torch.vstack([tensors[i], last_vector_repeated])
        result = torch.stack(tensors)
    return result


def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step, step_float=None, alpha=1.0):
    if step_float is None:
        step_float = float(current_step)

    param = c.batch[0][0].schedules[0].cond
    if isinstance(param, LayeredConditioning):
        # We need to manually reconstruct this because MulticondLearnedConditioning is complex
        # But we can just create two MulticondLearnedConditioning objects!
        
        batch_in_mid = []
        batch_out = []
        for composable_prompts in c.batch:
            scheds_in_mid = []
            scheds_out = []
            for cp in composable_prompts:
                s_in_mid = [ScheduledPromptConditioning(s.end_at_step, s.cond.in_mid_cond) for s in cp.schedules]
                s_out = [ScheduledPromptConditioning(s.end_at_step, s.cond.out_cond) for s in cp.schedules]
                scheds_in_mid.append(ComposableScheduledPromptConditioning(s_in_mid, cp.weight))
                scheds_out.append(ComposableScheduledPromptConditioning(s_out, cp.weight))
            batch_in_mid.append(scheds_in_mid)
            batch_out.append(scheds_out)
            
        m_in_mid = MulticondLearnedConditioning(c.shape, batch_in_mid)
        m_out = MulticondLearnedConditioning(c.shape, batch_out)
        
        comp_alpha = alpha if alpha < 0.5 else 0.8
        style_alpha = alpha if alpha < 0.5 else 0.0
        
        conds_list, stacked_in_mid = reconstruct_multicond_batch(m_in_mid, current_step, step_float, alpha=comp_alpha)
        _, stacked_out = reconstruct_multicond_batch(m_out, current_step, step_float, alpha=style_alpha)
        
        return conds_list, LayeredConditioning(stacked_in_mid, stacked_out)

    tensors = []
    conds_list = []

    for composable_prompts in c.batch:
        conds_for_batch = []

        for composable_prompt in composable_prompts:
            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(get_continuous_cond(composable_prompt.schedules, step_float, alpha=alpha))

        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked, shape=getattr(tensors[0], 'shape', None))
    else:
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)

    return conds_list, stacked


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res
