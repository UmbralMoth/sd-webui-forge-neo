from __future__ import annotations

import re
import math
from collections import namedtuple

import lark
import torch

from modules import shared

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75] [in foreground::0.6] [:in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25,  'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50,  'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60,  'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75,  'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(
    r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | blended | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
blended: "[" prompt ("~" [prompt])+ [":" NUMBER] "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|~]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""
)


def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False):
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

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            if 0:
                import traceback

                traceback.print_exc()
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
                weight = 0.5
                prompts = []
                for c in node.children:
                    if isinstance(c, lark.Token) and c.type == 'NUMBER':
                        weight = float(c)
                    else:
                        prompts.append(c)
                
                if len(prompts) < 2:
                    return eval_topdown(prompts[0], step, state) if prompts else ""
                    
                evaluated_prompts = [eval_topdown(p, step, state) if p else "" for p in prompts]
                
                result = evaluated_prompts[-1]
                for p in reversed(evaluated_prompts[:-1]):
                    result = LatentBlendNode(p, result, weight)
                return result
                
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
                        return LatentBlendNode(
                            combine([first.left, rest]),
                            combine([first.right, rest]),
                            first.weight
                        )
                    elif isinstance(rest, LatentBlendNode):
                        return LatentBlendNode(
                            combine([first, rest.left]),
                            combine([first, rest.right]),
                            rest.weight
                        )
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


class LatentBlendNode(str):
    def __new__(cls, left, right, weight):
        longest = str(left) if len(str(left)) > len(str(right)) else str(right)
        obj = str.__new__(cls, longest)
        obj.left = left
        obj.right = right
        obj.weight = float(weight)
        return obj


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


def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    r"""
    converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning,
    specifying the condition (cond), and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts_to_encode = []
        def extract_texts(node):
            if isinstance(node, LatentBlendNode):
                extract_texts(node.left)
                extract_texts(node.right)
            else:
                if node not in texts_to_encode:
                    texts_to_encode.append(node)
                    
        for _, text_or_blend in prompt_schedule:
            extract_texts(text_or_blend)
            
        texts_obj = SdConditioning(texts_to_encode, copy_from=prompts)
        encoded_conds = model.get_learned_conditioning(texts_obj)
        
        encoded_list = []
        for i in range(len(texts_to_encode)):
            if isinstance(encoded_conds, dict):
                encoded_list.append({k: v[i] for k, v in encoded_conds.items()})
            else:
                encoded_list.append(encoded_conds[i])
                
        cond_dict = dict(zip(texts_to_encode, encoded_list))
        
        def build_cond(node):
            if isinstance(node, LatentBlendNode):
                cond_left = build_cond(node.left)
                cond_right = build_cond(node.right)
                return equalized_blend_conds(cond_left, cond_right, node.weight, alpha=1.0)
            else:
                return cond_dict[node]
                
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


def _pad_seq(t1, t2):
    if getattr(t1, 'ndim', 0) < 2 or getattr(t2, 'ndim', 0) < 2: return t1, t2
    seq_dim = 1 if getattr(t1, 'ndim', 0) == 3 else 0
    if t1.shape[seq_dim] != t2.shape[seq_dim]:
        max_len = max(t1.shape[seq_dim], t2.shape[seq_dim])
        
        pad_shape1 = list(t1.shape)
        pad_shape1[seq_dim] = max_len - t1.shape[seq_dim]
        t1 = torch.cat([t1, torch.zeros(pad_shape1, dtype=t1.dtype, device=t1.device)], dim=seq_dim)
        
        pad_shape2 = list(t2.shape)
        pad_shape2[seq_dim] = max_len - t2.shape[seq_dim]
        t2 = torch.cat([t2, torch.zeros(pad_shape2, dtype=t2.dtype, device=t2.device)], dim=seq_dim)
    return t1, t2

def equalized_blend_conds(cond1, cond2, weight, alpha=1.0):
    if weight <= 0.001: return cond1
    if weight >= 0.999: return cond2
        
    if isinstance(cond1, dict):
        res = {}
        for k in cond1.keys():
            if isinstance(cond1[k], torch.Tensor) and isinstance(cond2[k], torch.Tensor):
                t1, t2 = _pad_seq(cond1[k], cond2[k])
                n1 = torch.linalg.vector_norm(t1, dim=-1, keepdim=True)
                n2 = torch.linalg.vector_norm(t2, dim=-1, keepdim=True)
                gm = torch.sqrt(n1 * n2).clamp(min=1e-12)
                
                target_n1 = torch.lerp(n1, gm, alpha)
                target_n2 = torch.lerp(n2, gm, alpha)
                
                t1s = t1 * (target_n1 / n1.clamp(min=1e-12))
                t2s = t2 * (target_n2 / n2.clamp(min=1e-12))
                
                blend = t1s * (1.0 - weight) + t2s * weight
                blend_norm = torch.linalg.vector_norm(blend, dim=-1, keepdim=True)
                target_norm = target_n1 * (1.0 - weight) + target_n2 * weight
                
                res[k] = blend * (target_norm / blend_norm.clamp(min=1e-12))
            else:
                res[k] = cond1[k]
        if hasattr(cond1, "shape"):
            return type(cond1)(res, shape=getattr(cond1, 'shape', None))
        return type(cond1)(res)
    elif isinstance(cond1, torch.Tensor) and isinstance(cond2, torch.Tensor):
        t1, t2 = _pad_seq(cond1, cond2)
        n1 = torch.linalg.vector_norm(t1, dim=-1, keepdim=True)
        n2 = torch.linalg.vector_norm(t2, dim=-1, keepdim=True)
        gm = torch.sqrt(n1 * n2).clamp(min=1e-12)
        
        target_n1 = torch.lerp(n1, gm, alpha)
        target_n2 = torch.lerp(n2, gm, alpha)
        
        t1s = t1 * (target_n1 / n1.clamp(min=1e-12))
        t2s = t2 * (target_n2 / n2.clamp(min=1e-12))
        
        blend = t1s * (1.0 - weight) + t2s * weight
        blend_norm = torch.linalg.vector_norm(blend, dim=-1, keepdim=True)
        target_norm = target_n1 * (1.0 - weight) + target_n2 * weight
        
        return blend * (target_norm / blend_norm.clamp(min=1e-12))
    return cond1


def blend_conds(cond1, cond2, weight):
    if weight <= 0.001:
        return cond1
    if weight >= 0.999:
        return cond2
    if isinstance(cond1, dict):
        res = {}
        for k in cond1.keys():
            if isinstance(cond1[k], torch.Tensor) and isinstance(cond2[k], torch.Tensor):
                t1, t2 = _pad_seq(cond1[k], cond2[k])
                blend = t1 * (1.0 - weight) + t2 * weight
                n1 = torch.linalg.vector_norm(t1, dim=-1, keepdim=True)
                n2 = torch.linalg.vector_norm(t2, dim=-1, keepdim=True)
                blend_norm = torch.linalg.vector_norm(blend, dim=-1, keepdim=True)
                target_norm = n1 * (1.0 - weight) + n2 * weight
                res[k] = blend * (target_norm / blend_norm.clamp(min=1e-12))
            else:
                res[k] = cond1[k]
        if hasattr(cond1, "shape"):
            return type(cond1)(res, shape=getattr(cond1, 'shape', None))
        return type(cond1)(res)
    else:
        t1, t2 = _pad_seq(cond1, cond2)
        blend = t1 * (1.0 - weight) + t2 * weight
        n1 = torch.linalg.vector_norm(t1, dim=-1, keepdim=True)
        n2 = torch.linalg.vector_norm(t2, dim=-1, keepdim=True)
        blend_norm = torch.linalg.vector_norm(blend, dim=-1, keepdim=True)
        target_norm = n1 * (1.0 - weight) + n2 * weight
        return blend * (target_norm / blend_norm.clamp(min=1e-12))


def get_continuous_cond(schedules, step_float):
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
        return blend_conds(cond1, cond2, w)
    else:
        # Sigmoid Hand-off Edit sequence
        current_cond = schedules[0].cond
        for i in range(1, len(schedules)):
            cond_next = schedules[i].cond
            swap_step = schedules[i-1].end_at_step
            # c = 2.0 creates a ~4 step blend window
            w = torch.sigmoid(torch.tensor(2.0 * (step_float - swap_step))).item()
            current_cond = blend_conds(current_cond, cond_next, w)
        return current_cond


def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step, step_float=None):
    if step_float is None:
        step_float = float(current_step)

    param = c[0][0].cond
    is_dict = isinstance(param, dict)

    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype) for k, param in dict_cond.items()}
        res = DictWithShape(res)
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)

    for i, cond_schedule in enumerate(c):
        cond_val = get_continuous_cond(cond_schedule, step_float)

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


def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step, step_float=None):
    if step_float is None:
        step_float = float(current_step)

    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for composable_prompts in c.batch:
        conds_for_batch = []

        for composable_prompt in composable_prompts:
            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(get_continuous_cond(composable_prompt.schedules, step_float))

        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked)
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
