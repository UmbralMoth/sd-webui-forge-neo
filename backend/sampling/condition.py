import math

import torch


def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class Condition:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        if hasattr(self.cond, "shape") and hasattr(other.cond, "shape"):
            if self.cond.shape != other.cond.shape:
                return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
            
        def _cat(lst):
            if not lst: return None
            if isinstance(lst[0], dict):
                res = {}
                for k in lst[0].keys():
                    tensors = [c[k] for c in lst if isinstance(c, dict) and k in c]
                    if all(isinstance(t, torch.Tensor) for t in tensors):
                        res[k] = torch.cat(tensors)
                    else:
                        res[k] = tensors[0]
                return type(lst[0])(res, shape=getattr(lst[0], 'shape', None))
            elif isinstance(lst[0], torch.Tensor):
                return torch.cat(lst)
            return lst[0]

        is_layered = any(hasattr(c, "in_mid_cond") for c in conds)
        if is_layered:
            in_mids = [c.in_mid_cond if hasattr(c, "in_mid_cond") else c for c in conds]
            outs = [c.out_cond if hasattr(c, "out_cond") else c for c in conds]
            cls = next(type(c) for c in conds if hasattr(c, "in_mid_cond"))
            return cls(_cat(in_mids), _cat(outs))

        return _cat(conds)


class ConditionNoiseShape(Condition):
    def process_cond(self, batch_size, device, area, **kwargs):
        data = self.cond[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
        return self._copy_with(repeat_to_batch_size(data, batch_size).to(device))


class ConditionCrossAttn(Condition):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:
                return False

            mult_min = lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4:
                return False
        return True

    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1)
            out.append(c)
            
        def _cat(lst):
            if not lst: return None
            if isinstance(lst[0], dict):
                res = {}
                for k in lst[0].keys():
                    tensors = [c[k] for c in lst if isinstance(c, dict) and k in c]
                    if all(isinstance(t, torch.Tensor) for t in tensors):
                        res[k] = torch.cat(tensors)
                    else:
                        res[k] = tensors[0]
                return type(lst[0])(res, shape=getattr(lst[0], 'shape', None))
            elif isinstance(lst[0], torch.Tensor):
                return torch.cat(lst)
            return lst[0]
            
        is_layered = any(hasattr(c, "in_mid_cond") for c in out)
        if is_layered:
            in_mids = [c.in_mid_cond if hasattr(c, "in_mid_cond") else c for c in out]
            outs = [c.out_cond if hasattr(c, "out_cond") else c for c in out]
            cls = next(type(c) for c in out if hasattr(c, "in_mid_cond"))
            return cls(_cat(in_mids), _cat(outs))

        return _cat(out)


class ConditionConstant(Condition):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond





def compile_conditions(cond):
    if cond is None:
        return None

    if hasattr(cond, "in_mid_cond") and hasattr(cond, "out_cond"):
        # LayeredConditioning support
        
        if isinstance(cond.in_mid_cond, torch.Tensor):
            result = dict(
                cross_attn=cond,
                model_conds=dict(
                    c_crossattn=ConditionCrossAttn(cond),
                ),
            )
            return [result]

        cross_attn = cond["crossattn"]
        pooled_output = cond["vector"]

        model_conds = dict(c_crossattn=ConditionCrossAttn(cross_attn), y=Condition(pooled_output))

        # We need to iterate over available keys in the underlying dicts
        # Since LayeredConditioning wraps dicts
        keys = set()
        if isinstance(cond.in_mid_cond, dict): keys.update(cond.in_mid_cond.keys())
        if isinstance(cond.out_cond, dict): keys.update(cond.out_cond.keys())

        for k in keys:
            if k in {"crossattn", "vector", "guidance"}:
                continue
            v = cond[k]
            model_conds[k] = Condition(v)

        result = dict(cross_attn=cross_attn, pooled_output=pooled_output, model_conds=model_conds)

        guidance = cond["guidance"]
        if guidance is not None:
             result["model_conds"]["guidance"] = Condition(guidance)

        return [result]

    if isinstance(cond, torch.Tensor):

        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cond),
            ),
        )
        return [
            result,
        ]

    cross_attn = cond["crossattn"]
    pooled_output = cond["vector"]

    model_conds = dict(c_crossattn=ConditionCrossAttn(cross_attn), y=Condition(pooled_output))

    # Allow model-specific extra tensor conditions (e.g. token ids).
    for k, v in cond.items():
        if k in {"crossattn", "vector", "guidance"}:
            continue
        if isinstance(v, torch.Tensor):
            model_conds[k] = Condition(v)

    result = dict(cross_attn=cross_attn, pooled_output=pooled_output, model_conds=model_conds)

    if "guidance" in cond:
        result["model_conds"]["guidance"] = Condition(cond["guidance"])

    return [
        result,
    ]


def compile_weighted_conditions(cond, weights):
    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, "advanced_indexing"):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = cond[current_indices]

        h = compile_conditions(feed)
        h[0]["strength"] = current_weight
        results += h

    return results