import copy

import torch


def clone_state_dict(state_dict: dict[str, torch.Tensor], to_cpu: bool = False) -> dict[str, torch.Tensor]:
    cloned = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if to_cpu:
                tensor = tensor.cpu()
            cloned[key] = tensor
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state dicts were provided for averaging.")

    if len(state_dicts) == 1:
        return clone_state_dict(state_dicts[0], to_cpu=True)

    averaged: dict[str, torch.Tensor] = {}
    reference = state_dicts[0]

    for key, value in reference.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                accumulator = value.detach().to(torch.float32).cpu().clone()
                for state_dict in state_dicts[1:]:
                    accumulator.add_(state_dict[key].detach().to(torch.float32).cpu())
                accumulator.div_(float(len(state_dicts)))
                averaged[key] = accumulator.to(dtype=value.dtype)
            else:
                averaged[key] = value.detach().cpu().clone()
        else:
            averaged[key] = copy.deepcopy(value)

    return averaged


class ExponentialMovingAverage:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = clone_state_dict(model.state_dict(), to_cpu=False)

    def update(self, model):
        current_state = model.state_dict()

        for key, value in current_state.items():
            shadow_value = self.shadow[key]

            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    shadow_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    shadow_value.copy_(value.detach())
            else:
                self.shadow[key] = copy.deepcopy(value)

    def state_dict(self, to_cpu: bool = False) -> dict[str, torch.Tensor]:
        return clone_state_dict(self.shadow, to_cpu=to_cpu)