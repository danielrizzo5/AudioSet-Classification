"""Learning rate schedules for Lightning training."""

import math

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class HeadGroupCosineAnnealingLR(LRScheduler):
    """Cosine anneal only param group 0; other groups keep their current ``lr``.

    ``BackboneFinetuning`` adds backbone weights as extra param groups and adjusts
    their learning rates each epoch. A plain ``CosineAnnealingLR`` would error once
    ``len(param_groups)`` grows, and would overwrite those backbone rates.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = max(int(T_max), 1)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        head_base = float(self.base_lrs[0])
        head_lr = (
            self.eta_min
            + (head_base - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
        )
        out: list[float | Tensor] = []
        for i, group in enumerate(self.optimizer.param_groups):
            if i == 0:
                out.append(head_lr)
            else:
                cur = group["lr"]
                out.append(
                    float(cur)
                    if not isinstance(cur, torch.Tensor)
                    else float(cur.item())
                )
        return out
