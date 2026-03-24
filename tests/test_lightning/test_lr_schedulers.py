"""Tests for custom LR schedulers."""

import math

import torch

from audioset_classification.lightning.lr_schedulers import HeadGroupCosineAnnealingLR


def test_head_group_cosine_annealing_single_param_group():
    """Head group lr follows cosine from base to eta_min over T_max steps."""
    m = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1.0)
    sched = HeadGroupCosineAnnealingLR(opt, T_max=4, eta_min=0.0)
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, rel_tol=1e-5)
    for _ in range(4):
        opt.step()
        sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.0, abs_tol=1e-6)


def test_head_group_cosine_second_group_passthrough():
    """Param group 1 lr is unchanged by the scheduler step."""
    m1 = torch.nn.Linear(2, 1)
    m2 = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(
        [
            {"params": m1.parameters(), "lr": 1.0},
            {"params": m2.parameters(), "lr": 0.05},
        ]
    )
    sched = HeadGroupCosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    before_g1 = opt.param_groups[1]["lr"]
    opt.step()
    sched.step()
    assert opt.param_groups[1]["lr"] == before_g1
