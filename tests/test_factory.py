from __future__ import annotations

import pytest
import torch

from lbm import create_solver


def test_unknown_model_raises_value_error(device):
    with pytest.raises(ValueError):
        create_solver("D2Q8", (8, 8), nu=0.1, device=device)


@pytest.mark.parametrize("model,shape", [("D3Q19", (4, 4, 4)), ("D3Q27", (4, 4, 4))])
def test_d3_stubs_construct_but_raise(model, shape, device):
    s = create_solver(model, shape, nu=0.1, device=device)
    assert s.model == model
    with pytest.raises(NotImplementedError):
        s.initialize()
    with pytest.raises(NotImplementedError):
        s.step(1)


def test_model_name_is_case_insensitive(device):
    s = create_solver("d2q9", (6, 6), nu=0.1, device=device)
    assert s.model == "D2Q9"


def test_shape_must_match_dimension(device):
    with pytest.raises(ValueError):
        create_solver("D2Q9", (4, 4, 4), nu=0.1, device=device)


def test_solid_shape_is_checked(device):
    with pytest.raises(ValueError):
        create_solver(
            "D2Q9", (8, 8), torch.zeros((4, 4), dtype=torch.bool), nu=0.1, device=device
        )
