from __future__ import annotations

import pytest

from lbm.stencils import D2Q9, D3Q19, D3Q27
from lbm.stencils.base import CS2


@pytest.mark.parametrize("spec", [D2Q9, D3Q19, D3Q27], ids=lambda s: s.name)
def test_spec_is_self_consistent(spec):
    """c/opp/w 的对合性、一阶矩、二阶矩。"""
    spec.check()  # 内部就是这些断言，不和规则会抛 ValueError


@pytest.mark.parametrize("spec", [D2Q9, D3Q19, D3Q27], ids=lambda s: s.name)
def test_opposite_maps_to_negated_vector(spec):
    for i, j in enumerate(spec.opp):
        assert tuple(-v for v in spec.c[i]) == spec.c[j]
        assert spec.opp[j] == i


def test_d2q9_direction_order_is_fixed():
    """方向编号是程序的约定，改动会让 ux/uy 互换。"""
    expected = [
        (0, 0),
        (1, 0),
        (0, -1),
        (-1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
        (-1, -1),
        (-1, 1),
    ]
    assert [tuple(c) for c in D2Q9.c] == expected


def test_d2q9_weights():
    assert D2Q9.w[0] == pytest.approx(4 / 9)
    assert all(w == pytest.approx(1 / 9) for w in D2Q9.w[1:5])
    assert all(w == pytest.approx(1 / 36) for w in D2Q9.w[5:])


def test_sound_speed_constant():
    assert CS2 == pytest.approx(1 / 3)
