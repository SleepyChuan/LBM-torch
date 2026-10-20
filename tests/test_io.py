from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from lbm import create_solver
from lbm.io.npz import read_metadata, read_pdf, read_solid


def test_dump_setup_writes_metadata_and_solid(tmp_path, device):
    solid = torch.zeros((8, 10), dtype=torch.bool)
    solid[:, 0] = True
    s = create_solver("D2Q9", (8, 10), solid, nu=0.07, force=(1e-4, 0.0), device=device)
    s.initialize(rho=1.0, u=(0.01, 0.0))

    meta_path, solid_path = s.dump_setup(
        tmp_path / "run", total_steps=1000, save_interval=100
    )

    meta = read_metadata(meta_path)
    for key in (
        "model",
        "shape",
        "nu",
        "tau",
        "force",
        "device",
        "compute_dtype",
        "output_dtype",
        "total_steps",
        "save_interval",
        "created_at",
    ):
        assert key in meta
    assert meta["model"] == "D2Q9"
    assert meta["total_steps"] == 1000
    assert meta["save_interval"] == 100
    assert meta["tau"] == pytest.approx(3 * 0.07 + 0.5)

    arr = read_solid(solid_path)
    assert arr.dtype == np.bool_
    assert arr.shape == (8, 10)
    assert bool(arr[:, 0].all())


def test_save_npz_name_shape_and_dtype(tmp_path, device):
    s = create_solver(
        "D2Q9", (8, 10), nu=0.1, device=device, output_dtype=torch.float16
    )
    s.initialize()
    s.step(25)
    path = s.save_npz(tmp_path)

    assert path.name == "pdf_00000025.npz"
    arr = read_pdf(path)
    assert arr.shape == (9, 8, 10)
    assert arr.dtype == np.float16


def test_saved_file_tracks_current_buffer(tmp_path, device):
    s = create_solver(
        "D2Q9",
        (8, 10),
        nu=0.1,
        force=(1e-3, 0.0),
        device=device,
        output_dtype=torch.float32,
    )
    s.initialize()
    p0 = s.save_npz(tmp_path)
    s.step(50)
    p1 = s.save_npz(tmp_path)

    a0 = read_pdf(p0)
    a1 = read_pdf(p1)
    assert p0 != p1
    assert not np.allclose(a0, a1)
    # 存的是当前 buffer，不是另一个
    cur = s.pdf[s.cur_idx].detach().cpu().numpy()
    assert np.allclose(a1, cur)


def test_save_npz_creates_directory(tmp_path, device):
    s = create_solver("D2Q9", (4, 4), nu=0.1, device=device)
    s.initialize()
    nested = tmp_path / "a" / "b"
    p = s.save_npz(nested)
    assert p.exists()


def test_h5_and_vtk_are_stubs():
    from lbm.io import h5, vtk

    with pytest.raises(NotImplementedError):
        h5.write_pdf_h5("x.h5", np.zeros((9, 4, 4)))
    with pytest.raises(NotImplementedError):
        vtk.write_vtk("x.vtu", np.zeros((4, 4)), np.zeros((2, 4, 4)))
