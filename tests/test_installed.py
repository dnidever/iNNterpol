"""Run against the installed distribution, including its real checkpoint resources."""
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("grid,depth", [("atlas9", 71), ("marcs", 56), ("phoenix", 64)])
def test_interpolation_from_unrelated_directory(grid, depth, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # A conflicting model module and invalid checkpoint must not affect the package.
    Path("model.py").write_text("raise RuntimeError('wrong model')")
    Path("unrelated.pth").write_text("not a checkpoint")
    interpolate = importlib.import_module(f"innterpol.{grid}").innterpol
    result = interpolate([0, 0, 5500, 2.5] if grid == "phoenix" else [0, 0, 0, 5500, 2.5])
    if grid != "atlas9":
        result, tau = result
        assert tau.shape == (depth,)
    assert result.shape == (depth, 4)
    assert np.isfinite(np.asarray(result, dtype=float)).all()


def test_default_api():
    from innterpol import innterpol
    from innterpol.atlas9 import innterpol as atlas9
    assert innterpol is atlas9


@pytest.mark.parametrize("grid", ["atlas9", "marcs", "phoenix"])
def test_invalid_input(grid):
    interpolate = importlib.import_module(f"innterpol.{grid}").innterpol
    with pytest.raises(ValueError, match="4 params" if grid == "phoenix" else "5 params"):
        interpolate([0, 5500, 2.5])
