"""The fast-kernel switch falls back to a fresh stock model on any error or mismatch (no GPU needed)."""
import numpy as np

from dittodub_longform import accel


def _patch(monkeypatch, before, after, fail=False):
    calls = iter([before, after])
    monkeypatch.setattr(accel, "_probe", lambda model: next(calls))

    def install(model):
        if fail:
            raise RuntimeError("kernel failed")
    monkeypatch.setattr(accel, "install", install)


def test_identical_results_keep_the_kernels(monkeypatch):
    x = [np.ones((2, 3), np.float32)]
    _patch(monkeypatch, x, x)
    assert accel.enable("model", lambda: "stock") == ("model", True)


def test_mismatch_or_error_falls_back(monkeypatch):
    _patch(monkeypatch, [np.ones(3, np.float32)], [np.zeros(3, np.float32)])
    assert accel.enable("model", lambda: "stock") == ("stock", False)
    _patch(monkeypatch, [np.ones(3, np.float32)], [np.ones(3, np.float32)], fail=True)
    assert accel.enable("model", lambda: "stock") == ("stock", False)
