"""Seed handling in the benchmark runner.

WHAT THESE TESTS ASSERT
-----------------------
That ``RunConfig.seeds`` actually controls the run: a stochastic method scored by
a stochastic metric gives identical rows under the same seed, different rows
under a different one, and the same rows whether or not the run was interrupted
and resumed. Every row records the seed it was produced under.

They use stub registry entries (a noise "attribution", a noisy "metric") so the
torch and NumPy generators are both exercised with no model download and none of
the optional benchmark dependencies.
"""
import json
import os

import numpy as np
import pytest
import torch

from xai_bench import isolation
from xai_bench.config import RunConfig
from xai_bench.registry import DATASETS, METHODS, METRICS, MODELS
from xai_bench.runner import BenchmarkRunner

N_IMAGES = 3


class _StubWrapper:
    is_vit = False
    supports_attention = False

    def __init__(self):
        self.model = torch.nn.Linear(4, 2)


def _noise_method(model_wrapper, model=None, **kwargs):
    return lambda inputs, target: torch.rand(1, 8, 8)                 # torch generator


def _noisy_metric(attr, **kwargs):
    return float(attr.sum()) + float(np.random.rand())                # NumPy generator


def _dataset(**kwargs):
    return [(torch.zeros(3, 8, 8), 0, {"img_name": f"img{i}"}) for i in range(N_IMAGES)]


@pytest.fixture
def stub_bench():
    entries = [(MODELS, "stub_model", _StubWrapper), (METHODS, "stub_noise", _noise_method),
               (DATASETS, "stub_data", _dataset), (METRICS, "stub_metric", _noisy_metric)]
    for registry, name, fn in entries:
        registry.add(name, fn)
    yield
    for registry, name, _ in entries:
        registry._entries.pop(name)


def _config(tmp_path, run_name, seeds):
    return RunConfig(models=["stub_model"], methods=["stub_noise"], metrics=["stub_metric"],
                     dataset="stub_data", seeds=seeds, device="cpu",
                     output_dir=str(tmp_path), run_name=run_name)


def _run(tmp_path, run_name, seeds=(0,)):
    BenchmarkRunner(_config(tmp_path, run_name, list(seeds))).run()
    with open(os.path.join(tmp_path, run_name, "results.json")) as f:
        return json.load(f)


def _scores(rows):
    return {(r["seed"], r["image_idx"]): r["stub_metric"] for r in rows}


def test_same_seed_reproduces_a_stochastic_run(tmp_path, stub_bench):
    assert _scores(_run(tmp_path, "a")) == _scores(_run(tmp_path, "b"))


def test_a_different_seed_changes_a_stochastic_run(tmp_path, stub_bench):
    s0 = _scores(_run(tmp_path, "a", seeds=[0]))
    s1 = _scores(_run(tmp_path, "b", seeds=[1]))
    assert all(s0[(0, i)] != s1[(1, i)] for i in range(N_IMAGES))


def test_every_row_records_its_seed_and_all_seeds_are_swept(tmp_path, stub_bench):
    rows = _run(tmp_path, "a", seeds=[0, 7])
    assert sorted((r["seed"], r["image_idx"]) for r in rows) == \
        [(s, i) for s in (0, 7) for i in range(N_IMAGES)]
    assert _scores(rows)[(0, 0)] != _scores(rows)[(7, 0)]


def test_resumed_run_matches_an_uninterrupted_one(tmp_path, stub_bench):
    full = _run(tmp_path, "full")
    os.makedirs(tmp_path / "resumed")
    with open(tmp_path / "resumed" / "results.json", "w") as f:
        json.dump(full[:1], f)                                        # crashed after image 0
    assert _scores(_run(tmp_path, "resumed")) == _scores(full)


def test_rows_from_before_seed_recording_are_kept_not_recomputed(tmp_path, stub_bench, capsys):
    legacy = [{"model": "stub_model", "method": "stub_noise", "image_idx": i,
               "image_name": f"img{i}", "target": 0, "time_ms": 1.0, "stub_metric": 42.0}
              for i in range(N_IMAGES)]
    os.makedirs(tmp_path / "legacy")
    with open(tmp_path / "legacy" / "results.json", "w") as f:
        json.dump(legacy, f)
    rows = _run(tmp_path, "legacy")
    assert rows == legacy
    assert "NOT produced under a fixed seed" in capsys.readouterr().out


def test_an_empty_seed_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="seeds"):
        _config(tmp_path, "a", seeds=[])


def test_runner_refuses_to_start_in_a_patched_process(tmp_path, stub_bench, monkeypatch):
    monkeypatch.setattr(isolation, "live_patches", lambda: ["hilrp.swin_lxt"])
    monkeypatch.setitem(MODELS._entries, "stub_model",
                        lambda: pytest.fail("model loaded before the isolation check"))
    with pytest.raises(RuntimeError, match="fresh Python process"):
        BenchmarkRunner(_config(tmp_path, "a", seeds=[0])).run()
