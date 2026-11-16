"""Failure accounting, timing, ViT Grad-CAM target layer and input statistics.

WHAT THESE TESTS ASSERT
-----------------------
Runner
  * an attribution that raises is recorded as a row with an "error" field and
    counted in summary.csv (n_failed) and failures.json, not dropped;
  * a resumed run retries the failed evaluations;
  * a metric that raises leaves the other metrics of the row intact;
  * a method that cannot be set up is listed in failures.json;
  * attribution time is recorded per row and summarised in summary.csv.
Grad-CAM
  * on a class-token ViT the spatial tokens of forward_features receive zero
    gradient (the July-November 2026 bug), and the bench's Grad-CAM reads a layer
    that does receive it;
  * a CAM whose target layer gets no spatial gradient raises instead of
    returning an all-zero map.
Input statistics
  * a model trained with non-ImageNet statistics (ViT, mean = std = 0.5) is wrapped
    so that the bench's ImageNet-normalised images reach it in its own input space.

All models are tiny and randomly initialised: no download, CPU only.
"""
import json

import pandas as pd
import pytest
import torch

from xai_bench.config import RunConfig
from xai_bench.methods.cam_methods import _check_spatial_gradient, _generic_gradcam
from xai_bench.models.timm_models import InputStatsAdapter, ModelWrapper, create_timm_model
from xai_bench.registry import DATASETS, METHODS, METRICS, MODELS
from xai_bench.runner import BenchmarkRunner

N_IMAGES = 3
FAIL_ON = {"image": 1}          # image index the flaky method fails on (None: never)
CONSTANT_ON = {"image": None}   # image index on which it returns a constant map
MAP_SUM = float(torch.arange(64).sum())


class _StubWrapper:
    is_vit = False
    supports_attention = False

    def __init__(self):
        self.model = torch.nn.Linear(4, 2)


def _flaky_method(model_wrapper, model=None, **kwargs):
    def method(inputs, target):
        if FAIL_ON["image"] is not None and float(inputs.flatten()[0]) == FAIL_ON["image"]:
            raise RuntimeError("CUDA out of memory (simulated)")
        if CONSTANT_ON["image"] is not None and float(inputs.flatten()[0]) == CONSTANT_ON["image"]:
            return torch.zeros(1, 8, 8)
        return torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
    return method


def _broken_setup(model_wrapper, model=None, **kwargs):
    raise ValueError("no rule for this layer (simulated)")


def _good_metric(attr, **kwargs):
    return float(attr.sum())


def _bad_metric(attr, **kwargs):
    raise ZeroDivisionError("simulated")


def _dataset(**kwargs):
    # image i is filled with the value i, so the flaky method can recognise it
    return [(torch.full((3, 8, 8), float(i)), 0, {"img_name": f"img{i}"}) for i in range(N_IMAGES)]


@pytest.fixture
def stub_bench():
    FAIL_ON["image"] = 1
    CONSTANT_ON["image"] = None
    entries = [(MODELS, "stub_model", _StubWrapper), (METHODS, "stub_flaky", _flaky_method),
               (METHODS, "stub_broken", _broken_setup), (DATASETS, "stub_data", _dataset),
               (METRICS, "stub_good", _good_metric), (METRICS, "stub_bad", _bad_metric)]
    for registry, name, fn in entries:
        registry.add(name, fn)
    yield
    for registry, name, _ in entries:
        registry._entries.pop(name)


def _run(tmp_path, run_name, methods=("stub_flaky",), metrics=("stub_good",)):
    cfg = RunConfig(models=["stub_model"], methods=list(methods), metrics=list(metrics),
                    dataset="stub_data", seeds=[0], device="cpu",
                    output_dir=str(tmp_path), run_name=run_name)
    BenchmarkRunner(cfg).run()
    d = tmp_path / run_name
    with open(d / "results.json") as f:
        rows = json.load(f)
    with open(d / "failures.json") as f:
        failures = json.load(f)
    return rows, pd.read_csv(d / "summary.csv"), failures


def test_a_failed_image_is_recorded_and_counted(tmp_path, stub_bench):
    rows, summary, failures = _run(tmp_path, "a")
    assert len(rows) == N_IMAGES
    bad = [r for r in rows if r.get("error")]
    assert [r["image_idx"] for r in bad] == [1]
    assert "out of memory" in bad[0]["error"] and bad[0]["stub_good"] is None
    s = summary.iloc[0]
    assert s["n_evaluations"] == N_IMAGES and s["n_failed"] == 1
    assert s["stub_good_count"] == N_IMAGES - 1          # mean over the good images only
    assert [e["image_idx"] for e in failures["evaluations"]] == [1]


def test_a_resumed_run_retries_failed_images(tmp_path, stub_bench):
    _run(tmp_path, "a")
    FAIL_ON["image"] = None                              # the transient failure is gone
    rows, summary, failures = _run(tmp_path, "a")
    assert sorted(r["image_idx"] for r in rows) == list(range(N_IMAGES))
    assert not any(r.get("error") for r in rows)
    assert summary.iloc[0]["n_failed"] == 0 and failures["evaluations"] == []


def test_a_failing_metric_keeps_the_other_metrics(tmp_path, stub_bench):
    FAIL_ON["image"] = None
    rows, summary, _ = _run(tmp_path, "a", metrics=("stub_good", "stub_bad"))
    assert all(r["stub_good"] == MAP_SUM and r["stub_bad"] is None for r in rows)
    assert all("ZeroDivisionError" in r["metric_errors"]["stub_bad"] for r in rows)
    assert summary.iloc[0]["stub_bad_undefined"] == N_IMAGES


def test_a_method_that_cannot_be_set_up_is_listed(tmp_path, stub_bench):
    FAIL_ON["image"] = None
    _, _, failures = _run(tmp_path, "a", methods=("stub_flaky", "stub_broken"))
    assert "no rule for this layer" in failures["method_init"]["stub_model/stub_broken"]


def test_attribution_time_is_recorded_and_summarised(tmp_path, stub_bench):
    rows, summary, _ = _run(tmp_path, "a")
    assert all(r["time_ms"] >= 0 for r in rows if not r.get("error"))
    assert summary.iloc[0]["time_ms_mean"] >= 0 and "time_ms_median" in summary.columns


# ------------------------------------------------------------------ constant maps
def test_a_constant_map_is_flagged_and_kept(tmp_path, stub_bench):
    FAIL_ON["image"], CONSTANT_ON["image"] = None, 2
    rows, summary, _ = _run(tmp_path, "a")
    flags = {r["image_idx"]: r["degenerate_map"] for r in rows}
    assert flags == {0: False, 1: False, 2: True}
    assert [r["stub_good"] for r in rows if r["image_idx"] == 2] == [None]   # raw: undefined
    s = summary.iloc[0]
    assert s["n_evaluations"] == N_IMAGES and s["n_constant_maps"] == 1 and s["stub_good_undefined"] == 1


def test_constant_maps_score_zero_credit_and_an_undefined_ms_is_unbounded():
    from xai_bench import scoring
    df = pd.DataFrame({"model": ["m"] * 4 + ["n"] * 2, "method": ["x"] * 6, "error": [None] * 6,
                       "degenerate_map": [False, False, True, True, False, True],
                       "pointing_game": [1.0, 0.0, None, None, 1.0, None],
                       "sparseness": [0.6, 0.4, None, None, 0.5, None],
                       "max_sensitivity": [0.5, 2.0, None, None, 0.1, None],
                       "relevance_rank_accuracy": [0.3, 0.4, None, None, 0.2, None]})
    s = scoring.score(df, ["pointing_game", "sparseness", "max_sensitivity", "relevance_rank_accuracy"])
    assert s.pointing_game.tolist() == [1, 0, 0, 0, 1, 0]
    assert s.sparseness.tolist() == [0.6, 0.4, 0, 0, 0.5, 0]
    inf = float("inf")
    assert s.max_sensitivity.tolist() == [0.5, 2.0, inf, inf, 0.1, inf]   # failure = unbounded
    assert s.relevance_rank_accuracy.isna().sum() == 3                    # not covered: undefined
    assert scoring.cell_statistic("max_sensitivity") == "median"
    assert scoring.cell_statistic("pointing_game") == "mean"
    # the median counts the failures without letting them set the value
    assert s[s.model == "m"].max_sensitivity.median() == inf          # 2 of 4 failed
    assert s.max_sensitivity.iloc[[0, 1, 2]].median() == 2.0          # 1 of 3 failed


def test_localisation_and_multiseed_columns_follow_the_same_rule():
    from xai_bench import scoring
    df = pd.DataFrame({"model": ["m"] * 3, "method": ["x"] * 3, "degenerate": [0, 1, 0],
                       "ebpg_abs": [0.4, None, 0.6], "bboxpg_pos": [1.0, None, 0.0],
                       "msK": [0.2, None, 0.9]})
    s = scoring.score(df, ["ebpg_abs", "bboxpg_pos", "msK"], constant_col="degenerate")
    assert s.ebpg_abs.tolist() == [0.4, 0.0, 0.6] and s.bboxpg_pos.tolist() == [1.0, 0.0, 0.0]
    assert s.msK.tolist() == [0.2, float("inf"), 0.9]


def test_is_constant_map():
    from xai_bench import scoring
    assert scoring.is_constant_map(torch.zeros(4, 4)) and scoring.is_constant_map(torch.full((4, 4), 3.0))
    assert scoring.is_constant_map(torch.tensor([[1.0, float("nan")]]))
    assert not scoring.is_constant_map(torch.arange(4.0))


# ------------------------------------------------------------------ Grad-CAM on ViT
def _tiny_vit():
    from timm.models.vision_transformer import VisionTransformer
    torch.manual_seed(0)
    vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=32, depth=2, num_heads=2,
                            num_classes=5)                # class-token head (global_pool="token")
    return ModelWrapper(vit.eval(), "tiny_vit", cam_format="tokens_cls", supports_attention=True)


def test_forward_features_of_a_class_token_vit_gives_no_spatial_gradient():
    mw, x = _tiny_vit(), torch.randn(1, 3, 32, 32)
    feat = mw.model.forward_features(x)
    feat.retain_grad()
    mw.model.forward_head(feat)[0, 2].backward()
    assert feat.grad[:, 1:].abs().sum() == 0             # patch tokens: zero gradient
    assert feat.grad[:, 0].abs().sum() > 0               # only the class token is read


@pytest.mark.parametrize("plus_plus", [False, True])
def test_bench_gradcam_on_a_class_token_vit_reads_a_layer_with_spatial_gradient(plus_plus):
    cam = _generic_gradcam(_tiny_vit(), torch.randn(1, 3, 32, 32), target=2, plus_plus=plus_plus)
    assert cam.shape == (1, 32, 32) and torch.isfinite(cam).all()   # the guard did not fire


def test_a_cam_target_without_spatial_gradient_raises():
    with pytest.raises(RuntimeError, match="zero gradient"):
        _check_spatial_gradient(torch.zeros(1, 4, 2, 2), _tiny_vit())


# ------------------------------------------------------------------ input statistics
def test_vit_receives_its_own_input_statistics():
    mw = create_timm_model("vit_tiny_patch16_224", pretrained=False)
    assert isinstance(mw.model, InputStatsAdapter)       # ViT expects mean = std = 0.5
    raw = torch.rand(1, 3, 8, 8)
    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    bench_input = (raw - imnet_mean) / imnet_std          # what the bench's datasets produce
    assert torch.allclose(mw.model._adapt(bench_input), (raw - 0.5) / 0.5, atol=1e-5)


def test_imagenet_statistics_models_are_left_unwrapped():
    mw = create_timm_model("resnet18", pretrained=False)
    assert not isinstance(mw.model, InputStatsAdapter)
