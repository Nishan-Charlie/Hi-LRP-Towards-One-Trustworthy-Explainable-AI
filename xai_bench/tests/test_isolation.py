"""Process-isolation rules for the class-patching attribution methods.

WHAT THESE TESTS ASSERT
-----------------------
That a sweep which cannot be honest inside one interpreter is REJECTED, both
statically (when the config is built) and dynamically (when a patch is already
live in the process). They load no model and no backend: a live patch is
simulated by planting a module that carries the ports' own ``_PATCHED`` flag.

They do NOT assert that patched gradients differ from clean ones. That is the
premise (see xai_bench/isolation.py), established by the two-pass Shapley runs.
"""
import pathlib
import types

import pytest

from xai_bench import isolation
from xai_bench.config import RunConfig

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"
FLAT, HIER = "vit_base_patch16_224", "swin_base_patch4_window7_224"


# ------------------------------------------------------------------ static
def test_baseline_sweep_is_accepted():
    RunConfig(models=[FLAT, HIER], methods=["saliency", "grad_cam", "rise"])


@pytest.mark.parametrize("method", sorted(isolation.PATCHING_METHODS))
def test_patching_method_alone_is_accepted(method):
    RunConfig(models=[HIER], methods=[method])


@pytest.mark.parametrize("methods", [
    ["hilrp", "grad_cam"],
    ["saliency", "attnlrp"],
    ["hilrp", "attnlrp"],
    ["HiLRP", "grad_cam"],          # the registry is case-insensitive, so is the rule
])
def test_patching_method_mixed_with_anything_is_rejected(methods):
    with pytest.raises(ValueError, match="cannot share one process"):
        RunConfig(models=[HIER], methods=methods)


def test_attnlrp_cannot_sweep_flat_and_hierarchical_backbones_together():
    with pytest.raises(ValueError, match="flat ViTs"):
        RunConfig(models=[FLAT, HIER], methods=["attnlrp"])
    RunConfig(models=[FLAT, "deit_base_patch16_224"], methods=["attnlrp"])
    RunConfig(models=[HIER, "pvt_v2_b2"], methods=["attnlrp"])


def test_hilrp_may_sweep_flat_and_hierarchical_backbones_together():
    RunConfig(models=[FLAT, HIER], methods=["hilrp"])


@pytest.mark.skipif(not CONFIG_DIR.is_dir(), reason="configs/ not shipped with this install")
@pytest.mark.parametrize("path", sorted(CONFIG_DIR.rglob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_is_runnable_in_one_process(path):
    RunConfig.from_yaml(str(path))


# ----------------------------------------------------------------- dynamic
def test_live_patches_reads_the_ports_own_flags(monkeypatch):
    port = types.ModuleType("xai_bench.methods.hilrp._planted_lxt")
    monkeypatch.setitem(isolation.sys.modules, port.__name__, port)
    assert port.__name__ not in isolation.live_patches()      # imported, not yet patched
    port._PATCHED = True
    assert port.__name__ in isolation.live_patches()


@pytest.fixture
def live(monkeypatch):
    """Set which modules report a live class-level patch in this process."""
    def _set(*modules):
        monkeypatch.setattr(isolation, "live_patches", lambda: sorted(modules))
    return _set


def test_clean_process_runs_anything(live):
    live()
    isolation.check_process(["saliency", "grad_cam"], [HIER])
    isolation.check_process(["hilrp"], [HIER])
    isolation.check_process(["attnlrp"], [HIER])


def test_baselines_refuse_a_process_with_a_live_patch(live):
    live("xai_bench.methods.hilrp.swin_lxt")
    with pytest.raises(RuntimeError, match="fresh Python process"):
        isolation.check_process(["saliency", "grad_cam"], [HIER])


def test_hilrp_runs_alongside_its_own_ports(live):
    live("xai_bench.methods.hilrp.swin_lxt", "xai_bench.methods.hilrp.pvt_lxt")
    isolation.check_process(["hilrp"], [HIER, "pvt_v2_b2"])


def test_naive_attnlrp_refuses_a_process_patched_by_a_hilrp_port(live):
    live("xai_bench.methods.hilrp.vit_lxt")
    with pytest.raises(RuntimeError, match="naive attnlrp"):
        isolation.check_process(["attnlrp"], [HIER])
    isolation.check_process(["attnlrp"], [FLAT])             # flat recipe IS that port


def test_naive_attnlrp_tolerates_its_own_patch(live):
    live("xai_bench.methods.attnlrp_method")
    isolation.check_process(["attnlrp"], [HIER])
