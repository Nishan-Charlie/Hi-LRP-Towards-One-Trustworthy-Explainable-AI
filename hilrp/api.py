"""One entry point for every supported backbone.

    import hilrp
    res = hilrp.attribute(model, x)

Each architecture family has its own port (``swin_lxt``, ``pvt_lxt``, ...) because
each wires the rules into a different set of timm classes. This module picks the
port from the timm architecture name, so callers do not have to. The ports stay
importable on their own for anything family-specific (``attn_mode`` on Swin and
ViT, the label-free and CLIP scalars).

Ports are imported lazily: importing ``hilrp`` needs only torch, while the ports
need the attribution backend (``lxt`` and ``zennit``).
"""
import functools
import importlib
import sys

# family -> (timm architecture-name prefixes, port module, entry point)
_PORTS = {
    "swin":         (("swin",), "swin_lxt", "attribute_swin"),
    "pvt_v2":       (("pvt",), "pvt_lxt", "attribute_pvt"),
    "maxvit":       (("maxvit", "maxxvit"), "maxvit_lxt", "attribute_maxvit"),
    "mobilevit":    (("mobilevit",), "mobilevit_lxt", "attribute_mobilevit"),
    "efficientvit": (("efficientvit",), "efficientvit_lxt", "attribute_efficientvit"),
    "vit":          (("vit", "deit", "beit"), "vit_lxt", "attribute_vit"),
}

_BACKEND_HINT = (
    "HiLRP attribution needs the LXT backend, which is not installed. "
    'Install it with:  pip install "lxt==2.1" "zennit>=1.0"'
)


@functools.lru_cache(maxsize=None)        # a passing check is not repeated
def _check_zennit():
    """Refuse a zennit that silently changes the maps.

    Under zennit 0.5.x the BatchNorm-merge canonizer is not forward-exact, and
    MobileViT / EfficientViT relevance differs from every published number. The
    package metadata already requires >= 1.0; this catches a manual downgrade.
    """
    from importlib.metadata import version

    installed = version("zennit")
    if int(installed.split(".")[0]) < 1:
        raise ImportError(
            f"hilrp needs zennit>=1.0, found {installed}: with 0.5.x the BatchNorm "
            "merge is not forward-exact and MobileViT / EfficientViT maps change. "
            'Upgrade with:  pip install "zennit>=1.0"'
        )


def supported_architectures():
    """Map each supported family to the timm architecture-name prefixes it covers."""
    return {family: prefixes for family, (prefixes, _, _) in _PORTS.items()}


def attributor_for(arch):
    """Return the attribution function for a timm architecture name.

    >>> attributor_for("swin_tiny_patch4_window7_224").__name__
    'attribute_swin'
    """
    name = arch.lower()
    for prefixes, module, entry in _PORTS.values():
        if name.startswith(prefixes):
            try:
                port = importlib.import_module(f".{module}", __package__)
            except ModuleNotFoundError as exc:
                if (exc.name or "").split(".")[0] in ("lxt", "zennit"):
                    raise ImportError(_BACKEND_HINT) from exc
                raise
            _check_zennit()
            return getattr(port, entry)
    raise ValueError(
        f"no HiLRP port for architecture '{arch}'. Supported timm name prefixes: "
        f"{sorted(p for prefixes, _, _ in _PORTS.values() for p in prefixes)}"
    )


def _architecture_of(model):
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
    arch = cfg.get("architecture")
    if not arch:
        raise ValueError(
            "could not read the timm architecture name from the model; pass it "
            "explicitly, e.g. hilrp.attribute(model, x, arch='swin_tiny_patch4_window7_224')"
        )
    return arch


def attribute(model, x, target=None, *, arch=None, **kwargs):
    """Explain one prediction of a supported timm backbone in a single backward pass.

    Args:
        model: a timm model in eval mode. It is not modified; the rules are
            applied as class-level patches the first time a family is used.
        x: input batch ``[1, 3, H, W]``, normalized as the model expects.
        target: class index to explain. Default: the predicted class.
        arch: timm architecture name. Default: read from ``model.pretrained_cfg``.
        **kwargs: forwarded to the family port, e.g. ``gamma`` (default 0.25),
            ``conv_gamma``, and ``attn_mode`` on Swin and ViT.

    Returns:
        dict with
          ``target``, ``logit``   the explained class and its logit
          ``pixel_map``           ``[H, W]`` signed relevance at the input pixels
          ``stage_sums``          ``[(capture, sum(R) / logit), ...]``, the
                                  conservation trace from input to head
          ``stage_maps``          (Swin) ``[(capture, [h, w] map), ...]``, one
                                  relevance map per resolution stage

    Note:
        The patches are process-global and are never undone. Compute any
        gradient-based baseline you want to compare against BEFORE the first
        call, or in a separate process.
    """
    return attributor_for(arch or _architecture_of(model))(model, x, target=target, **kwargs)


def patched_ports():
    """Port modules whose class-level patches are live in this interpreter."""
    prefix = __package__ + "."
    return sorted(
        name for name, module in list(sys.modules.items())
        if module is not None and name.startswith(prefix) and getattr(module, "_PATCHED", False)
    )
