"""Display normalization for relevance maps.

Conservation-based relevance is heavy-tailed: a small fraction of pixels can
carry a large share of the mass (on MobileViT's deep separable-conv stem, ~27%
of the absolute mass sits in the top 1% of pixels). Naive symmetric-max
normalization then maps everything but a few spikes to near-white, which reads
as a blank heatmap even though the map localizes correctly.

We therefore normalize for display by a robust percentile of the absolute
relevance (the standard practice for Grad-CAM / IG), optionally with a light
Gaussian smoothing that is clearly a post-hoc visualization step and does not
touch the underlying relevance values used for any metric.
"""
import numpy as np


def normalize_for_display(pixel_map, percentile=99.0, smooth_sigma=0.0):
    """Return (display_map, vmax) for a diverging colormap centered at 0.

    Args:
        pixel_map: [H, W] signed relevance (numpy array or torch tensor).
        percentile: clip magnitude to this percentile of |relevance| (99 = show
            structure, not just the top spikes). Use 100 for raw symmetric-max.
        smooth_sigma: optional Gaussian blur (display only). 0 disables.

    Returns:
        (m, vmax) with m clipped to [-vmax, vmax]; pass to imshow(cmap='bwr',
        vmin=-vmax, vmax=vmax).
    """
    m = pixel_map.detach().cpu().numpy() if hasattr(pixel_map, "detach") else np.asarray(pixel_map)
    m = m.astype(np.float64)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(m, sigma=smooth_sigma)
    vmax = np.percentile(np.abs(m), percentile) + 1e-12
    return np.clip(m, -vmax, vmax), vmax


def show_relevance(ax, pixel_map, title=None, percentile=99.0, smooth_sigma=0.0, cmap="bwr"):
    """imshow a relevance map on a matplotlib axis with robust normalization."""
    m, vmax = normalize_for_display(pixel_map, percentile, smooth_sigma)
    ax.imshow(m, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")
    return ax
