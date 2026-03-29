"""utils/quicklook.py — PNG thumbnail generation for pipeline outputs."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def save_quicklook(
    da: xr.DataArray,
    path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
    title: str = "",
) -> None:
    """Save a PNG thumbnail of a DataArray.

    Works with 2-D (y, x) or 3-band (3, y, x) arrays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    arr = da.values
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] >= 3:
            # RGB composite: normalise each band
            rgb = arr[:3].transpose(1, 2, 0).astype(np.float32)
            for i in range(3):
                band = rgb[:, :, i]
                lo, hi = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
                rgb[:, :, i] = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)
            ax.imshow(rgb)
            ax.set_title(title)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(str(path), bbox_inches="tight")
            plt.close(fig)
            logger.info("Quicklook saved: %s", path)
            return

    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Quicklook saved: %s", path)
