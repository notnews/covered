"""Render the Tier-A proportion figures from the speaker-mode HHI table.

Reads ``outputs/tables/hhi_speakers_modes.csv`` (written by
``scripts/trend_speakers.py``) and writes three figures to ``outputs/figures``:

    python scripts/trend_figures.py

* ``clip_topk_bands.png``  -- top-10 / next-15 / rest share of clip turns
* ``effective_voices.png`` -- 1/HHI for live vs clip
* ``clip_share.png``       -- clips as a share of all guest turns
* ``president_share.png``  -- sitting president's share of clip turns (Tier-1)

The first three need only the modes table; the last needs
``president_share_annual.csv`` (written alongside it by trend_speakers.py).
"""

import pandas as pd

from covered import figures
from covered.config import FIGURES, TABLES


def main() -> None:
    modes = pd.read_csv(TABLES / "hhi_speakers_modes.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)

    figures.plot_topk_bands(
        modes, FIGURES / "clip_topk_bands.png", variant="external-clip"
    )
    figures.plot_effective_voices(modes, FIGURES / "effective_voices.png")
    figures.plot_clip_share(modes, FIGURES / "clip_share.png")
    n = 3

    pres_path = TABLES / "president_share_annual.csv"
    if pres_path.exists():
        figures.plot_president_share(
            pd.read_csv(pres_path), FIGURES / "president_share.png"
        )
        n += 1

    print(f"wrote {n} figures -> {FIGURES}", flush=True)


if __name__ == "__main__":
    main()
