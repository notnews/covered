"""Download the CNN transcript CSVs from Harvard Dataverse.

The dataset is access-restricted (research use only), so a Dataverse API token
is required (env ``DATAVERSE_API_TOKEN``). Downloads are content-addressed: a
SHA-256 manifest is written so re-runs skip unchanged files and the analysis is
pinned to a specific dataset version.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from covered.config import (
    DATAVERSE_BASE_URL,
    DATAVERSE_DOI,
    DATAVERSE_VERSION,
    RAW,
    REFERENCE,
)
from covered.schema import validate_csv

if TYPE_CHECKING:
    from pyDataverse.api import DataAccessApi, NativeApi

__all__ = [
    "download_all",
    "extract_archives",
    "list_dataset_files",
    "load_raw_frames",
    "raw_files",
    "sha256_bytes",
]


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def _token() -> str:
    token = os.environ.get("DATAVERSE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "DATAVERSE_API_TOKEN is not set. Request access to "
            f"{DATAVERSE_DOI} on Dataverse and export the token."
        )
    return token


def _apis() -> tuple[NativeApi, DataAccessApi]:
    from pyDataverse.api import DataAccessApi, NativeApi

    token = _token()
    return (
        NativeApi(DATAVERSE_BASE_URL, token),
        DataAccessApi(DATAVERSE_BASE_URL, token),
    )


def list_dataset_files(version: str = DATAVERSE_VERSION) -> list[dict[str, object]]:
    """Return ``[{file_id, filename, md5}]`` for the pinned dataset version."""
    native, _ = _apis()
    resp = native.get_dataset(DATAVERSE_DOI, version=version)
    # pyDataverse types this as sync-or-async (union on whether an async httpx
    # client was passed to the constructor); _apis() never passes one, so this
    # is always the sync httpx.Response branch.
    data = resp.json()["data"]  # pyright: ignore[reportAttributeAccessIssue]
    files = data.get("latestVersion", data).get("files", data.get("files", []))
    out: list[dict[str, object]] = []
    for f in files:
        df = f.get("dataFile", {})
        out.append(
            {
                "file_id": df.get("id"),
                "filename": f.get("label") or df.get("filename"),
                "md5": df.get("md5") or df.get("checksum", {}).get("value"),
            }
        )
    return out


def download_all(
    dest: Path = RAW,
    version: str = DATAVERSE_VERSION,
    overwrite: bool = False,
) -> Path:
    """Download all dataset files to ``dest`` and write a checksum manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    _, access = _apis()
    manifest: list[dict[str, object]] = []
    for entry in list_dataset_files(version):
        filename = str(entry["filename"])
        target = dest / filename
        if target.exists() and not overwrite:
            content = target.read_bytes()
        else:
            # Same sync/async ambiguity as list_dataset_files above.
            datafile = access.get_datafile(entry["file_id"])
            content = datafile.content  # pyright: ignore[reportAttributeAccessIssue]
            target.write_bytes(content)
        manifest.append(
            {
                "filename": filename,
                "file_id": entry["file_id"],
                "size": len(content),
                "sha256": sha256_bytes(content),
                "dataverse_md5": entry["md5"],
            }
        )
    manifest_path = REFERENCE / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"doi": DATAVERSE_DOI, "version": version, "files": manifest},
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def extract_archives(raw: Path = RAW) -> list[Path]:
    """Extract any ``*.7z`` archives to CSV (older eras ship as 7-zip).

    Requires the ``7z`` CLI. Skips archives whose ``.csv`` already exists.
    Returns the list of CSV paths now present.
    """
    import shutil
    import subprocess

    for archive in sorted(raw.glob("*.7z")):
        if (raw / f"{archive.stem}.csv").exists():
            continue
        seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
        if not seven_zip:
            raise RuntimeError(
                "7z CLI not found; install p7zip to extract the .7z archives"
            )
        subprocess.run(  # noqa: S603 - seven_zip is resolved via shutil.which, args are fixed
            [seven_zip, "x", "-y", f"-o{raw}", str(archive)],
            check=True,
            capture_output=True,
        )
    return sorted(raw.glob("*.csv"))


def raw_files(raw: Path = RAW) -> list[Path]:
    """All per-era source files: extracted CSVs plus gzipped CSVs (pandas reads .gz)."""
    return sorted([*raw.glob("cnn-*.csv"), *raw.glob("cnn-*.csv.gz")])


def load_raw_frames(
    raw: Path = RAW,
    validate: bool = True,
    files: list[Path] | None = None,
) -> pd.DataFrame:
    """Load and concatenate the per-era source files into one validated frame.

    Pass ``files`` to load a subset (the full corpus is large — process era by
    era when memory is tight). 7-zip archives are extracted on demand.
    """
    extract_archives(raw)
    paths = files if files is not None else raw_files(raw)
    if not paths:
        raise FileNotFoundError(
            f"No source files in {raw}. Run `covered acquire` first."
        )
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return validate_csv(df) if validate else df
