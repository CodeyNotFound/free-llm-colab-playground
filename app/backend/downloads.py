from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .inference.models import GGUFFile


def safe_cache_dir(base_dir: str | Path, repo_id: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id).strip(".-")
    path = Path(base_dir).resolve() / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_gguf(
    gguf: GGUFFile,
    base_dir: str | Path = "models",
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    target = safe_cache_dir(base_dir, gguf.repo_id)
    filenames = list(gguf.split_files) or [gguf.filename]
    existing = [target / Path(name).name for name in filenames]
    if all(path.is_file() and path.stat().st_size > 0 for path in existing):
        if progress:
            progress(1.0, "Already cached")
        return existing[0]
    if progress:
        progress(0.05, f"Downloading {len(filenames)} GGUF file(s)…")
    if len(filenames) == 1:
        downloaded = hf_hub_download(gguf.repo_id, filenames[0], local_dir=target, local_dir_use_symlinks=False)
        result = Path(downloaded)
    else:
        snapshot_download(
            gguf.repo_id,
            allow_patterns=filenames,
            local_dir=target,
            local_dir_use_symlinks=False,
        )
        result = target / Path(filenames[0])
    if progress:
        progress(1.0, "Download complete")
    return result
