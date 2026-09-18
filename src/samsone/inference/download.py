"""Checkpoint discovery and download helpers for released SAMSONE models."""

from pathlib import Path

from filelock import FileLock
import requests

CACHE_DIR = Path.home() / ".cache" / "samsone"
RELEASE_URL_TEMPLATE = (
    "https://github.com/SamsungLabs/samsone/releases/download/v1.0.0/{filename}"
)


def get_checkpoint_path(
    filename: str, *, url: str | None = None, cache_dir: Path = CACHE_DIR
) -> Path:
    """Return a cached checkpoint, downloading it atomically when necessary."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / filename
    if checkpoint_path.is_file():
        return checkpoint_path

    download_url = url or RELEASE_URL_TEMPLATE.format(filename=filename)
    with FileLock(str(checkpoint_path) + ".lock"):
        if checkpoint_path.is_file():
            return checkpoint_path
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            with requests.get(download_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            temporary_path.replace(checkpoint_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    return checkpoint_path
