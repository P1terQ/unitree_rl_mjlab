"""Robot asset loading helpers."""

from pathlib import Path


def update_assets(assets: dict[str, bytes], assets_dir: Path, meshdir: str) -> None:
  """Populate a MuJoCo assets dict from a robot mesh directory."""
  if not assets_dir.exists():
    return
  for path in assets_dir.rglob("*"):
    if not path.is_file():
      continue
    key = Path(meshdir) / path.relative_to(assets_dir)
    assets[str(key)] = path.read_bytes()
