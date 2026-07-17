from pathlib import Path
from types import SimpleNamespace

import warp as wp


SRC_PATH: Path = Path(__file__).parent


def _install_warp_context_compat() -> None:
  """Provide the Warp API shape expected by mjlab 1.2.0."""
  if hasattr(wp, "context") or not hasattr(wp, "get_cuda_driver_version"):
    return

  class _RuntimeCompat:
    @property
    def driver_version(self):
      return wp.get_cuda_driver_version()

  wp.context = SimpleNamespace(runtime=_RuntimeCompat())  # type: ignore[attr-defined]


_install_warp_context_compat()
