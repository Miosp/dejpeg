import os
import sys

import pytest

# ensure src layout is importable when running `uv run pytest` before install
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Path to the sibling inference-core TS twin parser, relative to this package.
_REPO = os.path.dirname(_ROOT)
TS_PARSER = os.path.join(
    _REPO, "packages", "inference-core", "src", "codec", "jpegMeta.ts"
)
# Runner that prints canonical JSON for one fixture path.
TS_RUNNER = os.path.join(os.path.dirname(__file__), "_jpegmeta_runner.ts")


def _resolve_bun() -> str | None:
    """Find bun in PATH or at the WSL install location ~/.bun/bin/bun."""
    from shutil import which

    b = which("bun")
    if b:
        return b
    home = os.path.expanduser("~/.bun/bin/bun")
    if os.path.exists(home):
        return home
    return None


@pytest.fixture(scope="session")
def bun() -> str:
    b = _resolve_bun()
    if b is None:
        pytest.skip("bun not installed (needed for cross-language parser test)")
    return b
