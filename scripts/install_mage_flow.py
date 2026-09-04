from __future__ import annotations

import importlib.util
import subprocess
import sys


MAGE_REVISION = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
MAGE_REQUIREMENT = (
    "mage-flow @ git+https://github.com/microsoft/Mage.git@"
    f"{MAGE_REVISION}#subdirectory=mage_flow"
)


def _run(*args: str) -> None:
    subprocess.check_call([sys.executable, *args])


def _ensure_pip() -> None:
    if importlib.util.find_spec("pip") is None:
        _run("-m", "ensurepip", "--upgrade")


def _install_if_missing(module: str, requirement: str) -> None:
    if importlib.util.find_spec(module) is None:
        _run("-m", "pip", "install", "--disable-pip-version-check", requirement)


def main() -> None:
    _ensure_pip()

    # Mage currently declares torch>=2.13/torchvision>=0.28 upstream, while
    # OneTrainer's CUDA13 environment deliberately remains on torch 2.12 /
    # torchvision 0.27. Install the pinned Mage source without dependency
    # resolution so it cannot replace the working OneTrainer torch stack.
    if importlib.util.find_spec("mage_flow") is None:
        _run(
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            MAGE_REQUIREMENT,
        )

    # Runtime dependencies used by the Mage model/pipeline path. Gradio is an
    # upstream standalone-app dependency and is intentionally not needed here.
    _install_if_missing("loguru", "loguru>=0.7.0")
    _install_if_missing("einops", "einops>=0.8.0")
    _install_if_missing("pydantic", "pydantic>=2.0")

    import mage_flow.pipeline  # noqa: F401

    print(f"Mage-Flow runtime OK ({MAGE_REVISION[:12]})")


if __name__ == "__main__":
    main()
