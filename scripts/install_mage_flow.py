from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
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
        _run(
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            requirement,
        )
        importlib.invalidate_caches()


def _installed_mage_revision() -> str | None:
    try:
        distribution = importlib.metadata.distribution("mage-flow")
    except importlib.metadata.PackageNotFoundError:
        return None

    try:
        direct_url_text = distribution.read_text("direct_url.json")
        if not direct_url_text:
            return None
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url.get("vcs_info") or {}
        commit_id = vcs_info.get("commit_id")
        return str(commit_id) if commit_id else None
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        return None


def _install_pinned_mage(*, force: bool = False) -> None:
    args = [
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
    ]
    if force:
        args.extend(["--force-reinstall", "--no-cache-dir"])
    args.append(MAGE_REQUIREMENT)
    _run(*args)
    importlib.invalidate_caches()


def ensure_mage_flow_runtime() -> None:
    """Ensure OneTrainer's pinned Mage runtime is importable without changing torch."""
    _ensure_pip()

    # These are the Mage runtime dependencies that are not guaranteed by
    # OneTrainer itself. The rest of Mage's dependency contract is already
    # satisfied by the CUDA13 environment. Gradio is only for Mage's standalone
    # app and is intentionally not installed.
    _install_if_missing("loguru", "loguru>=0.7.0")
    _install_if_missing("einops", "einops>=0.8.0")
    _install_if_missing("pydantic", "pydantic>=2.0")

    installed_revision = _installed_mage_revision()
    if (
        importlib.util.find_spec("mage_flow") is None
        or installed_revision != MAGE_REVISION
    ):
        print(
            "[Mage-Flow] Installing pinned runtime "
            f"{MAGE_REVISION[:12]} without modifying torch/torchvision..."
        )
        _install_pinned_mage(force=installed_revision is not None)

    importlib.invalidate_caches()
    try:
        import mage_flow.pipeline  # noqa: F401
    except ImportError:
        # Repair a partial/corrupt install once before surfacing the real error.
        print("[Mage-Flow] Import failed; repairing the pinned runtime once...")
        _install_pinned_mage(force=True)
        importlib.invalidate_caches()
        import mage_flow.pipeline  # noqa: F401

    final_revision = _installed_mage_revision()
    if final_revision != MAGE_REVISION:
        raise RuntimeError(
            "Mage-Flow installed successfully but its VCS revision could not be "
            f"verified as {MAGE_REVISION} (found {final_revision!r})."
        )

    print(f"[Mage-Flow] Runtime OK ({MAGE_REVISION[:12]})")


def main() -> None:
    ensure_mage_flow_runtime()


if __name__ == "__main__":
    main()
