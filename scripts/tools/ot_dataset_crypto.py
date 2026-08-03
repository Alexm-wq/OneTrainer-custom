#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import getpass
import os
import secrets
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "mgds" / "src"))

try:
    from mgds.crypto import (
        EncryptedReader,
        SOURCE_PURPOSE,
        decrypt_file,
        encrypt_file,
        is_encrypted_file,
    )
except ModuleNotFoundError as exc:
    if exc.name == "cryptography":
        raise SystemExit(
            "The 'cryptography' package is required. Run this tool with the "
            "OneTrainer pixi Python, or install the updated OneTrainer dependencies."
        ) from exc
    raise


TRAINING_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".txt",
    ".webp",
}


def _read_key_file(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if value == "":
        raise SystemExit(f"Key file is empty: {path}")
    return value


def _get_key(args: argparse.Namespace, *, confirm: bool) -> str:
    if args.key_file is not None:
        return _read_key_file(args.key_file)

    first = getpass.getpass("Encryption key: ")
    if first == "":
        raise SystemExit("Encryption key cannot be empty.")
    if confirm:
        second = getpass.getpass("Confirm encryption key: ")
        if first != second:
            raise SystemExit("Encryption keys do not match.")
    return first


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_tree_paths(source: Path, destination: Path, key_file: Path | None) -> None:
    if not source.is_dir():
        raise SystemExit(f"Input directory does not exist: {source}")
    if source == destination:
        raise SystemExit("Input and output must be different directories.")
    if _inside(destination, source):
        raise SystemExit("Output cannot be placed inside the input dataset.")
    if key_file is not None and _inside(key_file.resolve(), source):
        raise SystemExit(
            "The key file must not be stored inside the input dataset; it could "
            "otherwise be uploaded with the encrypted data."
        )


def _extensions(args: argparse.Namespace) -> set[str]:
    values = set(TRAINING_EXTENSIONS)
    for extension in args.extension:
        value = extension.lower()
        values.add(value if value.startswith(".") else f".{value}")
    return values


def _prepare_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {path}\n"
            "Use a new directory to avoid mixing partial or stale output."
        )
    path.mkdir(parents=True, exist_ok=True)


def _copy_metadata(source: Path, destination: Path) -> None:
    try:
        shutil.copystat(source, destination, follow_symlinks=False)
    except OSError:
        pass


def _transform_tree(
        source: Path,
        destination: Path,
        key: str,
        *,
        decrypt: bool,
        extensions: set[str],
) -> None:
    _prepare_destination(destination)
    salt = None
    encrypted_count = 0
    decrypted_count = 0
    copied_count = 0

    files = sorted(path for path in source.rglob("*") if path.is_file() or path.is_symlink())
    for index, source_path in enumerate(files, start=1):
        if source_path.is_symlink():
            raise SystemExit(
                f"Refusing to follow dataset symlink: {source_path}\n"
                "Replace it with a real file before encryption."
            )

        relative = source_path.relative_to(source)
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        encrypted_source = is_encrypted_file(source_path)

        if decrypt and encrypted_source:
            decrypt_file(source_path, destination_path, key)
            decrypted_count += 1
        elif not decrypt and encrypted_source:
            # A mixed input tree is supported, but every encrypted member must
            # use the same key because OneTrainer intentionally keeps one
            # process-local dataset key.
            with EncryptedReader(source_path, SOURCE_PURPOSE, secret=key) as reader:
                while reader.read(4 * 1024 * 1024):
                    pass
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        elif not decrypt and source_path.suffix.lower() in extensions and not encrypted_source:
            salt = encrypt_file(
                source_path,
                destination_path,
                key,
                salt=salt,
            )
            encrypted_count += 1
        else:
            shutil.copy2(source_path, destination_path)
            copied_count += 1

        _copy_metadata(source_path, destination_path)
        if index % 100 == 0 or index == len(files):
            print(f"\rProcessed {index}/{len(files)} files", end="", flush=True)

    if files:
        print()
    print(
        f"Done: encrypted={encrypted_count}, decrypted={decrypted_count}, "
        f"copied={copied_count}, output={destination}"
    )


def _verify_tree(source: Path, key: str) -> None:
    encrypted_count = 0
    plaintext_count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if is_encrypted_file(path):
            with EncryptedReader(path, SOURCE_PURPOSE, secret=key) as reader:
                while reader.read(4 * 1024 * 1024):
                    pass
            encrypted_count += 1
        else:
            plaintext_count += 1
    print(
        f"Verified {encrypted_count} encrypted files; "
        f"{plaintext_count} ordinary files were left unchanged."
    )


def _keygen(output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing key file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    value = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
    output.write_text(value + "\n", encoding="utf-8")
    try:
        os.chmod(output, 0o600)
    except OSError:
        pass
    print(f"Created key file: {output}")
    print("Keep this file off the cloud instance and out of the dataset directory.")


def _add_key_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key-file",
        type=Path,
        help="Read the key from this local UTF-8 file instead of prompting.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encrypt/decrypt OneTrainer images and captions while preserving "
            "their extensions and directory layout."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    keygen = commands.add_parser("keygen", help="Create a strong random key file.")
    keygen.add_argument("--output", type=Path, required=True)

    for name in ("encrypt", "decrypt"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--extension",
            action="append",
            default=[],
            help="Also encrypt this extension; repeat as needed.",
        )
        _add_key_argument(command)

    verify = commands.add_parser("verify")
    verify.add_argument("--input", type=Path, required=True)
    _add_key_argument(verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "keygen":
        _keygen(args.output.expanduser().resolve())
        return

    source = args.input.expanduser().resolve()
    key_file = args.key_file.expanduser().resolve() if args.key_file is not None else None
    if args.command == "verify":
        if not source.is_dir():
            raise SystemExit(f"Input directory does not exist: {source}")
        _verify_tree(source, _get_key(args, confirm=False))
        return

    destination = args.output.expanduser().resolve()
    _validate_tree_paths(source, destination, key_file)
    _transform_tree(
        source,
        destination,
        _get_key(args, confirm=args.command == "encrypt"),
        decrypt=args.command == "decrypt",
        extensions=_extensions(args),
    )


if __name__ == "__main__":
    main()
