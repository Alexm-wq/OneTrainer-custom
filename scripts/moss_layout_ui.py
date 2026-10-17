"""Standalone MOSS image bucketing and phrase-to-box annotation tool."""
import argparse

from util.import_util import script_imports

script_imports()

from modules.ui.MOSSLayoutEditor import MOSSLayoutEditor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Open an image and its .txt / MOSS layout sidecar")
    parser.add_argument("--caption", help="Optional separate caption text file")
    parser.add_argument("--resolution", default="1024", help="Custom MGDS target: 1024, 768,1024, or 1024x768")
    parser.add_argument("--ui", choices=("qt", "ctk"), default="qt")
    args = parser.parse_args()
    if args.ui == "qt":
        from modules.util.ui.pyside6_util import create_application
        app = create_application()
        editor = MOSSLayoutEditor(None, qt=True)
    else:
        import customtkinter as ctk
        app = ctk.CTk()
        app.withdraw()
        editor = MOSSLayoutEditor(app, qt=False)
    editor._set("resolution", args.resolution)
    if args.image:
        editor.load_image(args.image)
    if args.caption:
        from pathlib import Path
        editor._set("summary", Path(args.caption).read_text(encoding="utf-8-sig").strip())
    if args.ui == "qt":
        app.exec()
    else:
        app.wait_window(editor.window)
        app.destroy()


if __name__ == "__main__":
    main()
