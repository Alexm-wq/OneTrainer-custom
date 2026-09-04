import ctypes
import os
import signal
import sys
from abc import ABCMeta
from pathlib import Path

import PySide6
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget


class QtABCMeta(type(QWidget), ABCMeta):
    # Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta.
    pass


def _preload_linux_xcb_runtime() -> None:
    """Prefer the XCB libraries shipped in the active Pixi environment.

    Qt 6.5+ requires libxcb-cursor at platform-plugin load time. Vast images do
    not consistently provide that Ubuntu package system-wide, while OneTrainer's
    Linux Pixi environment already declares xcb-util-cursor. Preloading the
    environment copies with RTLD_GLOBAL keeps the Qt plugin independent of the
    host image's optional desktop packages.
    """
    if not sys.platform.startswith("linux"):
        return

    lib_dir = Path(sys.prefix) / "lib"
    if not lib_dir.is_dir():
        return

    # Load dependencies before libxcb-cursor itself. Missing optional entries
    # are harmless; Qt will resolve whatever is available through its normal
    # loader path. The final cursor library is the important Qt 6.5 dependency.
    libraries = (
        "libxcb.so.1",
        "libxcb-render.so.0",
        "libxcb-shm.so.0",
        "libxcb-image.so.0",
        "libxcb-render-util.so.0",
        "libxcb-keysyms.so.1",
        "libxcb-icccm.so.4",
        "libxcb-xkb.so.1",
        "libxkbcommon.so.0",
        "libxkbcommon-x11.so.0",
        "libxcb-cursor.so.0",
    )

    for library in libraries:
        path = lib_dir / library
        if not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            # Do not turn a best-effort preload into a Python exception. If a
            # required dependency is genuinely absent Qt will print its normal
            # platform-plugin diagnostic below.
            print(f"[Qt] Could not preload {path.name}: {exc}")


def _restore_pyside6_plugin_path() -> None:
    """Keep OpenCV's bundled Qt plugins from shadowing PySide6 on Linux.

    The opencv-python wheel sets QT_QPA_PLATFORM_PLUGIN_PATH to its own
    cv2/qt/plugins directory when cv2 is imported. OneTrainer imports OpenCV
    before QApplication is constructed, which can make a Qt 6 PySide6 process
    try to load OpenCV's incompatible xcb plugin and abort during startup.

    Use PySide6's package directory directly instead of QLibraryInfo here: the
    latter can already be influenced by a contaminated plugin search path.
    """
    if not sys.platform.startswith("linux"):
        return

    pyside_root = Path(PySide6.__file__).resolve().parent
    plugin_root = pyside_root / "Qt" / "plugins"
    platform_plugins = plugin_root / "platforms"
    if not platform_plugins.is_dir():
        print(f"[Qt] PySide6 platform plugin directory not found: {platform_plugins}")
        return

    # OpenCV's non-headless wheel also exports QT_QPA_FONTDIR. It is irrelevant
    # to PySide6 and can point at another Qt installation, so discard it.
    os.environ.pop("QT_QPA_FONTDIR", None)
    os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugins)

    # Environment variables alone are insufficient if another imported module
    # has already changed Qt's internal library paths. Force the in-process Qt
    # search path as well before QApplication loads the xcb platform plugin.
    QCoreApplication.setLibraryPaths([str(plugin_root)])


def create_application() -> QApplication:
    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    _preload_linux_xcb_runtime()
    _restore_pyside6_plugin_path()
    app = QApplication(sys.argv)
    # Force Fusion everywhere: native styles (e.g. windowsvista) draw standard
    # controls via OS theme APIs, which breaks once an application stylesheet
    # is set, producing a flatter look than Fusion's own stylesheet-aware painting.
    app.setStyle(QStyleFactory.create("Fusion"))
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)

    app.setStyleSheet("""
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
            padding: 2px 2px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QProgressBar {
            background-color: #c8c8c8;
        }
        QToolButton {
            padding-top: 0px;
            padding-bottom: 0px;
            padding-right: 40px;
        }
        QToolButton::menu-indicator {
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 12px;
            height: 12px;
            right: 10px;
        }
    """)

    return app
