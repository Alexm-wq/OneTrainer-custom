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


def _restore_pyside6_plugin_path() -> None:
    """Keep OpenCV's bundled Qt plugins from shadowing PySide6 on Linux.

    OneTrainer uses the headless OpenCV wheel, but stale environments can still
    contain the GUI wheel and its cv2/qt/plugins path. Force PySide6's own Qt
    plugin tree before QApplication loads the platform plugin.
    """
    if not sys.platform.startswith("linux"):
        return

    pyside_root = Path(PySide6.__file__).resolve().parent
    plugin_root = pyside_root / "Qt" / "plugins"
    platform_plugins = plugin_root / "platforms"
    if not platform_plugins.is_dir():
        print(f"[Qt] PySide6 platform plugin directory not found: {platform_plugins}")
        return

    os.environ.pop("QT_QPA_FONTDIR", None)
    os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugins)

    # Do not manually preload XCB/xkb shared libraries here. Mixing libraries
    # from the Pixi environment with the host X11 stack via RTLD_GLOBAL can
    # leave Qt in an ABI-inconsistent state that only crashes when the first
    # keyboard event reaches the xcb plugin. The launcher installs the normal
    # Linux Qt/XCB runtime dependencies instead.
    QCoreApplication.setLibraryPaths([str(plugin_root)])


def create_application() -> QApplication:
    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

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
