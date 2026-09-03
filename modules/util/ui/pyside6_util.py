import os
import signal
import sys
from abc import ABCMeta

from PySide6.QtCore import QLibraryInfo, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget


class QtABCMeta(type(QWidget), ABCMeta):
    # Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta.
    pass


def _restore_pyside6_plugin_path() -> None:
    """Keep OpenCV's bundled Qt plugins from shadowing PySide6 on Linux.

    The opencv-python wheel sets QT_QPA_PLATFORM_PLUGIN_PATH to its own
    cv2/qt/plugins directory when cv2 is imported. OneTrainer imports OpenCV
    before QApplication is constructed, which can make a Qt 6 PySide6 process
    try to load OpenCV's incompatible xcb plugin and abort during startup.
    """
    if not sys.platform.startswith("linux"):
        return

    plugin_root = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    platform_plugins = os.path.join(plugin_root, "platforms")
    if not os.path.isdir(platform_plugins):
        return

    os.environ["QT_PLUGIN_PATH"] = plugin_root
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_plugins


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
