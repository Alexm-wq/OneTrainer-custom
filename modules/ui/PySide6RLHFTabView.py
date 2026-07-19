from modules.ui.BaseRLHFTabView import BaseRLHFTabView
from modules.ui.RLHFTabController import RLHFTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtWidgets import QWidget


class PySide6RLHFTabView(BaseRLHFTabView, QWidget, metaclass=QtABCMeta):
    def __init__(self, master, controller: RLHFTabController, ui_state):
        QWidget.__init__(self, master)
        BaseRLHFTabView.__init__(self, pyside6_components)

        self.controller = controller
        self.ui_state = ui_state

        scroll, frame = pyside6_components.scrollable_frame(self)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)

        layout = pyside6_components._layout(frame)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        self.build_content(frame, controller, ui_state)
        pyside6_components._pack_form(frame)
