from modules.ui.BaseRLHFTabView import BaseRLHFTabView
from modules.ui.MageFlowEMAUI import add_dpo_timestep_controls, add_mage_linear_dpo_ema_controls
from modules.ui.RLHFTabController import RLHFTabController
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkRLHFTabView(BaseRLHFTabView):
    def __init__(self, master, controller: RLHFTabController, ui_state):
        BaseRLHFTabView.__init__(self, ctk_components)

        self.controller = controller
        self.ui_state = ui_state

        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)

        self.build_content(frame, controller, ui_state)
        add_dpo_timestep_controls(
            ctk_components,
            frame,
            99,
            controller,
            ui_state,
        )
        add_mage_linear_dpo_ema_controls(
            ctk_components,
            frame,
            100,
            controller,
            ui_state,
        )
        frame.pack(fill="both", expand=1)
