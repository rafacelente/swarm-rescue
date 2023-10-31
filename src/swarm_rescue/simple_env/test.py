"""
This program can be launched directly.
Example of how to control one drone
"""
import random
import math
import os
import sys
from typing import Dict, Optional, Tuple, Union

import arcade
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.utils import clamp
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.misc_data import MiscData
from maps.map_simple import MyMapSimple
from random import randrange

from spg.playground import Playground


class SimpleDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self, 
            render_mode : Optional[str] =None, 
            playground: Optional[Playground] = None
        ) -> None:

        
        self._playground = playground
        self._screen_size = playground.size()
        self._renderer = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_mode is not None:
            # IMPLEMENT RENDERED
            pass

        def _get_observation(self):



def main():
    my_map = MyMapSimple()
    my_map._wounded_persons_pos = [random_target]

    playground = my_map.construct_playground(drone_type=MyDronePid)

    gui = GuiSR(playground=playground,
                the_map=my_map,
                use_keyboard=False,
                use_mouse_measure=True,
                enable_visu_noises=False,
                )

    gui.run()


if __name__ == '__main__':
    main()
