"""
This program can be launched directly.
Example of how to control one drone
"""
import random
import math
import os
import sys
from typing import List, Type, Tuple

import arcade
import numpy as np

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.utils import clamp
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.environments.env_sr_gui import EnvSRGui
from spg_overlay.environments.env_sr import EnvSR

#from spg_overlay.gui_map.env_logic import EnvLogic
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.misc_data import MiscData
from maps.map_simple import MyMapSimple
from maps.map_env import EnvMap
from random import randrange
import gymnasium as gym
from stable_baselines3 import PPO
from drones.RLDrone import RLDrone

        
def main():
    my_map = EnvMap()

    playground = my_map.construct_playground(drone_type=RLDrone)

    env = GuiSR(playground=playground,
                the_map=my_map,
                use_keyboard=False,
                use_mouse_measure=True,
                enable_visu_noises=False,
                )
    
    env.run()

    # obs, _ = env.reset()
    # n_steps = 500
    # eval_episodes = 2
    # total_reward = 0
    # for episode in range(eval_episodes):
    #     obs, _ = env.reset()
    #     episode_reward = 0
    #     for asd in range(n_steps):
    #         #action = 1
    #         action, _ = model.predict(obs, deterministic=True)
    #         print(f"Step {step+1}")
    #         print(f"Action: {action}")
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated
    #         # if asd % 100 == 0 or asd == 600:
    #             #print(f"obs = {obs}, reward = {reward}, done = {done}")
    #         if done:
    #             print(f"Goal reached! Reward = {reward}")
    #             break
    #         episode_reward += reward
    #     print(f'Episode {episode}: Reward = {episode_reward}')
    #     total_reward += reward

    # print(f'Average reward over {eval_episodes} episodes: {total_reward/eval_episodes}')


if __name__ == '__main__':
    main()
