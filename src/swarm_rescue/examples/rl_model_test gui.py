"""
This program can be launched directly.
Example of how to control one drone
"""
import os
import sys

import numpy as np

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.gui_map.gui_sr import GuiSR

#from spg_overlay.gui_map.env_logic import EnvLogic
from maps.map_env import EnvMap
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
