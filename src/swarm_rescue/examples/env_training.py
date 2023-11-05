"""
This program can be launched directly.
Example of how to control one drone
"""
import random
import math
import os
import sys
from typing import List, Type, Tuple
import gymnasium as gym

import arcade
import numpy as np

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.utils import clamp
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.closed_playground import ClosedPlayground
#from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.environments.env_sr import EnvSR
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.misc_data import MiscData
#from maps.map_simple import MyMapSimple
from maps.map_env import EnvMap
from random import randrange

#import pyvirtualdisplay

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from drones.RLDrone import RLDrone
        

def main():
    
    #pyvirtualdisplay.Display(visible=0, size=(1024, 768)).start()
    
    my_map = EnvMap()
    my_map.reset_map()
    
    playground = my_map.construct_playground(drone_type=RLDrone)

    env = EnvSR(playground=playground,
               the_map=my_map,
    )
    

    vec_env = make_vec_env(EnvSR, n_envs=1, env_kwargs=dict(playground=playground,
               the_map=my_map,))
    
    model = PPO("MlpPolicy", vec_env, gamma=0.999, n_epochs=32, gae_lambda=0.95, ent_coef=0.01, batch_size=256, verbose=1, device='cuda').learn(total_timesteps=2000000)
    #model = PPO("MlpPolicy", env, verbose=1).learn(5)

    save_path = '/notebooks/saved_models/continuous_04112023.zip'
    model.save(save_path)
    
    
    print('\n\nStarting evaluation\n\n')
    obs, _ = env.reset()
    n_steps = 500
    eval_episodes = 10
    total_reward = 0

    for episode in range(eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for asd in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            if asd % 100 == 0 or asd == 600:
                print(f"obs = {obs}, reward = {reward}, done = {done}")
            if done:
                print(f"Goal reached! Reward = {reward}")
                break
            episode_reward += reward
        print(f'Episode {episode}: Reward = {episode_reward}')
        total_reward += reward

    print(f'Average reward over {eval_episodes} episodes: {total_reward/eval_episodes}')

if __name__ == '__main__':
    main()
