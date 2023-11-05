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

random_target = (random.randrange(-300, 300), random.randrange(-300, 300))
fixed_target = (0,0)
class RLDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0
        self.command = {"forward": 0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
        self.initial_pos = np.array([295,118])
        self.current_location = np.array([295,118])
        self.orientation = math.pi #math.pi
        self.last_distance = 0

        self.targets = {
            'wounded_person': np.array([0,0]),
            'rescue_center': np.array([295, 205]) 
        }
        self.assigned_target = 'wounded_person'
        self.target_location = self.targets[self.assigned_target]

        self.model = PPO.load('/home/rafa/Desktop/Projects/swarm-rescue/saved_models/saved_models.zip')
        self.initial_obs = [
            self.initial_pos[0],
            self.initial_pos[1],
            np.linalg.norm(np.array(self.target_location - self.initial_pos)),
            self.last_distance
        ]
        self.inital_action, _ = self.model.predict(self.initial_obs, deterministic=True)


    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass
  
    def map_action(self, action, action_type='cont'):
        if action_type == 'discrete':
            if action == 0:
                self.command["forward"] = 0
                self.command["rotation"] = 0
            if action == 1:
                self.command["forward"] = 1
                self.command["rotation"] = 0
            if action == 2:
                self.command["forward"] = 0
                self.command["rotation"] = 1
            if action == 3:
                self.command["forward"] = 1
                self.command["rotation"] = 1
        else:
            self.command["forward"] = action[0]
            self.command["rotation"] = action[1]
        return self.command

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        self.counter += 1
        action = 0
        if self.counter == 1:
            action = self.inital_action
            return self.map_action(action)
        
        pos = np.array(self.true_position())
        distance = np.linalg.norm(np.array(self.target_location - pos))
        obs = [
            self.true_position()[0],
            self.true_position()[1],
            distance,
            self.last_distance

        ]
        self.last_distance = distance
        action, _ = self.model.predict(obs, deterministic=True)
        print(action)
        return self.map_action(action)
        

def main():
    my_map = EnvMap()

    playground = my_map.construct_playground(drone_type=RLDrone)

    env = EnvSR(playground=playground,
                the_map=my_map,
                )

    obs, _ = env.reset()
    n_steps = 500
    eval_episodes = 2
    total_reward = 0
    for episode in range(eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for asd in range(n_steps):
            #action = 1
            action, _ = model.predict(obs, deterministic=True)
            print(f"Step {step+1}")
            print(f"Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            # if asd % 100 == 0 or asd == 600:
                #print(f"obs = {obs}, reward = {reward}, done = {done}")
            if done:
                print(f"Goal reached! Reward = {reward}")
                break
            episode_reward += reward
        print(f'Episode {episode}: Reward = {episode_reward}')
        total_reward += reward

    print(f'Average reward over {eval_episodes} episodes: {total_reward/eval_episodes}')


if __name__ == '__main__':
    main()
