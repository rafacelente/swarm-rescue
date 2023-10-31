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
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.misc_data import MiscData
#from maps.map_simple import MyMapSimple
from state_machine import InformedSimpleDrone
from maps.map_env import EnvMap
from random import randrange

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

random_target = (random.randrange(-300, 300), random.randrange(-300, 300))
fixed_target = (0,0)
class MyDroneEnv(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0

        self.iter_path = 0
        self.path_done = Path()
        self.initial_pos = np.array([295,118])
        self.current_location = np.array([295,118])
        self.orientation = math.pi

        self.targets = {
            'wounded_person': fixed_target,
            'rescue_center': np.array([295, 205]) 
        }
        self.assigned_target = 'wounded_person'
        self.target_location = self.targets[self.assigned_target]

        self.alignment_threshold = 0.1
        self.diff_angle = self.get_rotate_angle()
        self.state_machine = InformedSimpleDrone(self)
        self.close_enough_threshold = 50
        
    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def get_rotate_angle(self):
        dist_vector = self.target_location - self.current_location
        theta = np.arctan2(dist_vector[1], dist_vector[0])
        return theta

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """

        self.counter += 1
        print(self.counter)


        # Updating information
        self.current_location = self.true_position()

        # Updating target information
        self.target_location = self.targets[self.assigned_target]

        #print(self.grasped_entities())
        #print(f'target: {self.assigned_target}')
        #print(f'target location: {self.target_location}')

        theta = self.get_rotate_angle()
        measured_angle = 0
        if self.measured_compass_angle() is not None:
            measured_angle = self.measured_compass_angle()
        
        self.diff_angle = normalize_angle(theta - measured_angle)

        # State machine step
        self.state_machine.send("cycle")
        return self.state_machine.get_command()

    def draw_bottom_layer(self):
        self.draw_vector()
        self.draw_direction()
        self.draw_interesting_points()

    def draw_path(self, path: Path(), color: Tuple[int, int, int]):
        length = path.length()
        # print(length)
        pt2 = None
        for ind_pt in range(length):
            pose = path.get(ind_pt)
            pt1 = pose.position + self._half_size_array
            # print(ind_pt, pt1, pt2)
            if ind_pt > 0:
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color)
            pt2 = pt1

    def draw_vector(self):     
        dist_vector = self.target_location - self.initial_pos 
        arcade.draw_line(self.initial_pos[0] + self._half_size_array[0],
                        self.initial_pos[1] + self._half_size_array[1],
                        dist_vector[0],
                        dist_vector[1],
                        color=(0,0,0))

    def draw_interesting_points(self):
        radius = 15
        theo_drone = np.array([295, 118]) + self._half_size_array
        theo_goal = np.array([-310, -180]) + self._half_size_array
        arcade.draw_circle_filled(theo_drone[0], theo_drone[1], radius, (0,0,0)) # Initial pos
        arcade.draw_circle_outline(theo_goal[0], theo_goal[1], radius, (0,255,0))
    
    def draw_direction(self):
        pt1 = np.array([self.true_position()[0], self.true_position()[1]])
        pt1 = pt1 + self._half_size_array
        pt2 = pt1 + 250 * np.array([math.cos(self.true_angle()), math.sin(self.true_angle())])
        color = (255, 64, 0)
        arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color)



def main():
    my_map = EnvMap()
    my_map._wounded_persons_pos = [fixed_target]

    playground = my_map.construct_playground(drone_type=MyDroneEnv)

    env = EnvSR(playground=playground,
               the_map=my_map,
               objective=fixed_target
    )
    
    # TEST RUN
    # obs, _ = env.reset()

    # n_steps = 20
    # FORWARD = 1
    # for step in range(n_steps):
    #     print(f"Step {step+1}")
    #     obs, reward, terminated, truncated, info = env.step(FORWARD)
    #     done = terminated or truncated
    #     print(f"obs = {obs}, reward = {reward}, done = {done}")
    #     if done:
    #         print(f"Goal reached! Reward = {reward}")
    #         break
    # vec_env = make_vec_env(EnvSR, n_envs=1, env_kwargs=dict(playground=playground,
    #            the_map=my_map,
    #            objective=fixed_target))
    
    model = PPO("MlpPolicy", env, gamma=0.999, gae_lambda=0.85, ent_coef=0.05, verbose=1).learn(total_timesteps=100000)
    #model = PPO("MlpPolicy", env, verbose=1).learn(5)

    print('\n\nStarting evaluation\n\n')
    obs, _ = env.reset()
    n_steps = 500
    eval_episodes = 1
    total_reward = 0

    for episode in range(eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for asd in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            #print(f"Step {step+1}")
            #print(f"Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            if asd % 100 == 0 or asd == 500:
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
