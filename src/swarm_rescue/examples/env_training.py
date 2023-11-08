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
import tensorflow as tf
import datetime

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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from drones.RLDrone import RLDrone


from tensorboardX import SummaryWriter

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.n_calls = 0
        self.episodes = 0
        self.average_reward = {
            'general': {
                'time_step_penalty': 0
            },
            'search': {
                'move_reward': 0,
                'collision_penalty': 0
            },
            'approach': {
                'bias': 0,
                'just_found_wounded': 0,
                #'alignment_dot_product': 0,
                'approach_reward': 0
            },
            'return': {
                'bias': 0,
                'just_grabbed_wounded': 0,
                #'alignment_dot_product': 0,
                'approach_reward': 0,
                'goal_reached': 0
            }
        }
    def _on_step(self) -> bool:
        self.n_calls += 1
        self.episodes += 1
        env = self.training_env.envs[0]
        reward_dict = env.reward_dict
        for state, rewards in reward_dict.items():
            for reward_name, reward_value in rewards.items():
                self.average_reward[state][reward_name] += reward_value
        return True

    def _on_rollout_end(self) -> None:
        env = self.training_env.envs[0]
        reward_dict = env.reward_dict

        # Here you log each component of the reward to TensorBoard
        for state, rewards in reward_dict.items():
            for reward_name, reward_value in rewards.items():
                self.writer.add_scalar(f'rewards/{state}/{reward_name}', reward_value/self.episodes, self.n_calls)
                self.average_reward[state][reward_name] = 0
        self.episodes = 0
        self.writer.flush()  # Make sure all pending events have been written to disk


def main():
    
    #pyvirtualdisplay.Display(visible=0, size=(1024, 768)).start()
    
    my_map = EnvMap()
    playground = my_map.construct_playground(drone_type=RLDrone)

    env = EnvSR(playground=playground,
               the_map=my_map,
    )
    env.reset()
    

    vec_env = make_vec_env(EnvSR, n_envs=1, env_kwargs=dict(playground=playground,
               the_map=my_map,))
    
    # Logging (tensorboard)
    logs_dir = './tensorboard/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

    model = PPO("MlpPolicy", 
                vec_env, gamma=0.9995,
                n_epochs=16,
                gae_lambda=0.95,
                ent_coef=0.001,
                batch_size=1024,
                verbose=1,
                device='cuda',           
                tensorboard_log=logs_dir,
                )

    # Logging and saving
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path='./saved_models',
        name_prefix='ss_197')
    reward_log_callback = RewardLoggerCallback(log_dir=f'{logs_dir}/rewards_log')

    save_path = './saved_models/ss_197_08112023.zip'
    
    
    # Training    
    model.learn(total_timesteps=4000000, 
                tb_log_name='ss_197', 
                reset_num_timesteps=False,
                callback=[checkpoint_callback, reward_log_callback])
    model.save(save_path)
    
    
    # print('\n\nStarting evaluation\n\n')
    # obs, _ = env.reset()
    # n_steps = 500
    # eval_episodes = 10
    # total_reward = 0

    # for episode in range(eval_episodes):
    #     obs, _ = env.reset()
    #     episode_reward = 0
    #     for asd in range(n_steps):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated
    #         if asd % 100 == 0 or asd == 600:
    #             print(f"obs = {obs}, reward = {reward}, done = {done}")
    #         if done:
    #             print(f"Goal reached! Reward = {reward}")
    #             break
    #         episode_reward += reward
    #     print(f'Episode {episode}: Reward = {episode_reward}')
    #     total_reward += reward

    # print(f'Average reward over {eval_episodes} episodes: {total_reward/eval_episodes}')

if __name__ == '__main__':
    main()
