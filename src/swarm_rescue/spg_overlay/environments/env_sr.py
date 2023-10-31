import arcade
import time
from typing import Optional, Tuple, List, Dict, Union, Type
import cv2
import numpy as np

from spg.agent.controller.controller import Command, Controller
from spg.playground import Playground
from spg.playground.playground import SentMessagesDict
import gymnasium as gym
from gymnasium import spaces

from spg_overlay.utils.constants import FRAME_RATE, DRONE_INITIAL_HEALTH
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.map_abstract import MapAbstract


class EnvSR(gym.Env):
    """
    The EnvSR class is a subclass of TopDownView and provides a graphical user interface for the simulation. It handles
    the rendering of the playground, drones, and other visual elements, as well as user input and interaction.
    """

    def __init__(
            self,
            the_map: MapAbstract,
            playground: Playground,
            objective : Optional[Tuple[int, int]] = None,
            size: Optional[Tuple[int, int]] = None,
            render_mode="rgb"
    ) -> None:
        super(EnvSR, self).__init__()

        self._size = size
        self._playground = playground
        self._playground.window.set_size(*self._playground.size)
        #self._playground.window.set_visible(True)

        self._the_map = the_map
        self._drones = self._the_map.drones
        self._number_drones = self._the_map.number_drones
        self.objective = objective

        self._real_time_limit = self._the_map.real_time_limit
        if self._real_time_limit is None:
            self._real_time_limit = 100

        self._time_step_limit = self._the_map.time_step_limit
        if self._time_step_limit is None:
            self._time_step_limit = 100

        self._drones_commands: Union[Dict[DroneAbstract, Dict[Union[str, Controller], Command]], Type[None]] = None
        if self._drones:
            self._drones_commands = {}


        # self._playground.window.on_draw = self.on_draw
        self._playground.window.on_update = self.step
        self._playground.window.set_update_rate(FRAME_RATE)


        self._total_number_wounded_persons = self._the_map.number_wounded_persons
        self._elapsed_time = 0
        self._start_real_time = time.time()
        self._real_time_limit_reached = False
        self._real_time_elapsed = 0

        self._last_image = None
        self._terminate = False
        self._truncate = False
        #self.game_over = False

        self.command = {"forward": 0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}

        # Forward, rotate / yes or no
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        
    
        min_dist = 0
        max_dist = np.linalg.norm(np.array(self._the_map._size_area))
        min_dist_to_target = 0
        max_dist_to_target = np.linalg.norm(np.array(self._the_map._size_area))
        min_t = 0
        max_t = self._time_step_limit
        low = np.array(
            [
                min_dist,
                min_t,
                min_dist_to_target
            ]
        ).astype(np.float32)
        high = np.array(
            [
                max_dist,
                max_t,
                max_dist_to_target
            ]
        ).astype(np.float32)


        self.observation_space = spaces.Box(low, high)
        self.render_mode = render_mode

    def reset(self, seed: Optional[int]=None):
        #print('reseting environment...')
        super().reset(seed=seed)
        self._real_time_elapsed = 0
        self._start_real_time = time.time()
        self._elapsed_time = 0
        self._playground.reset()
        self._terminate = False
        self._truncate = False
        #reset_step, _, _, _, _= self.step(0)
        #print(f"reset_step: {reset_step}")
        return self.step(0)[0], {}
        #return self.step(0), {}

    def step(self, action):
        #print(f'action: {action}')
        self._elapsed_time += 1

        # if self._elapsed_time < 2:
        #     self._playground.step(commands=self._drones_commands, messages=self._messages)
        #     # self._the_map.explored_map.update(self._drones)
        #     # self._the_map.explored_map._process_positions()
        #     # self._the_map.explored_map.display()
        #     return

        self._the_map.explored_map.update_drones(self._drones)

        # COMPUTE COMMANDS
        for i in range(self._number_drones):
            #command = self._drones[i].control()
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
            
            self._drones_commands[self._drones[i]] = self.command

        if self._drones:
            self._drones[0].display()

        self._playground.step(commands=self._drones_commands)

        # self._the_map.explored_map.display()

        # REWARDS
        reward = 0
        
        drone_pos = self._drones[0].true_position()
        #print(drone_pos)
        distance = np.linalg.norm(np.array(drone_pos))

        distance_to_target = np.linalg.norm(np.array(self.objective - drone_pos))
        end_real_time = time.time()
        last_real_time_elapsed = self._real_time_elapsed
        self._real_time_elapsed = (end_real_time - self._start_real_time)
        delta = self._real_time_elapsed - last_real_time_elapsed

        reward = -0.01*distance_to_target

        # print(self._elapsed_time)
        # print(self._real_time_elapsed)
        if self._elapsed_time > self._time_step_limit:
            self._elapsed_time = self._time_step_limit
            reward -= 5000
            self._truncate = True

        if self._real_time_elapsed > self._real_time_limit:
            self._real_time_elapsed = self._real_time_limit
            self._truncate = True

        epsilon = 50
        if (drone_pos[0] <= self.objective[0] + epsilon and drone_pos[0] >= self.objective[0] - epsilon) and (drone_pos[1] <= self.objective[1] + epsilon and drone_pos[1] >= self.objective[1] - epsilon):
            reward += 1000
            print('Goal reached')
            self._terminate = True
        

        #print(f'reward: {reward}')

        self._messages = {}

        state = [
            distance,
            self._elapsed_time,
            distance_to_target
        ]
        assert len(state) == 3
        #print([np.array(state, dtype=np.float32), reward, self._terminate, self._truncate, {}])
        return np.array(state, dtype=np.float32), reward, self._terminate, self._truncate, {}


    @property
    def elapsed_time(self):
        return self._elapsed_time

    @property
    def real_time_elapsed(self):
        return self._real_time_elapsed

    @property
    def real_time_limit_reached(self):
        return self._real_time_limit_reached
