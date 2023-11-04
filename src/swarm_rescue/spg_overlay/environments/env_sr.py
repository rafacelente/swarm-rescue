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
import random


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
            render_mode: Optional[str] ="rgb",
            reset_type : Optional[int] = 1,
    ) -> None:
        super(EnvSR, self).__init__()

        # Playground definition
        self._size = size
        self._playground = playground
        self._playground.window.set_size(*self._playground.size)
        self.last_distance = None

        # Map and drones definition
        self._the_map = the_map
        self._drones = self._the_map.drones
        self._number_drones = self._the_map.number_drones
        
        # Maybe change later so it doesn't know the objective state
        if objective is None:
            self.objective = (0,0)
        else:
            self.objective = self._the_map._wounded_persons_pos

        # Time constraints
        self._real_time_limit = self._the_map.real_time_limit
        if self._real_time_limit is None:
            self._real_time_limit = 100

        self._time_step_limit = self._the_map.time_step_limit
        if self._time_step_limit is None:
            self._time_step_limit = 100


        # Drone commands
        self._drones_commands: Union[Dict[DroneAbstract, Dict[Union[str, Controller], Command]], Type[None]] = None
        if self._drones:
            self._drones_commands = {}

        self.command = {"forward": 0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}


        # Playground step definitions
        self._playground.window.on_update = self.step
        self._playground.window.set_update_rate(FRAME_RATE)


        # Parameter initializations
        self._total_number_wounded_persons = self._the_map.number_wounded_persons
        self._elapsed_time = 0
        self._start_real_time = time.time()
        self._real_time_limit_reached = False
        self._real_time_elapsed = 0
        self._terminate = False
        self._truncate = False
        self._reset_type = reset_type

        # Action space definition (For now, forward and rotate)
        # Forward, rotate / yes or no
            # n_actions = 4
            # self.action_space = spaces.Discrete(n_actions) 
        
        # New action space: continuous
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        
        # State space definition
        max_x = np.array(self._the_map._size_area)[0]/2
        max_y = np.array(self._the_map._size_area)[1]/2
        min_dist_to_target = 0
        max_dist_to_target = np.linalg.norm(np.array(self._the_map._size_area))
        low = np.array(
            [
                -max_x,
                -max_y,
                min_dist_to_target,
                min_dist_to_target, # Added last distance
            ]
        ).astype(np.float32)
        high = np.array(
            [
                max_x,
                max_y,
                max_dist_to_target,
                max_dist_to_target
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        
        # I don't know what this is
        self.render_mode = render_mode

    # Doesn't check for overlapping...
    def reset_drone(self, reset_type: Optional[int] = 1, random_range: Optional[Tuple[int, int]] = None):
        # Reset Type 0 : fixed
        loc = (100,100)
        angle = np.pi
        # Reset type 1: random
        if reset_type == 1:
            if random_range is None:
                max_x = np.array(self._the_map._size_area)[0]/2 - 50
                max_y = np.array(self._the_map._size_area)[1]/2 - 50
            
            loc = (random.randrange(-max_x, max_x), random.randrange(-max_y, max_y))
            while np.linalg.norm(np.array(loc) - np.array(self.objective)) < 150:
                loc = (random.randrange(-max_x, max_x), random.randrange(-max_y, max_y))
            angle = random.uniform(-np.pi, np.pi)
        
        for drone in self._drones:
            drone.initial_coordinates = (loc, angle)
            drone.reset()

    def reset(self, seed: Optional[int]=None):
        #print('reseting environment...')
        super().reset(seed=seed)
        self._real_time_elapsed = 0
        self._elapsed_time = 0
        self._start_real_time = time.time()
        self.last_distance = None
        self._terminate = False
        self._truncate = False

        self._the_map.reset_map(reset_type=self._reset_type)
        self.reset_drone(reset_type=self._reset_type)
        self._playground.reset()
        self.objective = self._the_map._wounded_persons_pos[0]

        #print(f'Drone loc: {self._drones[0].true_position()}')
        #print(f'Target loc: {self._the_map._wounded_persons_pos[0]}')

        # return self.step(0)[0], {} DISCRETE
        return self.step(np.zeros(shape=(self.observation_space.shape)))[0], {}

    def step(self, action):
        self._elapsed_time += 1
        
        self._the_map.explored_map.update_drones(self._drones)

        # COMPUTE COMMANDS
        # for i in range(self._number_drones):
        #     #command = self._drones[i].control()
        #     if action == 0:
        #         self.command["forward"] = 0
        #         self.command["rotation"] = 0
        #     if action == 1:
        #         self.command["forward"] = 1
        #         self.command["rotation"] = 0
        #     if action == 2:
        #         self.command["forward"] = 0
        #         self.command["rotation"] = 1
        #     if action == 3:
        #         self.command["forward"] = 1
        #         self.command["rotation"] = 1
        for i in range(self._number_drones):
            self.command["forward"] = action[0]
            self.command["rotation"] = action[1]
            self._drones_commands[self._drones[i]] = self.command


        self._playground.step(commands=self._drones_commands)

        # REWARDS
        reward = 0
        
        drone_pos = self._drones[0].true_position()
        #drone_vel = self._drones[0].true_velocity()
        
        distance_to_target = np.linalg.norm(np.array(self.objective - drone_pos))
        if self.last_distance is not None:
            how_closer = self.last_distance - distance_to_target
        else:
                how_closer = 0
        self.last_distance = distance_to_target
        end_real_time = time.time()
        last_real_time_elapsed = self._real_time_elapsed
        self._real_time_elapsed = (end_real_time - self._start_real_time)

        reward = 1/(distance_to_target + 0.4) + 0.05*how_closer #*0.05- 0.0003*np.absolute(drone_pos[0]) - 0.0003*np.absolute(drone_pos[1]) + how_closer*0.05

        if self._elapsed_time > self._time_step_limit:
            self._elapsed_time = self._time_step_limit
            reward -= 50
            self._truncate = True

        if self._real_time_elapsed > self._real_time_limit:
            self._real_time_elapsed = self._real_time_limit
            self._truncate = True

        epsilon = 40
        if (np.linalg.norm(np.array(drone_pos) - np.array(self.objective)) < epsilon):
            reward += 1000
            print('Goal reached')
            self._terminate = True
        
        state = [
            drone_pos[0],
            drone_pos[1],
            distance_to_target,
            self.last_distance
        ]
        assert len(state) == 4
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
