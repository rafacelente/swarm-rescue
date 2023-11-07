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
        self.state = 'search'

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
        low=np.array([-1, -1, -1]).astype(np.float32)
        high=np.array([1, 1, 1]).astype(np.float32)
        self.continuous_action_space = spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32)
        self.binary_action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.concatenate((self.continuous_action_space.low, self.binary_action_space.low)),
            high=np.concatenate((self.continuous_action_space.high, self.binary_action_space.high)),
            dtype=np.float32
        )
        
        # State space definition

        # 1) Binary variables
        num_binary_vars = 3
        binary_space = spaces.MultiBinary(num_binary_vars) # collided, has_target, wounded_nearby

        # 2) Continuous variables
        max_x = np.array(self._the_map._size_area)[0]/2
        max_y = np.array(self._the_map._size_area)[1]/2
        max_v = 5.5 # Max velocity (vx, vy)
        max_w = 0.17 # Max angular velocity
        max_angle = np.pi
        min_view_distance = 0   # Minimum Lidar distance
        max_view_distance = 310 # Maximum Lidar distance
        min_semantic_distance = 0 # Minimum Semantic Sensor distance
        max_semantic_distance = 210 # Maximum Semantic Sensor distance
        min_distance_to_rc = 0
        max_distance_to_rc = max(np.array(self._the_map._size_area))
        
        fov = 120
        n_beams = 60 # Change this when fov changes
        min_view_distance_list = [min_view_distance for i in range(n_beams)]
        max_view_distance_list = [max_view_distance for i in range(n_beams)]

        low = np.array(
            [
                -max_x, # x
                -max_y, # y
                -max_v, # vx
                -max_v, # vy
                -max_w,
                -max_angle, # drone angle
                *min_view_distance_list, # Lidar values (fov)
                min_semantic_distance, # Semantic sensor distance
                -max_angle, # Semantic sensor relative angle
                min_distance_to_rc, # Distance to Rescue Center
                -max_angle # Angle to rescue center
            ]
        ).astype(np.float32)
        high = np.array(
            [
                max_x, # x
                max_y, # y
                max_v, # vx
                max_v, # vy
                max_w,
                max_angle, # drone angle
                *max_view_distance_list, # Lidar values (fov)
                max_semantic_distance, # Semantic sensor distance
                max_angle, # Semantic sensor relative angle
                max_distance_to_rc, # Distance to Rescue Center
                max_angle # Angle to rescue center
            ]
        ).astype(np.float32)
        continuous = spaces.Box(low=low, high=high)

        total_space_length = num_binary_vars + np.prod(continuous.shape)
        low = np.concatenate(([0] * num_binary_vars, continuous.low))
        high = np.concatenate(([1] * num_binary_vars, continuous.high))

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

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
            drone.just_grabbed_wounded = False
            drone.just_found_wounded = False
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

        return self.step(np.zeros(shape=(self.action_space.shape)))[0], {}

    def step(self, action):
        self._elapsed_time += 1
        
        self._the_map.explored_map.update_drones(self._drones)

        for i in range(self._number_drones):
            self.command = self._drones[i].map_action(action)
            self._drones_commands[self._drones[i]] = self.command


        self._playground.step(commands=self._drones_commands)

        # REWARDS
        reward = 0
        
        state_space = self._drones[0].state_space()
        
        drone_pos = state_space[0:2]
        drone_vel = state_space[2:4]
        drone_w = state_space[4]
        drone_angle = state_space[5]
        direction_vector = state_space[6:8]
        fov_view = state_space[8:68]
        distance_to_wounded = state_space[69]
        angle_to_wounded = state_space[70]
        distance_to_rc = state_space[71]
        angle_to_rc = state_space[72]
        collided = state_space[73]
        has_target = state_space[74]
        found_wounded = state_space[75]

        time_step_penalty = -0.015
        reward += time_step_penalty


        # Maybe change this to a state machine
        if not has_target:
            if not found_wounded:
                self.state = 'search'
                move_reward = drone_vel[0]*0.008 + drone_vel[1]*0.008
                collision_penalty = - 0.05*collided
                reward += (move_reward + collision_penalty)
                # print(f'move reward: {move_reward}')
                # print(f'collision penalty: {collision_penalty}')
            else:
                if self._drones[0].just_found_wounded:
                    self._drones[0].just_found_wounded = True
                    reward += 20
                relative_wounded_vector = np.array([distance_to_wounded*np.cos(drone_angle - angle_to_wounded), distance_to_wounded*np.sin(drone_angle - angle_to_wounded)])
                alignment_dot_product = np.dot(np.array(direction_vector), relative_wounded_vector)/(np.linalg.norm(direction_vector)*np.linalg.norm(relative_wounded_vector))
                approach_reward = 5/(np.exp((distance_to_wounded - 40)/10) + 1)
                
                reward += (0.2*alignment_dot_product + approach_reward + 0.01)
                self.state = 'approach'
        else:
            reward += 0.03
            self.state = 'return'
            if not self._drones[0].just_grabbed_wounded:
                self._drones[0].just_grabbed_wounded = True
                reward += 200
            
            relative_rc_vector = np.array([distance_to_wounded*np.cos(drone_angle - angle_to_rc), distance_to_wounded*np.sin(drone_angle - angle_to_rc)])
            alignment_dot_product = np.dot(np.array(direction_vector), relative_rc_vector)/(np.linalg.norm(direction_vector)*np.linalg.norm(relative_rc_vector))
            approach_reward = 10/(np.exp((distance_to_rc - 40)/10) + 1)

            reward += (0.2*alignment_dot_product + 2*approach_reward + 0.02)

            if distance_to_rc < 40:
                reward += 600
                print('Goal Reached')
                self._terminate = True


        # print(self.state)
        # print(reward)
        end_real_time = time.time()
        last_real_time_elapsed = self._real_time_elapsed
        self._real_time_elapsed = (end_real_time - self._start_real_time)

        if self._elapsed_time > self._time_step_limit:
            self._elapsed_time = self._time_step_limit
            reward -= 50
            self._truncate = True

        if self._real_time_elapsed > self._real_time_limit:
            self._real_time_elapsed = self._real_time_limit
            self._truncate = True

         
        return state_space, reward, self._terminate, self._truncate, {}


    @property
    def elapsed_time(self):
        return self._elapsed_time

    @property
    def real_time_elapsed(self):
        return self._real_time_elapsed

    @property
    def real_time_limit_reached(self):
        return self._real_time_limit_reached
