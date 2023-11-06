"""
This program can be launched directly.
Example of how to control one drone
"""
import math
import os
import sys
from typing import Optional, List, Type
import numpy as np

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor

from stable_baselines3 import PPO

fixed_target = (0,0)
class RLDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0
        self.command = {"forward": 0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
        self.just_found_wounded = False
        self.just_grabbed_wounded = False
        self.rescue_center = np.array([295, 205])

        self.model = PPO.load('/home/rafa/Desktop/Projects/swarm-rescue/saved_models/continuous_04112023.zip')
        self.initial_obs = self.state_space()
        #self.inital_action, _ = self.model.predict(self.initial_obs, deterministic=True)
        self.initial_action = np.array([0, 0, 0, 0]).astype(np.float32)

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass
  
    def map_action(self, action):
        self.command = {
                        "forward": action[0],
                        "lateral": action[1],
                        "rotation": action[2],
                        "grasper": np.round(action[3]).astype(int)
        }
        return self.command

    def state_space(self):
        found_wounded, distance_to_wounded, angle_to_wounded = self.search_targets()
        distance_to_rc, angle_to_rc = self.distance_to_rc()

        return [
            *self.position(),
            *self.velocity(),
            self.angular_vel(),
            self.angle(),
            *self.front_view(),
            distance_to_wounded,
            angle_to_wounded,
            distance_to_rc,
            angle_to_rc,
            self.collided(),
            self.has_target(),
            found_wounded
        ]

    # def control_in_training(self, action):
    #     self.counter += 1
    #     action = 0
    #     if self.counter == 1:
    #         action = self.inital_action
    #         return self.map_action(action)
        
    #     return self.map_action(action)

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        self.counter += 1
        action = 0
        if self.counter == 1:
            action = self.initial_action
            return self.map_action(action)
        
        
        obs = self.state_space()
        action, _ = self.model.predict(obs, deterministic=True)
        # self.last_distance = distance
        return self.map_action(action)
    
    def front_view(self, fov: Optional[int]=120):
        
        if self.lidar_values() is None:
            return None

        values = self.lidar_values()
        slices = int(360/fov)
        return values[int((len(values)-1)/slices):int((slices-1)*(len(values)-1)/slices)]
    
    def collided(self):
        if self.lidar_values() is None:
            return 0

        collided = 0
        dist = min(self.lidar_values())

        if dist < 40:
            collided = 1

        return collided
    
    def has_target(self):
        return int(len(self.base.grasper.grasped_entities) == 1)
    
    def search_targets(self):
        found_wounded = 0
        distance = -1
        angle = -1
        grasped = False

        detection_semantic = self.semantic_values()
        if detection_semantic is not None:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    distance = data.distance
                    angle = data.angle
                    grasped = data.grasped
        return found_wounded, distance, angle #, grasped
    
    def accurate_position(self):
        return self.true_position()
    
    def accurate_velocity(self):
        return self.true_velocity()
    
    # This may be changed to measured_gps
    def velocity(self):
        return self.true_velocity()
    
    # This may be changed to measured_gps
    def position(self):
        return self.accurate_position()
    
    # This may be changed to measured_compass_angle
    def angle(self):
        return self.true_angle()

    def angular_vel(self):
        return self.true_angular_velocity()
    
    def distance_to_rc(self):
        dist_vector = np.array(self.rescue_center) - np.array(self.position())
        distance = np.linalg.norm(dist_vector)
        angle = np.arctan2(dist_vector[1], dist_vector[0])
        return distance, angle
        