


import arcade
import math
import random
from typing import Optional

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle
import numpy as np

class MyDroneEval(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = True
        self.initial_pos = np.array([295, 118])
        self.orientation = math.pi
        self.goal_location = np.array([-310, -180])


    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided
    
    def draw_direction(self):
        pt1 = np.array([self.true_position()[0], self.true_position()[1]])
        pt1 = pt1 + self._half_size_array
        pt2 = pt1 + 250 * np.array([math.cos(self.true_angle()), math.sin(self.true_angle())])
        color = (255, 64, 0)
        arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color)

    def draw_vector(self):
        dist_vector = self.initial_pos + self.goal_location
        arcade.draw(self.initial_pos[0],
                    self.initial_pos[1],
                    self.goal_location[0],
                    self.goal_location[1])

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        dist_vector = self.initial_pos + self.goal_location
        theta = np.arctan2(dist_vector[1], dist_vector[0])

        

        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        command_turn = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}

        collided = self.process_lidar_sensor()

        self.counterStraight += 1

        if collided and not self.isTurning and self.counterStraight > self.distStopStraight:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        measured_angle = 0
        if self.measured_compass_angle() is not None:
            measured_angle = self.measured_compass_angle()
        
        diff_angle = normalize_angle(theta - measured_angle)
        print(f'diff_angle: {diff_angle}')

        if abs(diff_angle) > 0.2:
            self.isTurning = True
        else:
            self.isTurning = False

        print(f'policy self.isTurning: {self.isTurning}')

        if self.isTurning:
            return command_turn
        else:
            return command_straight

