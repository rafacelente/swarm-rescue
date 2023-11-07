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
#from spg_overlay.gui_map.env_logic import EnvLogic
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.misc_data import MiscData
from maps.map_simple import MyMapSimple
from state_machine import InformedSimpleDrone
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from random import randrange

random_target = (random.randrange(-300, 300), random.randrange(-300, 300))

class MyDronePid(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0

        self.iter_path = 0
        self.path_done = Path()
        self.initial_pos = np.array([295,118])
        self.current_location = np.array([295,118])
        self.orientation = -math.pi/2 #math.pi

        self.targets = {
            'wounded_person': random_target,
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
        self.draw_distance_vector()
        # self.draw_vector()
        # self.draw_direction()
        # self.draw_interesting_points()

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

    def draw_distance_vector(self):
        found_wounded, distance, angle = self.search_targets()
        pt1 = np.array([self.true_position()[0], self.true_position()[1]]) + self._half_size_array
        #print(self.true_angle())
        if found_wounded:
            print(f'distance: {distance}')
            relative_wounded_vector = np.array([distance*np.cos(self.true_angle() - angle), distance*np.sin(self.true_angle() - angle)])
            arcade.draw_line(pt1[0],
                        pt1[1],
                        relative_wounded_vector[0] + pt1[0],
                        relative_wounded_vector[1] + pt1[1],
                        color=(0,0,0))

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


class MyMapRandom(MapAbstract):
    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (900, 900)

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        self._drones_pos = []
        for i in range(self._number_drones):
            pos = ((0, 0), 0)
            self._drones_pos.append(pos)

        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type: Type[DroneAbstract]):
        playground = ClosedPlayground(size=self._size_area)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMapSimple()
    my_map._wounded_persons_pos = [random_target]

    playground = my_map.construct_playground(drone_type=MyDronePid)

    gui = GuiSR(playground=playground,
                the_map=my_map,
                use_keyboard=False,
                use_mouse_measure=True,
                enable_visu_noises=False,
                )

    gui.run()


if __name__ == '__main__':
    main()
