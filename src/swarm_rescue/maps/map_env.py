import math
import random
from typing import List, Optional, Tuple, Type

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.sensor_disablers import ZoneType, NoGpsZone, srdisabler_disables_device
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData

from .walls_intermediate_map_1 import add_walls, add_boxes
import numpy as np


class EnvMap(MapAbstract):

    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._time_step_limit = 500
        self._real_time_limit = 100

        # PARAMETERS MAP
        self._size_area = (800, 500)
        # 400
        # 150

        self._rescue_center = RescueCenter(size=(200, 80))
        self._rescue_center_pos = ((295, 205), 0)

        # self._no_gps_zone = NoGpsZone(size=(400, 500))
        # self._no_gps_zone_pos = ((-190, 0), 0)

        self._wounded_persons_pos = [(0,0)]
        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        orient = math.pi#math.pi
        self._drones_pos = [((100, 100), orient)]
        self._number_drones = len(self._drones_pos)
        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)

        playground.add(self._rescue_center, self._rescue_center_pos)

        #add_walls(playground)
        add_boxes(playground)

        self._explored_map.initialize_walls(playground)

        # DISABLER ZONES
        # playground.add_interaction(CollisionTypes.DISABLER,
        #                            CollisionTypes.DEVICE,
        #                            srdisabler_disables_device)

        # if ZoneType.NO_GPS_ZONE in self._zones_config:
        #     playground.add(self._no_gps_zone, self._no_gps_zone_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground

    def reset_map(self, reset_type: Optional[int] = 1, random_range: Optional[Tuple[int, int]] = None):
        # Reset Type 0 : fixed
        loc = self._wounded_persons_pos
        # Reset type 1: random
        if reset_type == 1:
            if random_range is None:
                max_x = np.array(self._size_area)[0]/2 - 50
                max_y = np.array(self._size_area)[1]/2 - 50
            
            loc = (random.randrange(-max_x, max_x), random.randrange(-max_y, max_y))
        for i in range(self._number_wounded_persons):
            self._wounded_persons_pos[i] = loc
