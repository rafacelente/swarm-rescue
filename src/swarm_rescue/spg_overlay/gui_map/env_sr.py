import arcade
import time
from typing import Optional, Tuple, List, Dict, Union, Type
import cv2

from spg.agent.controller.controller import Command, Controller
from spg.playground import Playground
from spg.playground.playground import SentMessagesDict
from spg.view import TopDownView

from spg_overlay.utils.constants import FRAME_RATE, DRONE_INITIAL_HEALTH
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.keyboard_controller import KeyboardController
from spg_overlay.utils.fps_display import FpsDisplay
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.mouse_measure import MouseMeasure
from spg_overlay.reporting.screen_recorder import ScreenRecorder
from spg_overlay.utils.visu_noises import VisuNoises


class EnvSR(TopDownView):
    """
    The GuiSR class is a subclass of TopDownView and provides a graphical user interface for the simulation. It handles
    the rendering of the playground, drones, and other visual elements, as well as user input and interaction.
    """

    def __init__(
            self,
            the_map: MapAbstract,
            playground: Playground,
            size: Optional[Tuple[int, int]] = None,
            print_rewards: bool = False,
            print_messages: bool = False,
    ) -> None:
        super().__init__(
            playground,
            size,
            (0,0),
            1,
            False,
            False,
            False,
            False,
        )

        self._size = size
        self._playground = playground
        self._playground.window.set_size(*self._playground.size)
        #self._playground.window.set_visible(True)

        self._the_map = the_map
        self._drones = self._the_map.drones
        self._number_drones = self._the_map.number_drones
        self._print_rewards = print_rewards
        self._print_messages = print_messages

        self._real_time_limit = self._the_map.real_time_limit
        if self._real_time_limit is None:
            self._real_time_limit = 100000000

        self._time_step_limit = self._the_map.time_step_limit
        if self._time_step_limit is None:
            self._time_step_limit = 100000000

        self._drones_commands: Union[Dict[DroneAbstract, Dict[Union[str, Controller], Command]], Type[None]] = None
        if self._drones:
            self._drones_commands = {}

        self._messages = None


        self._playground.window.on_draw = self.on_draw
        self._playground.window.on_update = self.on_update
        self._playground.window.set_update_rate(FRAME_RATE)

        # 'number_wounded_persons' is the number of wounded persons that should be retrieved by the drones.
        self._percent_drones_destroyed = 0.0
        self._mean_drones_health = 0.0

        self._total_number_wounded_persons = self._the_map.number_wounded_persons
        self._rescued_number = 0
        self._rescued_all_time_step = 0
        self._elapsed_time = 0
        self._start_real_time = time.time()
        self._real_time_limit_reached = False
        self._real_time_elapsed = 0

        self._last_image = None
        self._terminate = False

    def run(self):
        self._playground.window.run()

    def on_draw(self):
        self._playground.window.clear()
        self._fbo.use()

    def on_update(self, delta_time):
        self._elapsed_time += 1

        if self._elapsed_time < 2:
            self._playground.step(commands=self._drones_commands, messages=self._messages)
            # self._the_map.explored_map.update(self._drones)
            # self._the_map.explored_map._process_positions()
            # self._the_map.explored_map.display()
            return

        self._the_map.explored_map.update_drones(self._drones)
        # self._the_map.explored_map._process_positions()
        # self._the_map.explored_map.display()

        # COMPUTE ALL THE MESSAGES
        self._messages = self.collect_all_messages(self._drones)

        # COMPUTE COMMANDS
        for i in range(self._number_drones):
            command = self._drones[i].control()

            self._drones_commands[self._drones[i]] = command

        if self._drones:
            self._drones[0].display()

        self._playground.step(commands=self._drones_commands)

        # self._the_map.explored_map.display()

        # REWARDS
        new_reward = 0
        for i in range(self._number_drones):
            new_reward += self._drones[i].reward

        if new_reward != 0:
            self._rescued_number += new_reward

        if self._rescued_number == self._total_number_wounded_persons and self._rescued_all_time_step == 0:
            self._rescued_all_time_step = self._elapsed_time

        end_real_time = time.time()
        last_real_time_elapsed = self._real_time_elapsed
        self._real_time_elapsed = (end_real_time - self._start_real_time)
        delta = self._real_time_elapsed - last_real_time_elapsed
        # if delta > 0.5:
        #     print("self._real_time_elapsed = {:.1f}, delta={:.1f}, freq={:.1f}, freq moy={:.1f}".format(
        #         self._real_time_elapsed,
        #         delta,
        #         1 / (delta + 0.0001),
        #         self._elapsed_time / (self._real_time_elapsed + 0.00001)))
        if self._real_time_elapsed > self._real_time_limit:
            self._real_time_elapsed = self._real_time_limit
            self._real_time_limit_reached = True
            self._terminate = True

        if self._elapsed_time > self._time_step_limit:
            self._elapsed_time = self._time_step_limit
            self._terminate = True

        if self._print_rewards:
            for agent in self._playground.agents:
                if agent.reward != 0:
                    print(agent.reward)

        if self._print_messages:
            for drone in self._playground.agents:
                for comm in drone.communicators:
                    for _, msg in comm.received_messages:
                        print(f"Drone {drone.name} received message {msg}")

        self._messages = {}

        # print("can_grasp: {}, entities: {}".format(self._drone.base.grasper.can_grasp,
        #                                            self._drone.base.grasper.grasped_entities))

        if self._terminate:
            self.compute_health_stats()
            self._last_image = self.get_playground_image()
            arcade.close_window()

    def get_playground_image(self):
        self.update()
        # The image should be flip and the color channel permuted
        image = cv2.flip(self.get_np_img(), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def collect_all_messages(self, drones: List[DroneAbstract]):
        messages: SentMessagesDict = {}
        for i in range(self._number_drones):
            msg_data = drones[i].define_message_for_all()
            messages[drones[i]] = {drones[i].communicator: (None, msg_data)}
        return messages

    def compute_health_stats(self):
        sum_health = 0
        for drone in self._playground.agents:
            sum_health += drone.drone_health

        nb_destroyed = self._number_drones - len(self._playground.agents)

        if self._number_drones > 0:
            self._mean_drones_health = sum_health / self._number_drones
            self._percent_drones_destroyed = nb_destroyed / self._number_drones * 100
        else:
            self._mean_drones_health = DRONE_INITIAL_HEALTH
            self._percent_drones_destroyed = 0.0

    @property
    def last_image(self):
        return self._last_image

    @property
    def percent_drones_destroyed(self):
        return self._percent_drones_destroyed

    @property
    def mean_drones_health(self):
        return self._mean_drones_health

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @property
    def real_time_elapsed(self):
        return self._real_time_elapsed

    @property
    def rescued_number(self):
        return self._rescued_number

    @property
    def rescued_all_time_step(self):
        return self._rescued_all_time_step

    @property
    def real_time_limit_reached(self):
        return self._real_time_limit_reached
