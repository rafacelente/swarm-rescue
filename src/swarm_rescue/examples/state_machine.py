from statemachine import StateMachine, State
import numpy as np
import pydot

class InformedSimpleDrone(StateMachine):

    idle = State(initial=True)
    rotate = State()
    move = State()
    grasp = State()
    victory_spin = State()

    # Idle
    search_target = idle.to(idle, unless="target_found") | idle.to(rotate, cond="target_found")
    
    # Rotate
    rotate_to_target = rotate.to(rotate, unless="aligned") | rotate.to(move, cond="aligned")

    # Move
    move_to_target = move.to(move, unless="close_enough") | move.to(grasp, cond="close_enough")

    # Grasp
    #grasp_target = grasp.to(grasp, unless="has_target") | grasp.to(idle, cond="has_target")
    grasp_target = grasp.to(grasp, unless="has_target") | grasp.to(victory_spin, cond="has_target")

    # Celebrate spin (debug)
    spin = victory_spin.to(victory_spin, unless="false_cond") | victory_spin.to(idle, cond="false_cond")

    # rotate_to_target = rotate.to(move, on="aligned")
    # move_to_target = move.to(grasp, on="close_enough")
    # grasp_target = grasp.to(idle, on="has_target")

    cycle = search_target | rotate_to_target | move_to_target | grasp_target | spin

    def __init__(self, drone):
        self.drone = drone
        self.command = None
        super().__init__()

    def false_cond(self):
        return False

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        message = ". " + message if message else ""
        print(f"Running {event} from {source.id} to {target.id}{message}")
    
        
    def on_enter_idle(self):
        self.command = {"forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 1}
    
    def on_enter_rotate(self):
        angle = self.drone.get_rotate_angle()
        if angle > 0:
            self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.6}
        else:
            self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": -0.6}

    def on_enter_move(self):
        self.command = {"forward": 0.6,
                        "lateral": 0.0,
                        "rotation": 0.0}
        
    def on_exit_move(self):
        self.command = {"forward": -0.6,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
    
    def on_enter_grasp(self):
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 1}
    def on_exit_grasp(self):
        self.drone.assigned_target = False
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 1}
        
    def on_enter_victory_spin(self):
        print('uhuuul')
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 1}

    def get_command(self):
        return self.command
    # def before_cycle(self, event: str, source: State, target: State, message: str = ""):
    #     message = ". " + message if message else ""
    #     return f"Running {event} from {source.id} to {target.id}{message}"
    
    def target_found(self):
        return self.drone.assigned_target # Implement target_location
    
    def aligned(self):
        return abs(self.drone.diff_angle) < self.drone.alignment_threshold # < 0.1
    
    def close_enough(self):
        return np.linalg.norm(self.drone.current_location - self.drone.target_location) < self.drone.close_enough_threshold # implement close_enough_threshold and current_location
    
    def has_target(self):
        #return True # implement a way to check if the drone has the object
        return len(self.drone.grasped_entities()) > 0