

from typing import Optional, List
        
class Reward():
    def __init__(
            self,
            name: str,
            value: Optional[float] = 0) -> None:
        
        self._name = name
        self._value = value
        self.n_hits = 0
        self.average = 0
    
    def add_value(self, value) -> None:
        self._value += value
        self.n_hits += 1
    
    def get_average(self) -> float:
        if self.n_hits != 0:
            return self._value/self.n_hits
        return self._value
    
    def get_nhits(self):
        return self.n_hits

    def get_name(self) -> str:
        return self._name
    
    def get_value(self) -> float:
        return self._value
    
class State():
    def __init__(
            self,
            name: str = None,
            rewards: List[Reward] = None,
            initial_value: Optional[float] = 0.0
    ) -> None:
        self._name = name
        self.rewards = rewards
        self._value = initial_value

    def get_name(self) -> str:
        return self._name
    
    def get_value(self) -> float:
        self._value = 0
        for reward in self.rewards:
            self._value += reward.get_value()
        return self._value
    
    def get_rewards(self) -> List[Reward]:
        return self.rewards

class RewardLogger():
    def __init__(
            self,
            states: List[State]
            ) -> None:
        self.states = states
        self.n