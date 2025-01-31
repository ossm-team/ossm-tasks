import gymnasium as gym

import ap_tasks as apt
from ap_tasks.util.parsing import parse_stef_xml


class Task(gym.Env):

    def __init__(self, reward: apt.Reward, *args, **kwargs):
        super().__init__(**args, **kwargs)

        self.reward = reward

    def compute_reward(self, info):
        return self.reward.reward_function(info)

    @classmethod
    def from_stef(cls, stef_file):
        model = parse_stef_xml(stef_file)

        return cls(model)

    def to_stef(self):
        raise NotImplementedError