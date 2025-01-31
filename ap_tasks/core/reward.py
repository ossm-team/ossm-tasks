import abc
from abc import ABC


class Reward(ABC):
    """Reward base class.

    This is the class to implement to create new custom reward functions."""

    def __init__(self):
        self._parameters = self._default_parameters().copy()

    @abc.abstractmethod
    def reward_function(self, info):
        """Compute the reward."""
        raise NotImplementedError

    @abc.abstractmethod
    def _default_parameters(self) -> dict:
        raise NotImplementedError

    @property
    def parameters(self) -> dict:
        return self._parameters.copy()

    @property
    def parameter_list(self) -> list:
        return list(self._default_parameters().keys())

    def set_parameters(self, **kwargs):
        """Set the parameters of the reward function."""
        assert set(kwargs.keys()).issubset(self.parameter_list), "Invalid parameter name."

        for key, value in kwargs.items():
            self._parameters[key] = value

    def reset_parameters(self) -> None:
        """Reset the parameters to the default values."""
        self._parameters = self._default_parameters().copy()

    def validate(self):
        """Validate the reward function against the parameter list."""
        pass