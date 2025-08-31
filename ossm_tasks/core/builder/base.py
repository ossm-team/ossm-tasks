import abc

from dm_control import mjcf

from .entities import Body, Stage


class PhysicalWorld(abc.ABC):
    """A physical world in which an environment is simulated."""

    @property
    @abc.abstractmethod
    def body(self) -> Body:
        ...

    @property
    @abc.abstractmethod
    def stage(self) -> Stage:
        ...

    @property
    def root(self) -> mjcf.RootElement:
        """Returns the root element of the physical world."""
        return self.stage.mjcf_model
