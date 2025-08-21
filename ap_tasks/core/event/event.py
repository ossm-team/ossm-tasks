import abc

from mujoco import MjData, MjModel


class TaskEvent(abc.ABC):

    def __init__(self, actor: str, model: MjModel, data: MjData, params: dict):
        self.actor = actor
        self.model = model
        self.data = data

        self.actor_qpos_idx = self.model.joint(f'{actor}/{actor}_freejoint/').qposadr[0]

        self.params = params

        self._build(params)

    def _build(self, data):
        raise NotImplementedError()

    def update(self, time_step):
        """
        Update the event state based on the current time step.
        This method should be overridden by subclasses to implement specific event logic.
        """
        raise NotImplementedError()