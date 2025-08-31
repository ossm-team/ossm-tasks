import abc

import numpy as np

from ossm_tasks.core.event import TaskEvent


class Movement(TaskEvent, abc.ABC):

    def _build(self, params):
        self.duration = float(params.get('duration_ms'))
        self.loop = params.get('loop')
        self.startpoint = self.data.qpos[self.actor_qpos_idx:self.actor_qpos_idx + 3].copy()


class LineMovement(Movement):

    def _build(self, params):
        super()._build(params)

        try:
            self.endpoint = np.fromstring(params.get('endpoint'), sep=' ')
        except ValueError:
            try:
                self.endpoint = np.fromstring(params.get('startpoint'), dtype=int, sep=' ')
            except ValueError:
                raise ValueError("Invalid endpoint or startpoint format. Expected a space-separated string of numbers.")

    def update(self, time_step):
        """ Update the position of the actor. """

        # Calculate the fraction of the duration that has passed
        fraction = min(time_step / self.duration, 1.0)

        # Calculate the new position
        new_position = self.startpoint + fraction * (self.endpoint - self.startpoint)

        # print(f"Updating position from {self.startpoint} via  {self.data.qpos[self.actor_qpos_idx:self.actor_qpos_idx + 3]} "
        #       f"to {new_position} at time step {time_step}")

        # Update the actor's position
        self.data.qpos[self.actor_qpos_idx:self.actor_qpos_idx + 7] = np.concat([new_position, [1, 0, 0, 0]])
