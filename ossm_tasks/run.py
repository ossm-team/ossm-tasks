import os
import sys
import time

import mujoco.viewer

from ossm_tasks.core import Task
from ossm_tasks.config import PROJECT_ROOT

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    VISUALIZE = False

    task = Task.from_stef(PROJECT_ROOT + '/suite/eye_tracking/eye_tracking.xml')

    # Start the viewer
    if VISUALIZE:
        viewer = mujoco.viewer.launch_passive(task.model, task.data)

    # Main loop
    done = False
    while not done:
        if VISUALIZE:
            if not viewer.is_running():
                break

        step_starting_time = time.time()

        # Step
        stimulus, reward, done, _, info = task.step(task.action_space.sample())

        print(f"Step reward: {reward}, done: {done}, info: {info}")

        # Render
        if VISUALIZE:
            viewer.sync()

        # wait to sync time
        step_runtime = time.time() - step_starting_time
        time.sleep(max(0., task.config['timestep_ms'] / 1000 - step_runtime))

