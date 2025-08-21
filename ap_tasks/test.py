import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from ap_tasks.core import Task
from ap_tasks.config import PROJECT_ROOT

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    task = Task.from_stef(PROJECT_ROOT + '/suite/eye_tracking/eye_tracking.xml')

    # Start the viewer
    viewer = mujoco.viewer.launch_passive(task.model, task.data)

    # Main loop
    while viewer.is_running():
        step_starting_time = time.time()

        task.step(None)

        # Render
        viewer.sync()

        # wait to sync time
        step_runtime = time.time() - step_starting_time
        time.sleep(max(0., task.config['timestep_ms'] / 1000 - step_runtime))

