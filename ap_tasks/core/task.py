import os
import tempfile

import gymnasium as gym
import mujoco
import numpy
from dm_control.mujoco.wrapper import MjData

import ap_tasks as apt
from ap_tasks.config import PROJECT_ROOT
from ap_tasks.config import SCHEMA_PATH
from ap_tasks.util.parsing import parse_stef_xml
from dm_control import mjcf


def _evaluate_data_variable(data: MjData, variable):
    """ Evaluate the value of a variable in the Mujoco data context.

    Variables are identified by a path through the data structure, with children separated by dots.
    """
    parts = variable.split('.')

    root = data.body(parts[0])

    return root




class Task(gym.Env):
    """
    Task represents a sensorimotor task defined in a STEF XML file.
    It builds its environment config, body parts, environment objects, and reward function dynamically.
    """

    def __init__(
            self,
            config,
            body_parts,
            objectives,
            external_objects=None,
            events=None,
            scene=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Core config
        self.config = config
        self.body_parts = body_parts
        self.objectives = objectives

        # Optional task components
        self.external_objects = external_objects or []
        self.events = events or []
        self.scene = scene or {
            "background_color": [1.0, 1.0, 1.0],
            "camera_position": [0, 0, 2],
            "camera_lookat": [0, 0, 0]
        }

        # Validate required fields
        if not self.config:
            raise ValueError("Task configuration is required but not provided.")
        if not self.body_parts:
            raise ValueError("At least one body part must be defined in the task.")
        if not self.objectives:
            raise ValueError("At least one objective must be defined in the task.")

        # Build task components
        self.model, self.data = self._build_model()
        self.events = self._build_events()
        self.objectives = self._build_objectives()

    def compute_reward(self, info):
        """Compute the reward based on parsed or custom reward function."""
        return self.objectives.reward_function(info)

    def _build_model(self):
        """ Build the Mujoco model from parsed STEF XML data. """
        root = mjcf.RootElement()
        root.option.timestep = float(self.config.get("timestep_ms", 10)) / 1000.0
        root.option.gravity = [0, 0, 0]  # Default gravity

        # Apply scene configuration
        if self.scene:
            vg = root.visual.__getattr__('global')
            vg.fovy = 90
            vg.azimuth = 120
            vg.elevation = -30

            root.visual.headlight.set_attributes(
                ambient=(0.4, 0.4, 0.4),
                diffuse=(0.8, 0.8, 0.8),
                specular=(0.8, 0.8, 0.8),
            )

            root.asset.add(
                "material",
                name="grid",
                texrepeat=(5, 5),
                texuniform=True,
                reflectance=0.2,
            )

            root.worldbody.add(
                "geom",
                type="plane",
                size=(0, 0, 0.05),
                pos=(0, 0, -0.1),
                material="grid",
                contype=0,
                conaffinity=0,
            )

        # TODO validate all bodyparts and environment objects to include bodies

        self._agent_body_elements = []
        self._external_objects_elements = []

        # Attach each body part and environment object as a namespaced instance
        for part in self.body_parts:
            part_model = mjcf.from_path(os.path.join(PROJECT_ROOT, part["model_path"]))
            root.attach(part_model)
            self._agent_body_elements.append(part_model.root)

        for obj in self.external_objects:
            obj_model = mjcf.from_path(os.path.join(PROJECT_ROOT, obj["model_path"]))
            bodies = obj_model.worldbody.get_children('body')

            if not bodies:
                raise ValueError(f"Environment object {obj['name']} has no bodies defined in its model.")

            if len(bodies) > 1:
                raise ValueError(f"Environment object {obj['name']} has multiple bodies defined. Only one is allowed.")

            body = bodies[0]
            body.name = obj["name"]

            if bool(body.mocap):
                raise AttributeError(f"Environment objects cannot be mocap bodies, but {obj['name']} is.")

            attachment_site = root.attach(body.root)

            # add a free joint to make the object movable if it partakes in a movement event
            if obj['name'] in [e.get('actor') for e in self.events if e.get('type').split('.')[0] == 'movement']:
                attachment_site.add('freejoint', name=f'{obj["name"]}_freejoint')
                print(f"Added free joint to environment object {obj['name']} for movement events.")

            self._external_objects_elements.append(obj_model.root)

        model = mujoco.MjModel.from_xml_string(
            root.to_xml_string()
        )

        data = mujoco.MjData(model)

        return model, data

    def _build_events(self):
        """ Build events from the task's event definitions."""

        events = []

        for event_descr in self.events:
            event_type = event_descr.get("type")
            event_actor = event_descr.get("actor")
            event_params = {p['name']: p['value'] for p in event_descr.get("param", [])}

            assert event_type in apt.core.event.EVENT_MAP.keys(), f'Unknown event type: {event_type}'

            # Create the event instance using the factory method
            event_class = apt.core.event.EVENT_MAP[event_type]
            event = event_class(actor=event_actor, model=self.model, data=self.data, params=event_params)
            events.append(event)

        return events

    def _build_objectives(self):
        """ Build the reward function from the task's objectives. """
        if not self.objectives:
            raise ValueError("No objectives defined for the task.")

        objectives = []

        def _make_expr(item, data):
            if 'constituent' in item:
                # Nested group → sum of child expressions
                subs = item['constituent']
                f = getattr(numpy, item.get('function', None))
                if not callable(f):
                    raise ValueError(f"Function '{item.get('function')}' is not callable or not defined in numpy.")

                if isinstance(subs, dict):
                    subs = [subs]
                child_lambdas = [_make_expr(s, data) for s in subs]

                return lambda: f(sum(child_lambdas))  # todo allow for products etc as well
            else:
                # Leaf node → simple (weight * var) ** exponent
                variable = item.get('variable')
                weight = float(item.get('weight', 1.0))
                exponent = float(item.get('exponent', 1.0))
                return lambda: (weight * _evaluate_data_variable(data, variable) ** exponent)

        for objective_descr in self.objectives:
            maximize = objective_descr.get("maximize")
            name = objective_descr["name"]
            constituents = objective_descr.get("constituent", [])

            if not constituents:
                raise ValueError(f"Objective '{name}' has no constituents defined.")

            objective_parts = []
            for constituent in constituents:
                if not 'function' in constituent.keys() and not 'variable' in constituent.keys():
                    raise ValueError(f"Constituent of objective '{name}' must have either 'function' or 'variable' defined.")

                objective_parts.append(_make_expr(constituent, self.data))

            objectives.append(lambda: sum(objective_parts))

        return objectives

    @classmethod
    def from_stef(cls, stef_file):
        """
        Create a Task instance directly from a STEF XML file.
        """
        # Parse the XML with your custom parser
        model_STEF = parse_stef_xml(stef_file,
                               schema_file=SCHEMA_PATH)

        # Extract sections from the parsed model
        config = model_STEF.get("config")
        body_parts = model_STEF.get("body").get("bodypart", [])
        environment_objects = model_STEF.get("environment").get("externalobject", [])
        objectives = model_STEF.get("objective")
        scene = model_STEF.get("environment", {}).get("scene")
        events = model_STEF.get("taskevent", [])

        # Build the Task instance
        return cls(config=config,
                   body_parts=body_parts,
                   external_objects=environment_objects,
                   objectives=objectives,
                   events=events,
                   scene=scene)

    def to_stef(self):
        """
        Serialize this Task back to a STEF XML file.
        Not yet implemented.
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """
        Optionally, implement Gymnasium-compatible reset().
        """
        # TODO: Initialize your simulation state here
        raise NotImplementedError

    def step(self, action):
        """ Step the environment forward by one time step. """
        for event in self.events:
            event.update(self.data.time)

        mujoco.mj_step(self.model, self.data)


if __name__ == "__main__":
    # Example usage
    task = Task.from_stef("../suite/eye_tracking/eye_tracking.xml")
    print("Task created with config:", task.config)
    print("Body parts:", task.body_parts)
    print("Environment objects:", task.external_objects)
    print("Reward function:", task.objectives.reward_function)