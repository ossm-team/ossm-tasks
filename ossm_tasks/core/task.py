import os
import tempfile
from typing import Any
from typing import SupportsFloat

import gymnasium as gym
import mujoco
import numpy
import numpy as np
from dm_control.mujoco.wrapper import MjData

import ossm_tasks as apt
from ossm_tasks.config import PROJECT_ROOT
from ossm_tasks.config import SCHEMA_PATH
from ossm_tasks.util.parsing import parse_stef_xml
from dm_control import mjcf

from ossm_base.types import Stimulus





def _evaluate_data_variable(data: MjData, variable):
    """ Evaluate the value of a variable in the Mujoco data context.

    Variables are identified by a path through the data structure, with children separated by dots.
    """
    parts = variable.split('.')

    root = data.body(parts[0] + "/")

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
            actions,
            termination,
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
        self.actions = actions
        self.termination = termination

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

        # vars
        self.timestep = 0

        # Build task components
        self.model, self.data = self._build_model()
        self.events = self._build_events()
        self.objectives = self._build_objectives()

        self.is_terminated = self._build_termination_conditions()

        self.action_space = self._build_action_space()

    def compute_reward(self, info):
        """Compute the reward based on parsed or custom reward function."""
        objective_values = {f"objective_{i}": obj() for i, obj in enumerate(self.objectives)}

        reward = 0.0
        for value in objective_values.values():
            reward += value

        return reward, objective_values

    def get_state(self):
        return self.data.qpos

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
            if "model_path" in part.keys():
                part_model = mjcf.from_path(os.path.join(PROJECT_ROOT, part["model_path"]))
            else:
                part_model = mjcf.RootElement()
                dummy_body = part_model.worldbody.add('body', name=part.get('name', 'dummy'))
                dummy_body.add('geom', type='sphere', size=[1e-3], rgba=[0, 0, 0, 0])

            root.attach(part_model)
            self._agent_body_elements.append(part_model.root)

        for obj in self.external_objects:
            if "model_path" in obj.keys():
                obj_model = mjcf.from_path(os.path.join(PROJECT_ROOT, obj["model_path"]))
            else:
                obj_model = mjcf.RootElement()
                dummy_body = obj_model.worldbody.add('body', name=obj.get('name', 'dummy'))
                dummy_body.add('geom', type='sphere', size=[1e-3], rgba=[0, 0, 0, 0])

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

    def _evaluate_variable_numeric(self, var: str) -> float:
        """ Evaluate a variable to a numeric value."""

        comp_map = {"x": 0, "y": 1, "z": 2}
        name, comp = (var.split(".", 1) + [""])[:2]
        comp_idx = comp_map.get(comp)

        name = name + "/" if not name.endswith("/") else name

        try:
            v = self.data.site(name).xpos  # world position of site
            return float(v[comp_idx] if comp_idx is not None else v[0])
        except Exception:
            pass

        try:
            v = self.data.body(name).xpos  # world position of body COM
            return float(v[comp_idx] if comp_idx is not None else v[0])
        except Exception:
            pass

        try:
            v = self.data.geom(name).xpos  # world position of geom
            return float(v[comp_idx] if comp_idx is not None else v[0])
        except Exception:
            pass

        try:
            v = self.data.joint(name).qpos  # joint position
            return float(v[comp_idx] if comp_idx is not None else v[0])
        except Exception:
            pass

        raise ValueError(f"Variable '{var}' could not be resolved to a numeric value.")

    def _build_objectives(self):
        """Build the reward function callables from the task's objectives."""

        if not self.objectives:
            raise ValueError("No objectives defined for the task.")

        def _make_expr(item):
            # Group node: apply numpy function to children (default: sum)
            if 'constituent' in item:
                subs = item['constituent']
                if isinstance(subs, dict):
                    subs = [subs]
                child_funcs = [_make_expr(s) for s in subs]

                fname = item.get('function') or 'sum'
                f = getattr(np, fname, None)
                if not callable(f):
                    raise ValueError(f"Function '{fname}' is not callable or not in numpy.")

                # Evaluate children, stack as array when appropriate
                return lambda: f(np.array([g() for g in child_funcs], dtype=float))

            # Leaf: (weight * value) ** exponent
            variable = item.get('variable')
            if variable is None:
                raise ValueError("Leaf constituent requires 'variable'.")

            weight = float(item.get('weight', 1.0))
            exponent = float(item.get('exponent', 1.0))

            return lambda: np.power(weight * float(self._evaluate_variable_numeric(variable)), exponent)

        objective_funcs = []
        for obj in self.objectives:
            name = obj.get("name", "objective")
            parts = obj.get("constituent", [])
            if isinstance(parts, dict):
                parts = [parts]
            if not parts:
                raise ValueError(f"Objective '{name}' has no constituents defined.")

            part_funcs = [_make_expr(c) for c in parts]
            maximize = bool(obj.get("maximize", True))

            if maximize:
                objective_funcs.append(lambda pf=part_funcs: float(np.sum([f() for f in pf])))
            else:
                objective_funcs.append(lambda pf=part_funcs: -float(np.sum([f() for f in pf])))

        return objective_funcs

    def _build_termination_conditions(self):
        """ Build termination condition callables from the task's termination definitions. """

        term = getattr(self, "termination", None)
        self._termination_conditions = []

        # Helper to read '@key' or 'key'
        def _get(d, k, default=None):
            if d is None:
                return default
            return d.get(k, d.get(f"@{k}", default))

        # Normalize single-or-list fields to list
        def _as_list(v):
            if v is None:
                return []
            return v if isinstance(v, list) else [v]

        # Tries (a) objective values cache, (b) live info dict, (c) attributes on self, else 0.0.
        def _get_metric(name: str) -> float:
            # optional dotted path support like "moving_target.x" -> try dicts
            if hasattr(self, "metrics") and name in self.metrics:
                return float(self.metrics[name])
            if hasattr(self, "objective_values") and name in self.objective_values:
                return float(self.objective_values[name])
            # common live sources
            if hasattr(self, "last_info") and isinstance(self.last_info, dict):
                cur = self.last_info
                # dotted lookup
                for part in name.split('.'):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        cur = None
                        break
                if cur is not None:
                    try:
                        return float(cur)
                    except Exception:
                        pass
            # fallback: direct attribute (not dotted)
            if hasattr(self, name):
                try:
                    return float(getattr(self, name))
                except Exception:
                    pass
            return 0.0

        # Time condition(s)
        for t in _as_list(term.get("time")):
            value_ms = int(_get(t, "value_ms", 0))
            def make_time_cond(ms: int):
                return lambda: int(self.data.time * 1000.0) >= ms
            self._termination_conditions.append(make_time_cond(value_ms))

        # Shared builder for threshold-based conditions (objective/variable)
        def _build_threshold_cond(node: dict):
            target = _get(node, "target")
            if not target:
                raise ValueError("Termination condition missing required 'target'.")
            threshold = float(_get(node, "threshold", 0.0))
            direction = _get(node, "direction", "above").lower()  # 'above' or 'below'
            dur_ms = int(_get(node, "duration_ms", 0))

            # Stateful predicate with dwell time support
            state = {"since_ms": None}

            def cond():
                now_ms = int(self.data.time * 1000.0)
                val = _get_metric(target)
                ok = (val >= threshold) if direction == "above" else (val <= threshold)

                if not ok:
                    state["since_ms"] = None
                    return False

                if dur_ms <= 0:
                    return True  # instantaneous threshold

                if state["since_ms"] is None:
                    state["since_ms"] = now_ms
                    return False

                return (now_ms - state["since_ms"]) >= dur_ms

            return cond

        for n in _as_list(term.get("objective")):
            self._termination_conditions.append(_build_threshold_cond(n))

        for n in _as_list(term.get("variable")):
            self._termination_conditions.append(_build_threshold_cond(n))

        def _terminated():
            return any(c() for c in self._termination_conditions)

        return _terminated

    def _build_action_space(self):
        """
        Build a Gymnasium action space from STEF <action> entries.

        Expected self.actions element format per action (handles both xmlschema styles):
          {
            "@target_body" or "target_body": "fixation_point",
            "@shape" or "shape": "3" or 3,
            "min": {"@value" or "value": "-1 -1 0"},
            "max": {"@value" or "value": "1 1 0"}
          }
        """
        if not hasattr(self, "actions") or not self.actions:
            raise ValueError("No actions parsed from STEF; self.actions is empty.")

        def _get(attrdict, key):
            # support '@key' and 'key'
            return attrdict.get(key, attrdict.get(f"@{key}"))

        def _parse_vec(s, shape):
            if s is None:
                return None
            if isinstance(s, (list, tuple, np.ndarray)):
                v = np.asarray(s, dtype=np.float32)
            else:
                # split on whitespace and commas
                parts = [p for p in str(s).replace(",", " ").split() if p]
                v = np.array([float(p) for p in parts], dtype=np.float32)
            if v.size == 1 and shape > 1:
                v = np.full((shape,), v.item(), dtype=np.float32)
            if v.size != shape:
                raise ValueError(f"Vector size {v.size} does not match shape {shape} for action bounds '{s}'.")
            return v

        boxes = {}
        for i, act in enumerate(self.actions):
            target = _get(act, "target_body") or f"action_{i}"
            shape_raw = _get(act, "shape")
            if shape_raw is None:
                raise ValueError(f"Action '{target}': missing required 'shape'.")
            shape = int(shape_raw)

            min_node = act.get("min") or {}
            max_node = act.get("max") or {}
            low = _parse_vec(_get(min_node, "value"), shape)
            high = _parse_vec(_get(max_node, "value"), shape)

            # sensible defaults if omitted
            if low is None and high is None:
                low = -np.ones(shape, dtype=np.float32)
                high = np.ones(shape, dtype=np.float32)
            elif low is None:
                low = np.full((shape,), -np.inf, dtype=np.float32)
            elif high is None:
                high = np.full((shape,), np.inf, dtype=np.float32)

            # validate finite ordering where both provided
            mask = np.isfinite(low) & np.isfinite(high)
            if np.any(high[mask] < low[mask]):
                raise ValueError(f"Action '{target}': some high bounds are < low bounds.")

            boxes[target] = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Single action → Box; multiple → Dict
        if len(boxes) == 1:
            return next(iter(boxes.values()))

        return gym.spaces.Dict(boxes)


    @classmethod
    def from_stef(cls, stef_file) -> "Task":
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
        actions = model_STEF.get("action", [])
        termination = model_STEF.get("termination")

        # Build the Task instance
        return cls(config=config,
                   body_parts=body_parts,
                   actions=actions,
                   external_objects=environment_objects,
                   objectives=objectives,
                   events=events,
                   termination=termination,
                   scene=scene)

    def to_stef(self):
        """
        Serialize this Task back to a STEF XML file.
        Not yet implemented.
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        self.timestep = 0

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        super(Task, self).reset(*args, **kwargs)

        return Stimulus(data=self.get_state()), {}

    def step(self, action) -> tuple[Stimulus, SupportsFloat, bool, bool, dict[str, Any]]:
        """ Step the environment forward by one time step. """

        # validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in the action space {self.action_space}.")

        # apply action
        self.data.ctrl[:] = action

        # run events
        for event in self.events:
            event.update(self.data.time)

        mujoco.mj_step(self.model, self.data)

        # compute reward
        reward, _ = self.compute_reward(info={})

        # consider termination conditions
        terminated = self.is_terminated()

        # update vars
        self.timestep += 1

        return (
            Stimulus(data=self.get_state()),
            0, terminated, False,
            {
                "time": self.data.time,
                "timestep": self.timestep,
            }
        )