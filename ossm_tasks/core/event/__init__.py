from ossm_tasks.core.event.event import TaskEvent
from ossm_tasks.core.event.movement import LineMovement

EVENT_MAP = {
    'movement.line': LineMovement,
}