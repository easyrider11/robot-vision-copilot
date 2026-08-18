"""Environment protocol.

Any simulator we plug in - the built-in tabletop sim, real LIBERO, or later a
Gazebo/ROS 2 bridge - implements this. The agent state machine is written
against this interface only, so swapping the simulator never touches the agent.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rvc.types import Action, Observation


@runtime_checkable
class Env(Protocol):
    #: Short identifier that ends up in the run summary, e.g. "tabletop-sim".
    name: str
    #: True when this env is a stand-in rather than the real benchmark.
    degraded: bool
    #: Human-readable explanation of *why* it is degraded ("" when it is not).
    degraded_reason: str

    def reset(self) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """Apply a 7-DoF action. Returns (obs, reward, done, info)."""
        ...

    @property
    def instruction(self) -> str: ...

    def close(self) -> None: ...
