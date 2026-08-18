"""Policy protocol - the seam between the action model and everything else.

A Policy maps (image + language instruction) -> 7-DoF action. That is exactly
OpenVLA's contract, so the mock, the local OpenVLA and the remote OpenVLA
client are drop-in replacements for one another.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rvc.types import Action, Observation


class PolicyUnavailable(RuntimeError):
    """Backend cannot run here. Carries a precise, user-facing reason."""


@runtime_checkable
class Policy(Protocol):
    #: e.g. "openvla-7b", "openvla-remote", "scripted-mock"
    name: str
    #: True when this is NOT a real VLA. Propagates into every log and report.
    degraded: bool
    degraded_reason: str

    def predict(self, obs: Observation) -> Action: ...

    def describe(self) -> str:
        """One line shown in the terminal banner."""
        ...
