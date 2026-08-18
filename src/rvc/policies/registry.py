"""Backend resolution with honest, auditable degradation.

`resolve_policy` never silently pretends. It returns the policy it managed to
build *plus* a full trace of what it tried and why each option was rejected.
That trace is printed in the terminal banner and written into `summary.json`,
so a run can never be mistaken for a real OpenVLA rollout when it wasn't one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rvc.policies.base import Policy, PolicyUnavailable

BACKENDS = ("auto", "openvla-local", "openvla-remote", "mock")


@dataclass
class Resolution:
    chosen: str
    policy: Policy
    degraded: bool
    degraded_reason: str
    attempts: list[tuple[str, str]] = field(default_factory=list)  # (backend, reason-or-"OK")

    def banner_lines(self) -> list[str]:
        lines = [f"策略后端 policy backend : {self.chosen}"]
        for name, why in self.attempts:
            mark = "OK  " if why == "OK" else "SKIP"
            lines.append(f"  [{mark}] {name}: {why}")
        return lines


def _try_local(unnorm_key: str, model_id: str) -> Policy:
    from rvc.policies.openvla_local import OpenVLALocalPolicy

    return OpenVLALocalPolicy(model_id=model_id, unnorm_key=unnorm_key)


def _try_remote(unnorm_key: str, url: str) -> Policy:
    from rvc.policies.openvla_remote import OpenVLARemotePolicy

    return OpenVLARemotePolicy(url=url, unnorm_key=unnorm_key)


def resolve_policy(
    backend: str = "auto",
    *,
    unnorm_key: str = "bridge_orig",
    model_id: str = "openvla/openvla-7b",
    remote_url: str | None = None,
    allow_degraded: bool = True,
    mock_noise: float = 0.0,
    seed: int = 0,
) -> Resolution:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")

    remote_url = remote_url or os.environ.get("RVC_VLA_URL") or ""
    attempts: list[tuple[str, str]] = []

    order: list[str]
    if backend == "auto":
        order = ["openvla-local"]
        if remote_url:
            order.append("openvla-remote")
        else:
            attempts.append(("openvla-remote", "no RVC_VLA_URL set - skipped"))
        order.append("mock")
    else:
        order = [backend]

    for name in order:
        try:
            if name == "openvla-local":
                pol = _try_local(unnorm_key, model_id)
            elif name == "openvla-remote":
                if not remote_url:
                    raise PolicyUnavailable(
                        "openvla-remote requires --remote-url or RVC_VLA_URL"
                    )
                pol = _try_remote(unnorm_key, remote_url)
            elif name == "mock":
                from rvc.policies.mock import ScriptedMockPolicy

                pol = ScriptedMockPolicy(noise=mock_noise, seed=seed)
            else:  # pragma: no cover
                raise PolicyUnavailable(f"unhandled backend {name}")
        except PolicyUnavailable as exc:
            attempts.append((name, str(exc)))
            continue
        except Exception as exc:  # unexpected, still must not be silent
            attempts.append((name, f"{type(exc).__name__}: {exc}"))
            continue

        attempts.append((name, "OK"))
        if pol.degraded and not allow_degraded:
            raise PolicyUnavailable(
                f"resolved to degraded backend {name!r} but --no-degraded was requested.\n"
                + "\n".join(f"  - {n}: {w}" for n, w in attempts)
            )
        return Resolution(
            chosen=name,
            policy=pol,
            degraded=pol.degraded,
            degraded_reason=pol.degraded_reason,
            attempts=attempts,
        )

    raise PolicyUnavailable(
        "no usable policy backend.\n" + "\n".join(f"  - {n}: {w}" for n, w in attempts)
    )
