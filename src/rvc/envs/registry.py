"""Environment resolution: real LIBERO if installed, tabletop sim otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field

ENVS = ("auto", "libero", "tabletop")


@dataclass
class EnvResolution:
    chosen: str
    env: object
    degraded: bool
    degraded_reason: str
    attempts: list[tuple[str, str]] = field(default_factory=list)

    def banner_lines(self) -> list[str]:
        lines = [f"仿真环境 environment    : {self.chosen}"]
        for name, why in self.attempts:
            lines.append(f"  [{'OK  ' if why == 'OK' else 'SKIP'}] {name}: {why}")
        return lines


def resolve_env(
    kind: str = "auto",
    *,
    task_id: str = "pick_place_block",
    libero_suite: str = "libero_spatial",
    libero_task_index: int = 0,
    max_steps: int = 120,
    inject: str = "none",
    seed: int = 0,
    policy_kind: str | None = None,
) -> EnvResolution:
    """Pick an environment. In `auto` mode the *policy* matters.

    `policy_kind` lets auto-resolution avoid a pairing that cannot possibly
    work: the scripted mock servos on TabletopSim's privileged state, so
    pointing it at LIBERO guarantees a 0% run. Preferring LIBERO whenever it is
    merely *installed* would make the out-of-the-box `make demo-libero` fail on
    every machine that ran `make setup-libero`. Explicit `--env libero` still
    honours the request - and `rvc.compat` shouts about it.
    """
    if kind not in ENVS:
        raise ValueError(f"unknown env {kind!r}; choose from {ENVS}")

    attempts: list[tuple[str, str]] = []
    if kind != "auto":
        order = [kind]
    elif policy_kind == "mock":
        attempts.append(
            ("libero", "跳过：mock 策略驱动不了 LIBERO（要跑请显式 --env libero）")
        )
        order = ["tabletop"]
    else:
        order = ["libero", "tabletop"]

    for name in order:
        if name == "libero":
            from rvc.envs.libero_env import LiberoEnv, LiberoUnavailable, probe

            ok, why = probe()
            if not ok:
                attempts.append(("libero", why))
                if kind == "libero":
                    raise LiberoUnavailable(why)
                continue
            try:
                env = LiberoEnv(
                    task_suite=libero_suite,
                    task_index=libero_task_index,
                    max_steps=max_steps,
                    seed=seed,
                )
            except Exception as exc:
                attempts.append(("libero", f"{type(exc).__name__}: {exc}"))
                if kind == "libero":
                    raise
                continue
            attempts.append(("libero", "OK"))
            return EnvResolution("libero", env, False, "", attempts)

        from rvc.envs.tabletop import TabletopSim

        env = TabletopSim(task_id=task_id, max_steps=max_steps, inject=inject, seed=seed)
        attempts.append(("tabletop", "OK"))
        return EnvResolution("tabletop", env, True, env.degraded_reason, attempts)

    raise RuntimeError("no environment could be constructed")  # pragma: no cover
