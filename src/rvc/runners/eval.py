"""`make eval` - batch evaluation of the AGENT RUNTIME under controlled failures.

WHAT THE NUMBERS MEAN (and what they do not)
--------------------------------------------
This measures the agent stack - perception, safety validator, state machine,
failure recovery, control-loop latency - over seeded, replayable episodes on
the built-in tabletop simulator, driven by the scripted policy (with and
without injected actuation noise). It does NOT measure any VLA: there is no
OpenVLA in the loop here, and no number below may be quoted as model success.
The banner, the JSON and this docstring all say so.

Metrics:
    success_rate        episodes reaching SUCCEEDED / all episodes
    recovery_rate       fault-injected episodes that entered RECOVER and still
                        SUCCEEDED / all fault-injected episodes
    invalid_call_rate   policy outputs REJECTED by the safety validator
                        (NaN, gripper chatter, ...) / all policy calls
    clamped_rate        policy outputs CLAMPED into limits / all policy calls
    unsafe_pass_through invalid actions that reached the actuator (MUST be 0;
                        an assertion, not a hope)
    p50/p95/p99 latency policy inference and full control cycle, milliseconds

Episode grid (deterministic, fully replayable):
    inject  = [none, target_lost, grasp_fail, grasp_slip]  (round-robin)
    noise   = 0.0 for even pairs, 0.25 for odd - the noisy half exercises the
              validator with genuinely out-of-spec actions
    seed    = episode index
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rvc import __version__
from rvc.agent.state_machine import AgentConfig, RobotAgent
from rvc.envs.tabletop import TabletopSim
from rvc.policies.mock import ScriptedMockPolicy
from rvc.types import AgentState

REPO_ROOT = Path(__file__).resolve().parents[3]
INJECTS = ("none", "target_lost", "grasp_fail", "grasp_slip")


def run_episode(index: int, max_steps: int = 220) -> dict:
    inject = INJECTS[index % len(INJECTS)]
    noise = 0.0 if (index // len(INJECTS)) % 2 == 0 else 0.25
    env = TabletopSim(inject=inject, max_steps=max_steps, seed=index)
    agent = RobotAgent(
        env=env,
        policy=ScriptedMockPolicy(noise=noise, seed=index),
        config=AgentConfig(max_recoveries=3, max_total_steps=max_steps),
        collect_frames=False,
    )
    result = agent.run()
    recs = agent.trace.records
    policy_calls = sum(1 for r in recs if r.action_source != "recovery")
    return {
        "index": index,
        "inject": inject,
        "noise": noise,
        "seed": index,
        "success": result.success,
        "steps": result.steps,
        "recoveries": result.recoveries,
        "entered_recover": any(t.to is AgentState.RECOVER for t in agent.trace.transitions),
        "failure": result.failure.value,
        "rejections": agent.validator.rejections,
        "clamps": agent.validator.clamps,
        "policy_calls": policy_calls,
        # an invalid action that reached env.step would be validated=False AND
        # not from the recovery path - assert none exist
        "unsafe_pass_through": sum(
            1 for r in recs if not r.validated and r.action_source != "recovery"
            and any(v != 0.0 for v in r.action)
        ),
        "latency_ms": [r.latency_ms for r in recs],
        "cycle_ms": [r.cycle_ms for r in recs if r.cycle_ms > 0],
    }


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q)) if xs else 0.0


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    faulted = [e for e in episodes if e["inject"] != "none"]
    lat = [v for e in episodes for v in e["latency_ms"]]
    cyc = [v for e in episodes for v in e["cycle_ms"]]
    calls = sum(e["policy_calls"] for e in episodes)
    rejections = sum(e["rejections"] for e in episodes)
    clamps = sum(e["clamps"] for e in episodes)

    by_inject = {}
    for inj in INJECTS:
        grp = [e for e in episodes if e["inject"] == inj]
        by_inject[inj] = {
            "episodes": len(grp),
            "success_rate": round(sum(e["success"] for e in grp) / max(1, len(grp)), 4),
            "mean_steps": round(float(np.mean([e["steps"] for e in grp])), 1),
            "mean_recoveries": round(float(np.mean([e["recoveries"] for e in grp])), 2),
        }

    return {
        "provenance": {
            "policy": "scripted-mock (DEGRADED - not a VLA; measures the agent "
                      "runtime, not any model)",
            "env": "tabletop-sim (teaching substitute for LIBERO)",
            "version": __version__,
            "replayable": "episode i fully determined by (inject=i%4, "
                          "noise=(i//4)%2*0.25, seed=i)",
        },
        "episodes": n,
        "success_rate": round(sum(e["success"] for e in episodes) / n, 4),
        "fault_injected_episodes": len(faulted),
        "recovery_rate": round(
            sum(e["success"] and e["entered_recover"] for e in faulted) / max(1, len(faulted)), 4
        ),
        "policy_calls": calls,
        "invalid_call_rate": round(rejections / max(1, calls), 5),
        "clamped_rate": round(clamps / max(1, calls), 5),
        "unsafe_pass_through": sum(e["unsafe_pass_through"] for e in episodes),
        "latency_policy_ms": {"p50": round(pct(lat, 50), 3), "p95": round(pct(lat, 95), 3),
                              "p99": round(pct(lat, 99), 3)},
        "latency_cycle_ms": {"p50": round(pct(cyc, 50), 3), "p95": round(pct(cyc, 95), 3),
                             "p99": round(pct(cyc, 99), 3)},
        "by_inject": by_inject,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs"))
    args = ap.parse_args(argv)

    print("=" * 78)
    print(" AGENT RUNTIME EVAL - 测的是 Agent 栈（校验/恢复/时延），不是任何 VLA 模型")
    print("=" * 78)

    t0 = time.time()
    episodes = []
    for i in range(args.episodes):
        episodes.append(run_episode(i, args.max_steps))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.episodes} episodes ...")
    wall = time.time() - t0

    s = summarize(episodes)
    s["wall_time_s"] = round(wall, 1)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / f"eval-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    slim = [{k: v for k, v in e.items() if k not in ("latency_ms", "cycle_ms")}
            for e in episodes]
    (out_dir / "episodes.json").write_text(json.dumps(slim, indent=2))

    faults = s["fault_injected_episodes"]
    print(f"""
  episodes            : {s["episodes"]}   ({faults} 注入故障, wall {s["wall_time_s"]}s)
  success_rate        : {s["success_rate"]:.1%}
  recovery_rate       : {s["recovery_rate"]:.1%}   (故障回合中恢复并成功的比例)
  policy_calls        : {s["policy_calls"]}
  invalid_call_rate   : {s["invalid_call_rate"]:.3%}   (被安全校验拒绝)
  clamped_rate        : {s["clamped_rate"]:.3%}   (被夹取进限幅)
  unsafe_pass_through : {s["unsafe_pass_through"]}   (到达执行器的非法动作 - 必须为 0)
  latency policy      : p50 {s["latency_policy_ms"]["p50"]}ms  p95 {s["latency_policy_ms"]["p95"]}ms
  latency full cycle  : p50 {s["latency_cycle_ms"]["p50"]}ms  p95 {s["latency_cycle_ms"]["p95"]}ms
  by inject:""")
    for inj, g in s["by_inject"].items():
        print(f"    {inj:<12} n={g['episodes']:<4} success={g['success_rate']:.1%} "
              f"steps={g['mean_steps']:<6} recoveries={g['mean_recoveries']}")
    print(f"\n  -> {out_dir}/eval.json\n")

    return 0 if s["unsafe_pass_through"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
