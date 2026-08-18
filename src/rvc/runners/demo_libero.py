"""`make demo-libero` - the Stage 1 minimal demo.

What it does, in order:
  1. PREFLIGHT  - probe every backend and print exactly what is and is not
                  available on this machine, with reasons.
  2. RESOLVE    - pick the best policy + env that actually work here.
  3. ROLLOUT    - run one episode through the full agent state machine,
                  explaining every step in the terminal.
  4. ARTIFACTS  - write frames, an animated GIF, actions.jsonl,
                  transitions.jsonl and summary.json under runs/.

It never pretends. If the policy is the scripted mock, every artifact and the
final banner say DEGRADED.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from rvc import compat
from rvc.agent.state_machine import AgentConfig, RobotAgent
from rvc.agent.verifier import RewardVerifier, TabletopVerifier
from rvc.envs.registry import resolve_env
from rvc.perception.detector import ColorDetector
from rvc.policies.registry import resolve_policy
from rvc.report import Reporter, banner, rule

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def preflight(remote_url: str) -> list[str]:
    """Probe everything without loading or downloading anything."""
    lines: list[str] = []

    from rvc.policies.openvla_local import probe as vla_probe

    ok, why = vla_probe()
    lines.append(f"[{'OK  ' if ok else 'NO  '}] OpenVLA-7B 本地推理 : {why or '可用'}")

    if remote_url:
        from rvc.policies.openvla_remote import probe as rem_probe

        ok, why = rem_probe(remote_url)
        lines.append(f"[{'OK  ' if ok else 'NO  '}] OpenVLA 远程服务    : {why or remote_url}")
    else:
        lines.append("[SKIP] OpenVLA 远程服务    : 未设置 RVC_VLA_URL / --remote-url")

    from rvc.envs.libero_env import probe as lib_probe

    ok, why = lib_probe()
    lines.append(f"[{'OK  ' if ok else 'NO  '}] LIBERO 仿真环境     : {why or '可用'}")
    lines.append("[OK  ] 内置 tabletop 仿真   : 可用（LIBERO 的教学替代，始终标记为降级）")
    lines.append("[OK  ] 脚本 mock 策略      : 可用（不是 VLA，始终标记为降级）")
    return lines


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def write_artifacts(run_dir: Path, agent: RobotAgent, result, extra: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"

    if agent.trace.frames:
        from PIL import Image

        frames_dir.mkdir(exist_ok=True)
        pil = []
        for step, arr in agent.trace.frames:
            im = Image.fromarray(arr)
            im.save(frames_dir / f"step_{step:04d}.png")
            pil.append(im)
        if pil:
            pil[0].save(
                run_dir / "rollout.gif",
                save_all=True,
                append_images=pil[1:],
                duration=90,
                loop=0,
            )

    with (run_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
        for rec in agent.trace.records:
            f.write(rec.to_json() + "\n")

    with (run_dir / "transitions.jsonl").open("w", encoding="utf-8") as f:
        for tr in agent.trace.transitions:
            f.write(
                json.dumps(
                    {"step": tr.step, "from": tr.frm.value, "to": tr.to.value,
                     "reason": tr.reason},
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = result.to_dict()
    summary.update(extra)
    summary["validator"] = {
        "rejections": agent.validator.rejections,
        "clamps": agent.validator.clamps,
    }
    summary["detections_missed"] = agent.trace.detections_missed
    summary["state_visits"] = {
        s: sum(1 for t in agent.trace.transitions if t.to.value == s)
        for s in sorted({t.to.value for t in agent.trace.transitions})
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvc-demo",
        description="Stage 1: minimal OpenVLA / LIBERO demo with an agent state machine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", default="auto",
                   choices=["auto", "openvla-local", "openvla-remote", "mock"],
                   help="动作模型后端；auto 会按 本地VLA -> 远程VLA -> mock 的顺序降级")
    p.add_argument("--env", default="auto", choices=["auto", "libero", "tabletop"])
    p.add_argument("--task", default="pick_place_block",
                   help="tabletop 任务 id；LIBERO 时忽略")
    p.add_argument("--libero-suite", default="libero_spatial")
    p.add_argument("--libero-task-index", type=int, default=0)
    p.add_argument("--mode", default=None, choices=["subgoal", "e2e"],
                   help="默认：tabletop 用 subgoal，LIBERO 用 e2e（子目标是 tabletop 专用的）")
    p.add_argument("--inject", default="none",
                   choices=["none", "target_lost", "grasp_fail", "grasp_slip"],
                   help="注入故障，用来观察 RECOVER 分支")
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--max-recoveries", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--explain", default="full", choices=["full", "compact"])
    p.add_argument("--no-frames", action="store_true", help="不保存图像，只写日志")
    p.add_argument("--no-degraded", action="store_true",
                   help="拒绝降级：拿不到真实 VLA 就直接失败退出")
    p.add_argument("--remote-url", default=os.environ.get("RVC_VLA_URL", ""))
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--model-id", default="openvla/openvla-7b")
    p.add_argument("--mock-noise", type=float, default=0.0)
    p.add_argument("--planner", default="rule", choices=["rule", "llm"],
                   help="llm = Anthropic-backed task decomposition + recovery choice; "
                        "falls back to rule-based (and says so) if the SDK/key is missing")
    p.add_argument("--detector", default="color", choices=["color", "yolo"],
                   help="yolo needs models/yolo-tabletop.pt (make yolo)")
    p.add_argument("--out", default=str(REPO_ROOT / "runs"))
    return p


def resolve_planner(kind: str):
    """rule | llm. The LLM path is an add-on: unavailable => rule-based, loudly."""
    from rvc.agent.planner import LLMPlanner, RuleBasedPlanner

    if kind != "llm":
        return RuleBasedPlanner(), "rule-based (deterministic, no network)"
    from rvc.agent.llm_anthropic import make_anthropic_completer

    complete, why = make_anthropic_completer()
    planner = LLMPlanner(complete=complete)
    if complete is None:
        return planner, f"llm requested but unavailable -> rule-based. ({why})"
    return planner, f"llm planner active ({why})"


def resolve_detector(kind: str):
    """color | yolo. YOLO is Stage-3 optional; missing weights => color, loudly."""
    if kind != "yolo":
        return ColorDetector(), "color-threshold detector"
    weights = REPO_ROOT / "models" / "yolo-tabletop.pt"
    if not weights.exists():
        return ColorDetector(), f"yolo requested but {weights} missing (run `make yolo`) -> color"
    try:
        from rvc.perception.detector import YoloDetector

        return YoloDetector(str(weights)), f"yolo detector ({weights.name})"
    except Exception as exc:
        return ColorDetector(), f"yolo unavailable ({type(exc).__name__}: {exc}) -> color"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(banner(preflight(args.remote_url), "PREFLIGHT · 依赖与模型可用性"))

    # -- resolve backends ----------------------------------------------------
    try:
        pol_res = resolve_policy(
            args.backend,
            unnorm_key=args.unnorm_key,
            model_id=args.model_id,
            remote_url=args.remote_url or None,
            allow_degraded=not args.no_degraded,
            mock_noise=args.mock_noise,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"\n无法获得可用的动作模型后端:\n{exc}\n", file=sys.stderr)
        return 2

    env_res = resolve_env(
        args.env,
        task_id=args.task,
        libero_suite=args.libero_suite,
        libero_task_index=args.libero_task_index,
        max_steps=args.max_steps,
        inject=args.inject,
        seed=args.seed,
       policy_kind=pol_res.chosen,
    )
    print(banner(pol_res.banner_lines() + env_res.banner_lines(), "RESOLVE · 实际使用的后端"))

    if pol_res.degraded or env_res.degraded:
        print(rule("⚠  DEGRADED DEMO 降级演示", "═"))
        for why in (pol_res.degraded_reason, env_res.degraded_reason):
            if why:
                print(f"  · {why}")
        print("  · 本次运行的成功率不能作为 OpenVLA 的性能证据。")
        print(rule("", "═"))

    # Some policy x env pairings run but mean nothing. Say so loudly.
    notes = compat.check(pol_res.chosen, env_res.chosen)
    for note in notes:
        mark = {"error": "⛔", "warn": "⚠", "info": "ℹ"}[note.level]
        print(rule(f"{mark}  {note.title}", "═"))
        for line in note.lines:
            print(f"  {line}" if line else "")
        print(rule("", "═"))

    # The rule-based sub-goals ("move above the red block") are TabletopSim
    # vocabulary. On LIBERO the whole instruction goes straight to the model.
    mode = args.mode or ("e2e" if env_res.chosen == "libero" else "subgoal")
    if args.mode is None and mode == "e2e":
        print(f"  模式 mode 自动设为 e2e（LIBERO 任务：{env_res.env.instruction!r}）\n")

    planner, planner_note = resolve_planner(args.planner)
    detector, detector_note = resolve_detector(args.detector)
    print(f"  规划器 planner : {planner_note}")
    print(f"  检测器 detector: {detector_note}\n")

    # -- run -----------------------------------------------------------------
    reporter = Reporter(style=args.explain)
    agent = RobotAgent(
        env=env_res.env,
        policy=pol_res.policy,
        verifier=TabletopVerifier() if env_res.chosen == "tabletop" else RewardVerifier(),
        detector=detector,
        planner=planner,
        config=AgentConfig(
            max_recoveries=args.max_recoveries,
            max_total_steps=args.max_steps,
            mode=mode,
            perceive_required=env_res.chosen == "tabletop",
        ),
        on_event=reporter,
        collect_frames=not args.no_frames,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / f"{stamp}_{env_res.chosen}_{pol_res.chosen}_{args.inject}"
    result = agent.run(run_dir=str(run_dir))

    write_artifacts(run_dir, agent, result, extra={
        "policy_attempts": pol_res.attempts,
        "env_attempts": env_res.attempts,
        "args": vars(args),
        "mode": mode,
        "planner": getattr(agent.planner, "name", "?"),
        "planner_note": planner_note,
        "planner_last_error": getattr(agent.planner, "last_error", ""),
        "detector": getattr(agent.detector, "name", "?"),
        "detector_note": detector_note,
    })

    if getattr(agent.planner, "last_error", ""):
        print(f"  规划器备注 planner: {agent.planner.last_error}")
    print(f"  已写入 {len(agent.trace.records)} 条动作日志 -> {run_dir}/actions.jsonl")
    if agent.trace.frames:
        print(f"  已写入 {len(agent.trace.frames)} 帧图像 -> {run_dir}/frames/ 与 rollout.gif")
    print(f"  摘要 -> {run_dir}/summary.json\n")

    with contextlib.suppress(Exception):
        env_res.env.close()

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
