"""`make bc-data` / `make bc-train` / `make bc-eval` - the LIBERO BC baseline.

    data   download one LIBERO task's 50 human demos (~0.5 GB) from HF
    train  behaviour-clone ResNet18x2 + MLP on them (Apple MPS / CUDA / CPU)
    eval   roll the policy out on LIBERO through the agent runtime and report
           success rate over the task's official init states

Everything the numbers depend on is written to models/*.json and runs/bc-eval-*/
so they can be quoted with provenance. This is a LEARNED policy but NOT a VLA.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "external" / "libero-datasets"
HF_REPO = "yifengzhu-hf/LIBERO-datasets"


def parse_tasks(spec: str | int) -> list[int]:
    """--task-index accepts '0' or '0,1,2'."""
    return [int(x) for x in str(spec).split(",")]


def task_language(suite: str, index: int) -> str:
    from rvc.envs.libero_bootstrap import bootstrap

    ok, why = bootstrap()  # bare `import libero` fails without the path fixes
    if not ok:
        raise SystemExit(f"LIBERO not available: {why}")
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().get_task(index).language


def task_hdf5(suite: str, index: int) -> Path:
    """Resolve the demo file for (suite, index) via LIBERO's own task registry."""
    from rvc.envs.libero_bootstrap import bootstrap

    ok, why = bootstrap()
    if not ok:
        raise SystemExit(f"LIBERO not available: {why}")
    from libero.libero import benchmark

    task = benchmark.get_benchmark_dict()[suite]().get_task(index)
    return DATA_DIR / suite / task.bddl_file.replace(".bddl", "_demo.hdf5")


def cmd_data(args) -> int:
    from huggingface_hub import snapshot_download

    for idx in parse_tasks(args.task_index):
        path = task_hdf5(args.suite, idx)
        if path.exists():
            print(f"✓ already present: {path} ({path.stat().st_size / 1e6:.0f} MB)")
            continue
        print(f"[data] downloading {path.name} from {HF_REPO} ...")
        snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                          allow_patterns=[f"{args.suite}/{path.name}"], local_dir=str(DATA_DIR))
        print(f"✓ {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


def cmd_train(args) -> int:
    from rvc.policies.bc_libero import BCConfig, train

    tasks = parse_tasks(args.task_index)
    paths = [task_hdf5(args.suite, i) for i in tasks]
    for path in paths:
        if not path.exists():
            print(f"✗ no demos at {path} - run `make bc-data` first")
            return 1
    cfg = BCConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                   pretrained=not args.scratch, seed=args.seed)
    if len(tasks) == 1:
        print(f"[train] {paths[0].name}\n        {cfg}")
        meta = train(paths[0], args.checkpoint, cfg)
    else:
        spec = [(p, task_language(args.suite, i)) for p, i in zip(paths, tasks, strict=True)]
        print(f"[train] language-conditioned, {len(spec)} tasks:")
        for _, lang in spec:
            print(f"        - {lang}")
        meta = train(spec, args.checkpoint, cfg)
    print(f"✓ {args.checkpoint}  train_time {meta['train_time_s']}s  "
          f"final val_loss {meta['history'][-1]['val_loss']}")
    return 0


def cmd_eval(args) -> int:
    from rvc.agent.state_machine import AgentConfig, RobotAgent
    from rvc.agent.verifier import RewardVerifier
    from rvc.envs.libero_env import LiberoEnv
    from rvc.perception.detector import ColorDetector
    from rvc.policies.bc_libero import LiberoBCPolicy

    tasks = parse_tasks(args.task_index)
    if args.policy == "smolvla":
        from rvc.policies.smolvla_remote import SmolVLARemotePolicy

        policy = SmolVLARemotePolicy(url=args.url)
        resolution = 256  # SmolVLA's training resolution
        print(f"[eval] {policy.describe()}")
        print("       REAL VLA - first non-degraded model backend in this repo")
    else:
        policy = LiberoBCPolicy(args.checkpoint)
        resolution = 128  # BC's training resolution
        print(f"[eval] {policy.describe()}")
        print(f"       device={policy.device}")
        print("       DEGRADED: " + policy.degraded_reason)
    print(f"       suite={args.suite} tasks={tasks} episodes={args.episodes}/task "
          f"max_steps={args.max_steps}")
    override = ""
    if args.wrong_instruction_from is not None:
        override = task_language(args.suite, args.wrong_instruction_from)
        print(f"       ABLATION: every task hears the WRONG instruction: {override!r}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"{args.policy}-eval-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    per_task: dict[int, list] = {}
    t0 = time.time()
    for task_index in tasks:
        episodes = per_task.setdefault(task_index, [])
        for i in range(args.episodes):
            env = LiberoEnv(task_suite=args.suite, task_index=task_index,
                            init_state_index=i, resolution=resolution,
                            max_steps=args.max_steps, seed=i,
                            instruction_override=override)
            if hasattr(policy, "reset"):
                policy.reset()  # SmolVLA holds a 50-step action-chunk queue per episode
            agent = RobotAgent(
                env=env, policy=policy, verifier=RewardVerifier(), detector=ColorDetector(),
                config=AgentConfig(max_recoveries=0, max_total_steps=args.max_steps,
                                   mode="e2e", perceive_required=False),
                collect_frames=i < args.save_gifs,
            )
            gif_dir = str(out / f"t{task_index}-ep{i:02d}") if i < args.save_gifs else ""
            res = agent.run(run_dir=gif_dir)
            if gif_dir:
                from rvc.runners.demo_libero import write_artifacts

                write_artifacts(Path(gif_dir), agent, res,
                                extra={"policy": policy.describe(), "init_state": i,
                                       "task_index": task_index})
            lat = [r.latency_ms for r in agent.trace.records]
            ep = {"init_state": i, "success": res.success, "steps": res.steps,
                  "policy_ms_p50": round(float(np.median(lat)), 1) if lat else None,
                  "rejections": agent.validator.rejections, "clamps": agent.validator.clamps}
            episodes.append(ep)
            print(f"       t{task_index} ep {i:>2}  "
                  f"{'SUCCESS' if res.success else 'fail   '}  steps={res.steps:<4} "
                  f"policy p50 {ep['policy_ms_p50']} ms  clamps={ep['clamps']}", flush=True)
            env.close()

    if args.policy == "smolvla":
        prov_policy = policy.describe()
        prov_extra = {"chunk_forwards": len(policy.chunk_latencies_ms),
                      "chunk_ms_p50": round(float(np.median(policy.chunk_latencies_ms)), 1)
                      if policy.chunk_latencies_ms else None}
    else:
        lang = bool(policy.meta.get("lang_dim"))
        prov_policy = ("bc-libero language-conditioned (ResNet18x2+MLP + frozen MiniLM "
                       "instruction embedding - learned, uses language, NOT a VLA)"
                       if lang else
                       "bc-libero (ResNet18x2+MLP behaviour cloning, NOT a VLA, no language)")
        prov_extra = {"checkpoint": policy.checkpoint,
                      "checkpoint_meta": {k: v for k, v in policy.meta.items()
                                          if k not in ("history", "lang_emb")}}
    summary = {}
    for ti, eps in per_task.items():
        succ = sum(e["success"] for e in eps)
        summary[str(ti)] = {"successes": succ, "episodes": len(eps),
                            "success_rate": round(succ / max(1, len(eps)), 4)}
    n = sum(len(v) for v in per_task.values())
    succ = sum(s["successes"] for s in summary.values())
    report = {
        "provenance": {
            "policy": prov_policy,
            **prov_extra,
            "env": f"LIBERO {args.suite} tasks {tasks}, {resolution}x{resolution}, "
                   f"init states 0..{args.episodes - 1} per task, max_steps {args.max_steps}",
            "runtime": "RobotAgent e2e mode + ActionValidator (no recovery, no planner)",
            "instruction_override": override or None,
        },
        "per_task": summary,
        "episodes": n,
        "successes": succ,
        "success_rate": round(succ / max(1, n), 4),
        "wall_time_s": round(time.time() - t0, 1),
        "per_episode": {str(k): v for k, v in per_task.items()},
    }
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    print()
    for ti, s in summary.items():
        print(f"  task {ti}: {s['successes']}/{s['episodes']} = {s['success_rate']:.0%}")
    print(f"  overall : {succ}/{n} = {report['success_rate']:.0%}   "
          f"(wall {report['wall_time_s']}s)\n  -> {out}/eval.json\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--suite", default="libero_spatial")
        s.add_argument("--task-index", default="0", help="single index or comma list '0,1,2'")
        s.add_argument("--checkpoint", default=str(REPO_ROOT / "models" / "bc-libero-spatial-0.pt"))
        if name == "train":
            s.add_argument("--epochs", type=int, default=30)
            s.add_argument("--batch", type=int, default=64)
            s.add_argument("--lr", type=float, default=3e-4)
            s.add_argument("--scratch", action="store_true", help="no ImageNet init")
            s.add_argument("--seed", type=int, default=0)
        if name == "eval":
            s.add_argument("--policy", default="bc", choices=("bc", "smolvla"))
            s.add_argument("--url", default="http://127.0.0.1:8100")
            s.add_argument("--episodes", type=int, default=20)
            s.add_argument("--max-steps", type=int, default=220)
            s.add_argument("--save-gifs", type=int, default=2)
            s.add_argument("--out", default=str(REPO_ROOT / "runs"))
            s.add_argument("--wrong-instruction-from", type=int, default=None,
                           help="ablation: every evaluated task hears THIS task's "
                                "instruction instead of its own")
    args = ap.parse_args(argv)
    return {"data": cmd_data, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
