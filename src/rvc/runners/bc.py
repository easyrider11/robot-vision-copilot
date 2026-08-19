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

    path = task_hdf5(args.suite, args.task_index)
    if path.exists():
        print(f"✓ already present: {path} ({path.stat().st_size / 1e6:.0f} MB)")
        return 0
    print(f"[data] downloading {path.name} from {HF_REPO} ...")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                      allow_patterns=[f"{args.suite}/{path.name}"], local_dir=str(DATA_DIR))
    print(f"✓ {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


def cmd_train(args) -> int:
    from rvc.policies.bc_libero import BCConfig, train

    path = task_hdf5(args.suite, args.task_index)
    if not path.exists():
        print(f"✗ no demos at {path} - run `make bc-data` first")
        return 1
    cfg = BCConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                   pretrained=not args.scratch, seed=args.seed)
    print(f"[train] {path.name}\n        {cfg}")
    meta = train(path, args.checkpoint, cfg)
    print(f"✓ {args.checkpoint}  train_time {meta['train_time_s']}s  "
          f"final val_loss {meta['history'][-1]['val_loss']}")
    return 0


def cmd_eval(args) -> int:
    from rvc.agent.state_machine import AgentConfig, RobotAgent
    from rvc.agent.verifier import RewardVerifier
    from rvc.envs.libero_env import LiberoEnv
    from rvc.perception.detector import ColorDetector
    from rvc.policies.bc_libero import LiberoBCPolicy

    policy = LiberoBCPolicy(args.checkpoint)
    print(f"[eval] {policy.describe()}")
    print(f"       suite={args.suite} task={args.task_index} episodes={args.episodes} "
          f"max_steps={args.max_steps} device={policy.device}")
    print("       DEGRADED: " + policy.degraded_reason)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"bc-eval-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    episodes = []
    t0 = time.time()
    for i in range(args.episodes):
        env = LiberoEnv(task_suite=args.suite, task_index=args.task_index,
                        init_state_index=i, resolution=128, max_steps=args.max_steps, seed=i)
        agent = RobotAgent(
            env=env, policy=policy, verifier=RewardVerifier(), detector=ColorDetector(),
            config=AgentConfig(max_recoveries=0, max_total_steps=args.max_steps,
                               mode="e2e", perceive_required=False),
            collect_frames=i < args.save_gifs,
        )
        res = agent.run(run_dir=str(out / f"ep{i:02d}") if i < args.save_gifs else "")
        if i < args.save_gifs:
            from rvc.runners.demo_libero import write_artifacts

            write_artifacts(out / f"ep{i:02d}", agent, res,
                            extra={"policy": policy.describe(), "init_state": i})
        lat = [r.latency_ms for r in agent.trace.records]
        ep = {"init_state": i, "success": res.success, "steps": res.steps,
              "policy_ms_p50": round(float(np.median(lat)), 1) if lat else None,
              "rejections": agent.validator.rejections, "clamps": agent.validator.clamps}
        episodes.append(ep)
        print(f"       ep {i:>2}  {'SUCCESS' if res.success else 'fail   '}  steps={res.steps:<4} "
              f"policy p50 {ep['policy_ms_p50']} ms  clamps={ep['clamps']}")
        env.close()

    n = len(episodes)
    succ = sum(e["success"] for e in episodes)
    report = {
        "provenance": {
            "policy": "bc-libero (ResNet18x2+MLP behaviour cloning, NOT a VLA, no language)",
            "checkpoint": policy.checkpoint,
            "checkpoint_meta": {k: v for k, v in policy.meta.items() if k != "history"},
            "env": f"LIBERO {args.suite} task {args.task_index}, 128x128, "
                   f"init states 0..{n - 1}, max_steps {args.max_steps}",
            "runtime": "RobotAgent e2e mode + ActionValidator (no recovery, no planner)",
        },
        "episodes": n,
        "successes": succ,
        "success_rate": round(succ / max(1, n), 4),
        "mean_steps": round(float(np.mean([e["steps"] for e in episodes])), 1),
        "wall_time_s": round(time.time() - t0, 1),
        "per_episode": episodes,
    }
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    print(f"\n  success_rate : {succ}/{n} = {report['success_rate']:.0%}   "
          f"(wall {report['wall_time_s']}s)\n  -> {out}/eval.json\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--suite", default="libero_spatial")
        s.add_argument("--task-index", type=int, default=0)
        s.add_argument("--checkpoint", default=str(REPO_ROOT / "models" / "bc-libero-spatial-0.pt"))
        if name == "train":
            s.add_argument("--epochs", type=int, default=30)
            s.add_argument("--batch", type=int, default=64)
            s.add_argument("--lr", type=float, default=3e-4)
            s.add_argument("--scratch", action="store_true", help="no ImageNet init")
            s.add_argument("--seed", type=int, default=0)
        if name == "eval":
            s.add_argument("--episodes", type=int, default=20)
            s.add_argument("--max-steps", type=int, default=220)
            s.add_argument("--save-gifs", type=int, default=2)
            s.add_argument("--out", default=str(REPO_ROOT / "runs"))
    args = ap.parse_args(argv)
    return {"data": cmd_data, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
