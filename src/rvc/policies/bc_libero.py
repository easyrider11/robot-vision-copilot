"""Behaviour-cloning baseline for LIBERO - a learned policy that is NOT a VLA.

WHY THIS EXISTS
---------------
The repo's action-model slot was designed for OpenVLA, which this machine
cannot run. Rather than leave LIBERO as an empty exam hall, this module trains
the classic small baseline - ResNet18 encoders + MLP head, behaviour-cloned
from the 50 human demos LIBERO ships per task - locally on Apple MPS, and
evaluates it through the same agent runtime. It answers a different question
than OpenVLA would ("how far does 50 demos + a small CNN get you on one task?")
and it is labelled as such: `degraded=True`, reason "BC baseline, not a VLA".

DATA CONVENTIONS (all measured, see docs/08-bc-baseline.md)
------------------------------------------------------------
    hdf5 images are stored upside-down relative to LiberoEnv observations
      -> flip [::-1, ::-1]  (corr 0.968 after flip vs -0.236 before)
    hdf5 actions: 6 OSC deltas in [-1, 1] + gripper in {-1 open, +1 close}
      -> contract gripper = (1 - g) / 2  (OpenVLA convention: 1 open, 0 closed)
    proprio: ee_pos (3) + gripper_states (2)  == robot0_eef_pos + gripper_qpos

Everything here is torch-only on purpose (no ultralytics, no transformers):
`pip install -e ".[bc]"`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rvc.types import Action, Gripper, Observation

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = REPO_ROOT / "models" / "bc-libero-spatial-0.pt"
IMG = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class BCConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    shift_pad: int = 6  # DrQ-style random-shift augmentation
    val_demos: int = 5  # last N demos held out for validation loss
    pretrained: bool = True  # ImageNet-initialised ResNet18 (downloads 45 MB once)
    seed: int = 0


# --- data -------------------------------------------------------------------


def load_demos(hdf5_path: str | Path) -> dict[str, np.ndarray]:
    """Flatten all demos into arrays already in the *contract* convention."""
    import h5py

    agent, wrist, proprio, actions, episode = [], [], [], [], []
    with h5py.File(str(hdf5_path), "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for ep, name in enumerate(demos):
            d = f["data"][name]
            a = np.asarray(d["actions"], dtype=np.float32)
            g = a[:, 6]
            a = a.copy()
            a[:, 6] = (1.0 - g) / 2.0  # LIBERO {-1 open, +1 close} -> contract {1 open, 0 closed}
            agent.append(np.asarray(d["obs"]["agentview_rgb"])[:, ::-1, ::-1])
            wrist.append(np.asarray(d["obs"]["eye_in_hand_rgb"])[:, ::-1, ::-1])
            proprio.append(np.concatenate(
                [np.asarray(d["obs"]["ee_pos"]), np.asarray(d["obs"]["gripper_states"])], axis=1
            ).astype(np.float32))
            actions.append(a)
            episode.append(np.full(len(a), ep, dtype=np.int32))
    return {
        "agent": np.ascontiguousarray(np.concatenate(agent)),
        "wrist": np.ascontiguousarray(np.concatenate(wrist)),
        "proprio": np.concatenate(proprio),
        "actions": np.concatenate(actions),
        "episode": np.concatenate(episode),
        "n_demos": len(demos),
    }


# --- model ------------------------------------------------------------------


def _build_net(pretrained: bool, proprio_dim: int = 5):
    import torch
    import torch.nn as nn
    import torchvision

    def trunk():
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            m = torchvision.models.resnet18(weights=weights)
        except Exception:  # offline - fall back to random init, and say so
            print("      (ImageNet weights unavailable offline - training from scratch)")
            m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Identity()
        return m

    class BCNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc_agent = trunk()
            self.enc_wrist = trunk()
            self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(512 + 512 + 64, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 7),
            )
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def _prep(self, x):  # uint8 NHWC -> normalised float NCHW
            x = x.permute(0, 3, 1, 2).float() / 255.0
            return (x - self.mean) / self.std

        def forward(self, agent, wrist, proprio):
            f = torch.cat([
                self.enc_agent(self._prep(agent)),
                self.enc_wrist(self._prep(wrist)),
                self.proprio(proprio),
            ], dim=1)
            return self.head(f)  # [:, :6] deltas, [:, 6] gripper logit (1 = open)

    return BCNet()


def _random_shift(x, pad: int):
    """Per-sample random crop after replicate-padding; x is uint8 NHWC."""
    import torch
    import torch.nn.functional as F

    n, h, w, c = x.shape
    xp = F.pad(x.permute(0, 3, 1, 2).float(), (pad, pad, pad, pad), mode="replicate")
    out = torch.empty((n, c, h, w), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    for i in range(n):
        out[i] = xp[i, :, oy[i] : oy[i] + h, ox[i] : ox[i] + w]
    return out.permute(0, 2, 3, 1).to(torch.uint8)


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- training ---------------------------------------------------------------


def train(hdf5_path: str | Path, out_path: str | Path = DEFAULT_CKPT, cfg: BCConfig | None = None,
          log=print) -> dict:
    import torch
    import torch.nn.functional as F

    cfg = cfg or BCConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device()
    t0 = time.time()

    data = load_demos(hdf5_path)
    n_tr_demos = data["n_demos"] - cfg.val_demos
    tr = data["episode"] < n_tr_demos
    va = ~tr
    log(f"      demos={data['n_demos']} steps={len(tr)} train={int(tr.sum())} "
        f"val={int(va.sum())} device={device}")

    pm, ps = data["proprio"][tr].mean(0), data["proprio"][tr].std(0) + 1e-6
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    A, W = to_t(data["agent"]).to(device), to_t(data["wrist"]).to(device)
    P = to_t((data["proprio"] - pm) / ps).to(device)
    Y = to_t(data["actions"]).to(device)
    idx_tr = torch.from_numpy(np.nonzero(tr)[0]).to(device)
    idx_va = torch.from_numpy(np.nonzero(va)[0]).to(device)

    net = _build_net(cfg.pretrained).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(idx_tr) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)

    def loss_fn(pred, y):
        # .contiguous(): MPS smooth_l1 views its inputs; column slices are strided
        l_xyz = F.smooth_l1_loss(pred[:, :6].contiguous(), y[:, :6].contiguous())
        l_grip = F.binary_cross_entropy_with_logits(pred[:, 6].contiguous(), y[:, 6].contiguous())
        return l_xyz + 0.5 * l_grip, l_xyz, l_grip

    history = []
    for ep in range(cfg.epochs):
        net.train()
        perm = idx_tr[torch.randperm(len(idx_tr), device=device)]
        tot = n = 0.0
        for b in range(steps_per_epoch):
            i = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            a = _random_shift(A[i], cfg.shift_pad) if cfg.shift_pad else A[i]
            w = _random_shift(W[i], cfg.shift_pad) if cfg.shift_pad else W[i]
            pred = net(a, w, P[i])
            loss, _, _ = loss_fn(pred, Y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(i)
            n += len(i)
        net.eval()
        with torch.no_grad():
            vl = vg = vn = 0.0
            for b in range(0, len(idx_va), 256):
                i = idx_va[b : b + 256]
                pred = net(A[i], W[i], P[i])
                loss, _, _ = loss_fn(pred, Y[i])
                vl += float(loss) * len(i)
                vg += float(((pred[:, 6] > 0).float() == Y[i, 6]).float().mean()) * len(i)
                vn += len(i)
        rec = {"epoch": ep + 1, "train_loss": round(tot / n, 4),
               "val_loss": round(vl / vn, 4), "val_gripper_acc": round(vg / vn, 4),
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        log(f"      epoch {ep + 1:>3}/{cfg.epochs}  train {rec['train_loss']:.4f}  "
            f"val {rec['val_loss']:.4f}  grip-acc {rec['val_gripper_acc']:.3f}  "
            f"[{rec['elapsed_s']:.0f}s]")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "bc-libero",
        "hdf5": str(hdf5_path),
        "config": asdict(cfg),
        "proprio_mean": pm.tolist(),
        "proprio_std": ps.tolist(),
        "history": history,
        "train_time_s": round(time.time() - t0, 1),
        "device": device,
        "provenance": "Behaviour cloning on LIBERO human demos - a learned policy, NOT a VLA. "
                      "No language conditioning; single task.",
    }
    torch.save({"state_dict": net.state_dict(), "meta": meta}, out_path)
    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    return meta


# --- policy -----------------------------------------------------------------


class LiberoBCPolicy:
    """Policy-protocol wrapper around a trained BC checkpoint."""

    name = "bc-libero"
    degraded = True
    degraded_reason = (
        "Small behaviour-cloning baseline (ResNet18 x2 + MLP, 50 demos, one task) - "
        "a learned policy but NOT a vision-language-action model; ignores language."
    )

    def __init__(self, checkpoint: str | Path = DEFAULT_CKPT, device: str | None = None) -> None:
        import torch

        from rvc.policies.base import PolicyUnavailable

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise PolicyUnavailable(
                f"BC checkpoint not found at {ckpt_path} - run `make bc-train` first"
            )
        self.device = device or pick_device()
        blob = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.meta = blob["meta"]
        self.net = _build_net(pretrained=False).to(self.device)
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self.pm = np.asarray(self.meta["proprio_mean"], dtype=np.float32)
        self.ps = np.asarray(self.meta["proprio_std"], dtype=np.float32)
        self.last_latency_ms = 0.0
        self.checkpoint = str(ckpt_path)

    def describe(self) -> str:
        h = self.meta.get("history") or [{}]
        return (f"{self.name} ({Path(self.checkpoint).name}, "
                f"{self.meta['config']['epochs']} epochs, val_loss {h[-1].get('val_loss', '?')})")

    @staticmethod
    def _to128(img: np.ndarray | None) -> np.ndarray:
        if img is None:
            return np.zeros((IMG, IMG, 3), dtype=np.uint8)
        if img.shape[0] != IMG or img.shape[1] != IMG:
            img = np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))
        return np.ascontiguousarray(img, dtype=np.uint8)

    def predict(self, obs: Observation) -> Action:
        import torch

        t0 = time.perf_counter()
        ee = np.asarray(obs.proprio if obs.proprio is not None else np.zeros(3), dtype=np.float32)
        gq = np.asarray(obs.privileged.get("gripper_qpos", [0.0, 0.0]), dtype=np.float32)
        proprio = (np.concatenate([ee[:3], gq[:2]]) - self.pm) / self.ps
        with torch.no_grad():
            a = torch.from_numpy(self._to128(obs.image)[None]).to(self.device)
            w = torch.from_numpy(self._to128(obs.wrist_image)[None]).to(self.device)
            p = torch.from_numpy(proprio[None]).to(self.device)
            out = self.net(a, w, p)[0].float().cpu().numpy()
        vec = np.zeros(7, dtype=np.float32)
        vec[:6] = np.clip(out[:6], -1.0, 1.0)
        vec[6] = Gripper.OPEN if out[6] > 0 else Gripper.CLOSED
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return Action(vec)
