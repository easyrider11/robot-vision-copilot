"""Language-conditioned BC: unit tests that need no downloads and no LIBERO.

The MiniLM encoder itself is exercised only in the real training run; here the
384-dim embeddings are stubbed, which is exactly the point - the policy and
net must treat language as an opaque conditioning vector.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rvc.policies.bc_libero import LANG_DIM, _build_net  # noqa: E402


def test_lang_dim_zero_is_the_legacy_architecture():
    old = _build_net(pretrained=False)
    new = _build_net(pretrained=False, lang_dim=0)
    assert {k: v.shape for k, v in old.state_dict().items()} \
        == {k: v.shape for k, v in new.state_dict().items()}
    assert new.lang is None


def test_conditioned_net_forward_and_gradient_through_language():
    net = _build_net(pretrained=False, lang_dim=LANG_DIM)
    a = torch.zeros(2, 128, 128, 3, dtype=torch.uint8)
    w = torch.zeros(2, 128, 128, 3, dtype=torch.uint8)
    p = torch.zeros(2, 5)
    lang = torch.randn(2, LANG_DIM, requires_grad=True)
    out = net(a, w, p, lang)
    assert out.shape == (2, 7)
    out.sum().backward()
    assert lang.grad is not None and float(lang.grad.abs().sum()) > 0, \
        "language input must be load-bearing, not ignored"


def test_different_instruction_changes_the_action():
    torch.manual_seed(0)
    net = _build_net(pretrained=False, lang_dim=LANG_DIM).eval()
    a = torch.zeros(1, 128, 128, 3, dtype=torch.uint8)
    w = torch.zeros(1, 128, 128, 3, dtype=torch.uint8)
    p = torch.zeros(1, 5)
    with torch.no_grad():
        o1 = net(a, w, p, torch.randn(1, LANG_DIM))
        o2 = net(a, w, p, torch.randn(1, LANG_DIM))
    assert not torch.allclose(o1, o2), "same scene, different instruction -> same action?"


def test_policy_looks_up_instruction_from_checkpoint_table(tmp_path):
    from rvc.policies.bc_libero import LiberoBCPolicy
    from rvc.types import Observation

    net = _build_net(pretrained=False, lang_dim=LANG_DIM)
    emb = np.random.default_rng(0).normal(size=(2, LANG_DIM)).astype(np.float32)
    meta = {
        "kind": "bc-libero", "config": {"epochs": 1},
        "proprio_mean": [0.0] * 5, "proprio_std": [1.0] * 5,
        "history": [{"val_loss": 0.0}], "lang_dim": LANG_DIM,
        "instructions": ["pick up the left bowl", "pick up the right bowl"],
        "lang_emb": emb.tolist(),
    }
    ckpt = tmp_path / "lang.pt"
    torch.save({"state_dict": net.state_dict(), "meta": meta}, ckpt)

    pol = LiberoBCPolicy(ckpt, device="cpu")
    assert "USES language" in pol.degraded_reason and pol.degraded
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    def act(instr):
        obs = Observation(image=img, wrist_image=img, instruction=instr, step=0,
                          proprio=np.zeros(3, dtype=np.float32),
                          privileged={"gripper_qpos": [0.0, 0.0]})
        return pol.predict(obs).vector

    a0, a1 = act(meta["instructions"][0]), act(meta["instructions"][1])
    assert not np.allclose(a0[:6], a1[:6]), "conditioning must reach the output"
    # unseen instruction without transformers -> honest failure, never silent
    import sys
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
            has_tf = True
        except ImportError:
            has_tf = False
    else:
        has_tf = True
    if not has_tf:
        with pytest.raises(RuntimeError, match="not in the checkpoint's table"):
            act("a sentence it never saw")
