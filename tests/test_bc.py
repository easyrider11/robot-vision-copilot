"""Behaviour-cloning baseline: data conventions and policy contract.

Heavy parts (torch/h5py/LIBERO, trained weights) skip themselves when absent;
the conversion math always runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rvc.types import Gripper, Observation

CKPT = Path(__file__).resolve().parents[1] / "models" / "bc-libero-spatial-0.pt"


def test_libero_gripper_to_contract_math():
    # LIBERO hdf5: -1 = open, +1 = close  ->  contract: 1 = open, 0 = closed
    g = np.array([-1.0, 1.0, -1.0])
    contract = (1.0 - g) / 2.0
    assert contract.tolist() == [1.0, 0.0, 1.0]
    assert contract[0] == Gripper.OPEN and contract[1] == Gripper.CLOSED


def test_load_demos_converts_and_flips(tmp_path):
    h5py = pytest.importorskip("h5py")
    from rvc.policies.bc_libero import load_demos

    path = tmp_path / "demo.hdf5"
    T = 4
    img = np.zeros((T, 128, 128, 3), dtype=np.uint8)
    img[:, 0, 0, 0] = 255  # a marker pixel in the top-left corner
    with h5py.File(path, "w") as f:
        for k in range(2):
            d = f.create_group(f"data/demo_{k}")
            acts = np.zeros((T, 7), dtype=np.float32)
            acts[:, 6] = [-1, -1, 1, 1]
            d.create_dataset("actions", data=acts)
            d.create_dataset("obs/agentview_rgb", data=img)
            d.create_dataset("obs/eye_in_hand_rgb", data=img)
            d.create_dataset("obs/ee_pos", data=np.ones((T, 3), dtype=np.float32))
            d.create_dataset("obs/gripper_states", data=np.zeros((T, 2), dtype=np.float32))
    data = load_demos(path)
    assert data["n_demos"] == 2 and data["actions"].shape == (2 * T, 7)
    assert data["actions"][:, 6].tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    # the marker pixel moved to the bottom-right: frames are flipped [::-1, ::-1]
    assert data["agent"][0, -1, -1, 0] == 255 and data["agent"][0, 0, 0, 0] == 0
    assert data["proprio"].shape == (2 * T, 5)


@pytest.mark.skipif(not CKPT.exists(), reason="run `make bc-train` first")
def test_trained_policy_respects_the_contract():
    pytest.importorskip("torch")
    from rvc.policies.bc_libero import LiberoBCPolicy

    pol = LiberoBCPolicy(CKPT)
    assert pol.degraded and "NOT a vision-language-action" in pol.degraded_reason
    obs = Observation(
        image=np.zeros((256, 256, 3), dtype=np.uint8),  # resized to 128 internally
        wrist_image=np.zeros((128, 128, 3), dtype=np.uint8),
        instruction="pick up the black bowl",
        proprio=np.zeros(3, dtype=np.float32),
        privileged={"gripper_qpos": [0.04, -0.04]},
    )
    a = pol.predict(obs)
    assert a.vector.shape == (7,)
    assert np.all(np.abs(a.vector[:6]) <= 1.0)
    assert a.gripper in (Gripper.OPEN, Gripper.CLOSED)
