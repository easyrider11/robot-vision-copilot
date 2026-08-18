"""Real OpenVLA-7B inference, in-process.

REQUIREMENTS (none of which the audited MacBook Air M3 meets - see
docs/00-environment-audit.md):

  * CUDA GPU, >= 16 GB VRAM for bf16 (or >= 8 GB with 4-bit bitsandbytes,
    which is CUDA-only and has no MPS backend)
  * ~15.1 GB free disk for `openvla/openvla-7b`
  * `uv pip install -e ".[vla]"`

SECURITY NOTE: OpenVLA ships custom modelling code, so loading it requires
`trust_remote_code=True`, i.e. executing Python downloaded from the Hub. That is
the upstream-documented path, but it is a real trust decision - pin a revision
in production. `revision` is exposed as a constructor argument for that reason.

CHECKPOINT / unnorm_key PAIRING - the single most common source of "the robot
twitches but never moves":

    openvla/openvla-7b                          -> unnorm_key="bridge_orig"
    openvla/openvla-7b-finetuned-libero-spatial -> unnorm_key="libero_spatial"
    openvla/openvla-7b-finetuned-libero-object  -> unnorm_key="libero_object"
    openvla/openvla-7b-finetuned-libero-goal    -> unnorm_key="libero_goal"
    openvla/openvla-7b-finetuned-libero-10      -> unnorm_key="libero_10"

The un-normalisation statistics are baked into the checkpoint; asking for a key
it was not trained with either raises or silently rescales the actions.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
from PIL import Image

from rvc.policies.base import PolicyUnavailable
from rvc.types import Action, Observation

DEFAULT_MODEL = "openvla/openvla-7b"
WEIGHTS_BYTES = 15_082_600_824  # measured from the Hub, 3 safetensors shards

# OpenVLA's exact prompt template. Reproduced verbatim; changing the wording
# moves the model off-distribution.
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def probe(min_free_disk_bytes: int = WEIGHTS_BYTES + 8 * 1024**3) -> tuple[bool, str]:
    """Can real OpenVLA run here? Returns (ok, reason). Never raises, never downloads."""
    try:
        import torch
    except Exception as exc:
        return False, f"torch not installed ({type(exc).__name__}). Run: uv pip install -e '.[vla]'"

    # Hardware verdict BEFORE the missing-package verdict: on a machine with no
    # CUDA device, "pip install transformers" is not the fix, and saying so
    # first would send the reader down a dead end.
    if not torch.cuda.is_available():
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return False, (
                "Only Apple MPS is available. A 7B VLA in bf16 needs ~15 GB of weights "
                "resident; MPS shares system RAM and bitsandbytes 4-bit is CUDA-only. "
                "Use --backend openvla-remote against a cloud GPU instead."
            )
        return False, "No CUDA device found. OpenVLA-7B needs a CUDA GPU with >= 16 GB VRAM."

    try:
        import transformers  # noqa: F401
    except Exception as exc:
        return False, (
            f"transformers not installed ({type(exc).__name__}). "
            "Run: uv pip install -e '.[vla]'"
        )

    free_vram = torch.cuda.mem_get_info()[0]
    if free_vram < 15 * 1024**3:
        return False, (
            f"CUDA device has only {free_vram / 1024**3:.1f} GB free VRAM; bf16 OpenVLA-7B "
            f"needs ~15 GB. Try load_in_4bit=True (needs bitsandbytes)."
        )

    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    free_disk = shutil.disk_usage(os.path.dirname(cache) or "/").free
    if free_disk < min_free_disk_bytes:
        return False, (
            f"Only {free_disk / 1024**3:.1f} GB free on the HF cache volume; the weights "
            f"alone are {WEIGHTS_BYTES / 1024**3:.1f} GB."
        )
    return True, ""


class OpenVLALocalPolicy:
    """Loads `openvla/openvla-7b` with transformers and calls `predict_action`."""

    degraded = False
    degraded_reason = ""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        unnorm_key: str = "bridge_orig",
        device: str = "cuda:0",
        load_in_4bit: bool = False,
        revision: str | None = None,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        ok, reason = probe()
        if not ok:
            raise PolicyUnavailable(reason)

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.name = f"openvla:{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.unnorm_key = unnorm_key
        self.device = device
        self._torch = torch

        kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if revision:
            kwargs["revision"] = revision
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = attn_implementation
        except Exception:
            # flash-attn is a hard build dependency upstream but not strictly
            # required; SDPA is slower yet numerically fine.
            kwargs["attn_implementation"] = "sdpa"

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            kwargs.pop("torch_dtype", None)

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, **({"revision": revision} if revision else {})
        )
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()

    def describe(self) -> str:
        return f"{self.name} on {self.device}, unnorm_key={self.unnorm_key}"

    def predict(self, obs: Observation) -> Action:
        torch = self._torch
        img = Image.fromarray(np.asarray(obs.image, dtype=np.uint8)).convert("RGB")
        prompt = PROMPT_TEMPLATE.format(instruction=obs.instruction.lower().rstrip("."))
        inputs = self.processor(prompt, img).to(self.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            raw = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return Action(np.asarray(raw, dtype=np.float32).reshape(-1))
