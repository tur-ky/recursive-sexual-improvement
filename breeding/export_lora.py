"""Turn a spec (base + scaled deltas, with layer / module / rank filters) into one PEFT LoRA adapter for vLLM.

python export_lora.py specs/x.json loras/x
Deltas stack by concatenating their factors (B1 A1 + B2 A2 = [B1 B2] [A1; A2]), so the result is exact, not an
approximation. Each module is zero-padded to one rank, rounded up to a size vLLM accepts, and lora_alpha = r so the
adapter's scale is 1 (each delta already has its own scale folded into B).
"""
import json, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rsi"))
from rsi import BASE, layer_of, truncate  # noqa: E402

VLLM_RANKS = [8, 16, 32, 64, 128, 256, 320, 512]


def factors_for(spec):
    """{module: (A [R, in], B [out, R])} exactly as rsi.apply_spec would add them."""
    acc = {}
    for d in spec.get("deltas", []):
        sd = load_file(str(HERE / d["path"] / "delta.safetensors"))
        alpha = float(d.get("alpha", 1.0))
        layers = set(d["layers"]) if d.get("layers") is not None else None
        kinds = set(d["modules"]) if d.get("modules") else None
        for n in sorted({k.rsplit(".", 1)[0] for k in sd}):
            if layers is not None and layer_of(n) not in layers:
                continue
            if kinds is not None and n.split(".")[-1] not in kinds:
                continue
            if f"{n}.bias" in sd:
                raise ValueError(f"{n}: bias deltas cannot be expressed as a LoRA")
            A, B = sd[f"{n}.A"].float(), sd[f"{n}.B"].float()
            if d.get("rank"):
                A, B = truncate(A, B, int(d["rank"]))
            B = B * alpha
            if n in acc:
                A0, B0 = acc[n]
                A, B = torch.cat([A0, A], 0), torch.cat([B0, B], 1)
            acc[n] = (A, B)
    return acc


def export(spec_path, out_dir):
    spec = json.loads(Path(spec_path).read_text())
    acc = factors_for(spec)
    if not acc:
        return None  # the base model itself: no adapter
    need = max(A.shape[0] for A, _ in acc.values())
    r = next(x for x in VLLM_RANKS if x >= need)
    sd = {}
    for n, (A, B) in acc.items():
        pad = r - A.shape[0]
        A = torch.cat([A, A.new_zeros(pad, A.shape[1])], 0)
        B = torch.cat([B, B.new_zeros(B.shape[0], pad)], 1)
        sd[f"base_model.model.{n}.lora_A.weight"] = A.to(torch.bfloat16).contiguous()
        sd[f"base_model.model.{n}.lora_B.weight"] = B.to(torch.bfloat16).contiguous()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": BASE,
        "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
        "target_modules": sorted({n.split(".")[-1] for n in acc}),
    }, indent=1))
    return r


if __name__ == "__main__":
    r = export(sys.argv[1], sys.argv[2])
    print("base model, no adapter" if r is None else f"rank {r} -> {sys.argv[2]}")
