"""Bake a LoRA adapter (export_lora.py output) into a full copy of the base weights, so vLLM generates at plain-model
speed (an unmerged adapter halves throughput on the 2070).

python merge_lora.py loras/x merged/x [--dtype float16|bfloat16]
Runs on tranny with the vLLM env (base model from its HF cache). W' = W + (lora_alpha / r) B A, summed in fp32 and
rounded once to --dtype; pick the dtype the GPU will run (2070: float16, 3070: bfloat16), since rounding the merged
weight to bf16 can swallow part of a small delta. Prints how much of the delta that rounding lost.
"""
import argparse, json, shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lora")
    ap.add_argument("out")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    src = Path(snapshot_download(BASE, local_files_only=True))
    lora = load_file(str(Path(a.lora) / "adapter_model.safetensors"))
    cfg = json.loads((Path(a.lora) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy(f, out / f.name)
    mcfg = json.loads((out / "config.json").read_text())
    mcfg["torch_dtype"] = a.dtype
    (out / "config.json").write_text(json.dumps(mcfg, indent=2))

    used, err2, d2 = set(), 0.0, 0.0
    for wf in sorted(src.glob("*.safetensors")):
        sd = load_file(str(wf))
        for k in list(sd):
            n = k[: -len(".weight")] if k.endswith(".weight") else None
            ka = f"base_model.model.{n}.lora_A.weight"
            if n and ka in lora:
                delta = scale * (lora[f"base_model.model.{n}.lora_B.weight"].float() @ lora[ka].float())
                exact = sd[k].float() + delta
                sd[k] = exact.to(dtype)
                err2 += (sd[k].float() - exact).pow(2).sum().item()
                d2 += delta.pow(2).sum().item()
                used.add(ka)
            else:
                sd[k] = sd[k].to(dtype)
        save_file(sd, str(out / wf.name), metadata={"format": "pt"})
    missing = {k for k in lora if k.endswith("lora_A.weight")} - used
    if missing:
        raise SystemExit(f"{len(missing)} adapter modules not found in the base, e.g. {sorted(missing)[0]}")
    print(f"merged {len(used)} modules -> {out} ({a.dtype}); rounding error = {(err2 / d2) ** 0.5:.1%} of the delta")


if __name__ == "__main__":
    main()
