"""vLLM generation worker (runs on tranny in /mnt/wdblue/claude-breeding/vllm-env), same output as run.py gen.

python vgen.py --problems data/p.jsonl --out runs/x.jsonl [--lora loras/x] [--k 4 --temp 0.8 --seed 0]
               [--dtype bfloat16|float16] [--max_new 640]
Pick the GPU with CUDA_VISIBLE_DEVICES. The adapter (export_lora.py output) is applied unmerged by vLLM.
"""
import argparse, json, time
from pathlib import Path

import fields as F

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def run_job(llm, tok, lreq, problems, out, k=1, temp=0.0, seed=0, max_new=640):
    from vllm import SamplingParams
    probs = [json.loads(l) for l in open(problems) if l.strip()]
    jobs = [(i, kk) for i in range(len(probs)) for kk in range(k)]
    prompts = [tok.apply_chat_template([{"role": "user", "content": probs[i]["q"]}], tokenize=False,
                                       add_generation_prompt=True) for i, _ in jobs]
    params = [SamplingParams(max_tokens=max_new, temperature=temp, top_p=1.0, top_k=-1,
                             seed=(seed * 1_000_003 + j) if temp > 0 else None) for j in range(len(jobs))]
    t0 = time.time()
    outs = llm.generate(prompts, params, lora_request=lreq)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for (i, kk), o in zip(jobs, outs):
            p, text = probs[i], o.outputs[0].text
            ok, readable = F.check(p, text)
            fh.write(json.dumps({**p, "k": kk, "text": text, "ok": ok, "readable": readable,
                                 "ans": F.answer_of(p, text)}) + "\n")
    print(f"[vgen] {out}: {len(jobs)} answers, {ntok} tokens in {dt:.0f}s = {ntok / dt:.0f} tok/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems")
    ap.add_argument("--out")
    ap.add_argument("--jobs", default="", help="JSON list of {problems, out, k?, temp?, seed?, max_new?}: "
                                               "several runs on one engine start")
    ap.add_argument("--model", default=BASE, help="base model, or a merge_lora.py output (then no --lora)")
    ap.add_argument("--lora", default="")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=640)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem", type=float, default=0.85)
    a = ap.parse_args()

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    if a.jobs:
        todo = json.loads(Path(a.jobs).read_text())
    else:
        todo = [dict(problems=a.problems, out=a.out, k=a.k, temp=a.temp, seed=a.seed, max_new=a.max_new)]
    lora_rank = json.loads((Path(a.lora) / "adapter_config.json").read_text())["r"] if a.lora else 0
    llm = LLM(model=a.model, dtype=a.dtype, gpu_memory_utilization=a.mem, max_model_len=2048, seed=a.seed,
              enable_lora=bool(a.lora), max_lora_rank=max(lora_rank, 8), max_loras=1)
    tok = llm.get_tokenizer()
    lreq = LoRARequest("genome", 1, str(Path(a.lora).resolve())) if a.lora else None
    for j in todo:
        run_job(llm, tok, lreq, **j)


if __name__ == "__main__":
    main()
