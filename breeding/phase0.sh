#!/usr/bin/env bash
# Phase 0: base accuracy for every field, then a verifier test in one new field (long division).
cd "$(dirname "$0")"
P=../rsi/.venv/Scripts/python
busy() {
  powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'rsi.py|dist.py' }).Count" | tr -d '\r'
}
rbusy() { ssh tranny 'pgrep -fc "[r]si[.]py" || true' | tr -d '\r'; }  # bracket stops pgrep matching its own shell
echo "waiting for the 19 November run to free the GPUs"
quiet=0
while [ $quiet -lt 3 ]; do
  if [ "$(busy)" = "0" ] && [ "$(rbusy)" = "0" ]; then quiet=$((quiet + 1)); else quiet=0; fi
  sleep 20
done
date; echo "GPUs free"

# 1. base accuracy, 9 fields x 4 levels x 40, greedy
$P bdist.py gen --spec specs/base.json --problems data/phase0_base.jsonl --out runs/p0_base.jsonl
$P freport.py runs/p0_base.jsonl

# 2. verifier test in long division: base samples (4 per problem), a division specialist as the peer
$P run.py probs --fields div --levels 2,3,4 --n 100 --seed 201 --out data/p0_div_pool.jsonl
$P run.py probs --fields div --levels 2,3,4 --n 150 --seed 202 --out data/p0_div_train.jsonl
$P bdist.py gen --spec specs/base.json --problems data/p0_div_pool.jsonl --out runs/p0_div_samples.jsonl --k 4 --temp 0.8 --seed 3
$P bdist.py gen --spec specs/base.json --problems data/p0_div_train.jsonl --out runs/p0_div_trainsamp.jsonl --k 4 --temp 0.8 --seed 4
# specialist: answer-key rejection sampling on separate problems, MLP-only (the 19 November recipe)
$P -c "
import json, collections
by = collections.defaultdict(list)
for l in open('runs/p0_div_trainsamp.jsonl'):
    r = json.loads(l)
    if r['ok'] and r['text'] not in [x['text'] for x in by[r['pid']]] and len(by[r['pid']]) < 2:
        by[r['pid']].append(r)
rows = [r for v in by.values() for r in v]
open('data/p0_div_sft.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print('specialist training examples', len(rows))
"
$P run.py train --spec specs/base.json --data data/p0_div_sft.jsonl --out deltas/div1 --bs 8 --accum 2 --epochs 2 2>&1 | grep -E "saved|Error"
echo '{"deltas":[{"path":"deltas/div1","alpha":1.0}]}' > specs/div1.json
BDIST_REMOTE_ONLY=1 $P bdist.py feats --spec specs/base.json --rows runs/p0_div_samples.jsonl --out runs/p0_div_feats.pt --bs 8 &
FE=$!
$P run.py gen --spec specs/div1.json --problems data/p0_div_pool.jsonl --out runs/p0_div_peer.jsonl --bs 128
wait $FE
$P freport.py runs/p0_div_peer.jsonl
$P verify0.py runs/p0_div_samples.jsonl --peer runs/p0_div_peer.jsonl --feats runs/p0_div_feats.pt --transfer ../rsi/probes/g0_big.pt
echo PHASE0_DONE
