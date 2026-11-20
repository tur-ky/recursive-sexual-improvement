#!/usr/bin/env bash
# Phase 0, second half (restarted): the division specialist trains on levels 1-2 only, where the base
# model gets some answers right; levels 3-4 produced no usable data and very long answers.
cd "$(dirname "$0")"
P=../rsi/.venv/Scripts/python
$P run.py probs --fields div --levels 1,2 --n 225 --seed 202 --out data/p0_div_train.jsonl
$P bdist.py gen --spec specs/base.json --problems data/p0_div_train.jsonl --out runs/p0_div_trainsamp.jsonl --k 4 --temp 0.8 --seed 4 --max_new 512
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
$P run.py gen --spec specs/div1.json --problems data/p0_div_pool.jsonl --out runs/p0_div_peer.jsonl --bs 128 2>&1 | tail -1
wait $FE
$P freport.py runs/p0_div_samples.jsonl runs/p0_div_peer.jsonl
$P verify0.py runs/p0_div_samples.jsonl --peer runs/p0_div_peer.jsonl --feats runs/p0_div_feats.pt --transfer ../rsi/probes/g0_big.pt
echo PHASE0_DONE
