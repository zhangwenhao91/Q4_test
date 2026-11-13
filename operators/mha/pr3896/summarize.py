"""Summarize measured AC/TL and old-TL/PR ratios without an NPU."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--input', required=True, type=Path)
a = p.parse_args()
if (a.input/'plan.json').exists():
    plan = json.loads((a.input/'plan.json').read_text())
    for tag in plan['tags']:
        for n in plan['lengths']:
            folder = a.input/tag/str(n)
            required = ['correctness.json', 'metadata.json']
            if tag.startswith('bench-'):
                required += ['summary.json', 'timings.csv']
            for name in required:
                if not (folder/name).is_file():
                    raise SystemExit('Incomplete run: missing ' + str(folder/name))
checks = list(a.input.glob('*/*/correctness.json'))
rows = [r for f in checks for r in json.loads(f.read_text())]
if not rows or not all(r['passed'] for r in rows):
    raise SystemExit('Missing or failed accuracy checks; no speedup reported')
print('Accuracy:', len(rows), 'checks passed')
lengths = sorted({int(f.parent.name) for f in a.input.glob('*/*/correctness.json')})
for length in lengths:
    for mode in ('per_call',):
        runs = [json.loads(f.read_text()) for f in sorted(a.input.glob(f'bench-{mode}-rep*/{length}/summary.json'))]
        if not runs:
            continue
        if len(runs) != 3:
            raise SystemExit('Expected three process summaries')
        labels = (('tilelang_q128', 'pr_static', 'pr_dynamic', 'ascendc')
                  if 'tilelang_q128' in runs[0] else ('pr_dynamic', 'ascendc'))
        med = {label: statistics.median(r[label]['event_us'] for r in runs) for label in labels}
        print('S=', length, 'metadata=', mode, 'median_event_us=', json.dumps(med))
        for label in labels[:-1]:
            paired = [r['ascendc']['event_us'] / r[label]['event_us'] for r in runs]
            print(label, 'AC/TL=', med['ascendc']/med[label], 'paired_ratios=', paired)
            if 'tilelang_q128' in med:
                print('old_TL/current=', med['tilelang_q128']/med[label])
