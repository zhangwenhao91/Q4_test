"""MHA-only correctness and API latency harness migrated from Q4."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def implementation_order(labels, round_index):
    order = list(labels)
    if len(order) > 2:
        shift = round_index % len(order)
        order = order[shift:] + order[:shift]
        if (round_index // len(order)) % 2:
            order.reverse()
    elif round_index % 2:
        order.reverse()
    return order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['mha_bf16'], required=True)
    p.add_argument('--mha-seqlen', type=int, choices=[3584, 4096], default=4096)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ac-metadata', choices=['cached', 'per_call'], default='cached')
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if min(a.iters, a.rounds) <= 0 or a.warmup < 0:
        p.error('Invalid iteration count')
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    import tilelang
    from mha_bf16 import make_case
    torch.manual_seed(a.seed)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = True
    with torch.no_grad():
        functions, refs, names, meta = make_case(a)
    props = torch.npu.get_device_properties(0)
    meta.update(q4_revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                seed=a.seed, device=str(props), device_name=torch.npu.get_device_name(0),
                torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                tilelang_version=tilelang.__version__,
                timing=dict(warmup=a.warmup, iters=a.iters, rounds=a.rounds))
    (a.output/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    records = []
    def validate(label, outputs, expected):
        if len(outputs) != len(expected):
            raise AssertionError('Wrong output count')
        for name, y, r in zip(names, outputs, expected):
            y, r = y.detach().cpu(), r.detach().cpu()
            if y.shape != r.shape or y.dtype != r.dtype:
                raise AssertionError('Output shape/dtype mismatch: ' + label)
            delta = y.double() - r.double()
            err = delta.norm().item() / max(r.double().norm().item(), 1e-30)
            finite = bool(torch.isfinite(y).all() and torch.isfinite(r).all())
            maximum = delta.abs().max().item()
            records.append(dict(implementation=label, output=name, shape=list(y.shape), dtype=str(y.dtype),
                finite=finite, max_abs=maximum, relative_l2=err,
                rmse=delta.square().mean().sqrt().item(),
                gate=dict(relative_l2_max=.01, max_abs_max=.02),
                passed=finite and err <= .01 and maximum <= .02))
    values = {}
    with torch.no_grad():
        for label, fn in functions.items():
            outputs = tuple(fn())
            torch.npu.synchronize()
            values[label] = tuple(y.detach().cpu() for y in outputs)
            validate(label, values[label], refs)
        for left, right in itertools.combinations(functions, 2):
            validate(left + '_vs_' + right, values[left], values[right])
    (a.output/'correctness.json').write_text(json.dumps(records, indent=2))
    if not all(r['passed'] for r in records):
        raise AssertionError('Accuracy failed; timing skipped')
    print('Accuracy passed:', a.output, flush=True)
    del values, outputs, refs
    if a.check_only:
        return
    rows = []
    with torch.no_grad():
        for fn in functions.values():
            for _ in range(a.warmup):
                fn()
            torch.npu.synchronize()
        with (a.output/'timings.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'implementation', 'event_us', 'wall_us'])
            writer.writeheader()
            for rnd in range(a.rounds):
                for label in implementation_order(functions, rnd):
                    fn = functions[label]
                    torch.npu.synchronize()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    begin = time.perf_counter()
                    start.record()
                    for _ in range(a.iters):
                        fn()
                    end.record()
                    end.synchronize()
                    row = dict(round=rnd, implementation=label,
                        event_us=start.elapsed_time(end)*1000/a.iters,
                        wall_us=(time.perf_counter()-begin)*1e6/a.iters)
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()
    summary = {label: {key: statistics.median(r[key] for r in rows if r['implementation']==label)
                          for key in ['event_us', 'wall_us']} for label in functions}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
