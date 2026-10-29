"""Summarize paired short runs; do not extrapolate a subset to a full epoch."""
import argparse
import json
from pathlib import Path
import statistics

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = {'scope': 'four GPUs, same 256-image subset, eight optimizer updates; first two updates excluded'}
    logs = {}
    for name in ('before', 'after'):
        run = args.directory / f'{name}_run'
        rows = [json.loads(line) for line in (run / 'result/stage1/train_log.txt').read_text().splitlines() if line.startswith('{')]
        assert len(rows) == 8 and rows[-1]['step'] == 8
        intervals = [b['timestamp'] - a['timestamp'] for a, b in zip(rows[1:-1], rows[2:])]
        logs[name] = rows
        report[name] = {'seconds_per_update': statistics.mean(intervals),
                        'intervals_seconds': intervals, 'losses': [r['loss'] for r in rows]}
    report['throughput_speedup'] = report['before']['seconds_per_update'] / report['after']['seconds_per_update']
    report['max_absolute_logged_loss_difference'] = max(abs(a['loss']-b['loss']) for a,b in zip(logs['before'], logs['after']))
    states = [torch.load(args.directory / f'{name}_run/ckpt/stage1/latest_network.pth', map_location='cpu')['model']
              for name in ('before', 'after')]
    assert states[0].keys() == states[1].keys()
    delta2 = total2 = 0.
    worst = []
    for key, a in states[0].items():
        b = states[1][key]
        assert a.shape == b.shape
        if a.is_floating_point():
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), key
            delta = (a.double() - b.double())
            delta2 += delta.square().sum().item()
            total2 += a.double().square().sum().item()
            worst.append((delta.abs().max().item(), key))
    report['state_relative_l2_difference'] = (delta2 / max(total2, 1e-30)) ** .5
    report['largest_state_differences'] = sorted(worst, reverse=True)[:10]
    report['limitations'] = ['Short subset, not full training or a GPU scaling measurement.',
                            'CUDA reduction ordering may change; checkpoint equality is diagnostic, not a bitwise guarantee.']
    (args.directory / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
