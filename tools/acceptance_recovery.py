"""Disposable exact E0 training recovery on explicitly selected devices."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from core.e0_two_state import E0TwoStateCore
from curriculum.generators import generate_episode
from runtime.heart.host import HeartHost
from runtime.heart.host_e0 import E0Loop


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def check(root, device):
    episode = generate_episode('delayed_recall', 43, {'length': 2, 'delay': 3})
    def build(path):
        torch.manual_seed(43); random.seed(43); np.random.seed(43)
        host = HeartHost(path, consolidator_ids=('consolidator',)); host.start()
        core = E0TwoStateCore()
        loop = E0Loop(path, host, core, [episode], device=device, response_tick_budget=8)
        optimizer = torch.optim.Adam(core.parameters(), lr=.001); loop.register_optimizer(optimizer)
        return host, core, loop, optimizer
    host, core, loop, optimizer = build(root/'reference')
    try:
        expected = loop.run_episode(episode, train=True, optimizer=optimizer)
        weights = {k:v.detach().clone() for k,v in core.state_dict().items()}
        opt = optimizer.state_dict(); generation = host.generation
    finally: host.stop()
    host, core, loop, optimizer = build(root/'interrupted')
    original = core.step
    def interrupted(ids, reasoning, response):
        if loop._generation.get('response_ticks') == 1:
            loop.save('mid-response')
            raise RuntimeError('intentional acceptance interruption')
        return original(ids, reasoning, response)
    core.step = interrupted
    try:
        try: loop.run_episode(episode, train=True, optimizer=optimizer)
        except RuntimeError as exc:
            if str(exc) != 'intentional acceptance interruption': raise
        else: raise AssertionError('interruption not reached')
        saved_cpu = torch.get_rng_state().clone()
        saved_cuda = torch.cuda.get_rng_state_all() if device.startswith('cuda') else []
        saved_numpy = np.random.get_state(); saved_python = random.getstate()
    finally: host.stop()
    host, core, loop, optimizer = build(root/'interrupted')
    try:
        loaded = loop.load('mid-response')
        rng = torch.equal(saved_cpu, torch.get_rng_state()) and equal(saved_cuda, torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
        rng = rng and np.array_equal(saved_numpy[1], np.random.get_state()[1]) and saved_python == random.getstate()
        actual = loop.run_episode(episode, train=True, optimizer=optimizer)
        checks = {'weights_bit_equal': equal(weights, core.state_dict()), 'optimizer_bit_equal': equal(opt, optimizer.state_dict()),
                  'losses_bit_equal': expected['training'] == actual['training'], 'response_equal': expected['prediction_text'] == actual['prediction_text'],
                  'no_duplicate_commits': host.generation == generation, 'all_rng_restored': rng, 'optimizer_restored': loaded['optimizer_restored']}
        return {'device':device, 'passed':all(checks.values()), 'checks':checks}
    finally: host.stop()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    result = {'schema':'axon-recovery-acceptance-v1', 'devices':[check(args.output/device.replace(':','-'), device) for device in ('cpu','cuda:0')]}
    result['passed'] = all(row['passed'] for row in result['devices'])
    (args.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 2

if __name__ == '__main__': raise SystemExit(main())
