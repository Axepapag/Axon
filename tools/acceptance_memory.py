"""Bounded, disposable E0 D512 memory mechanism trial through the real Heart.

Frozen contexts have disjoint train/validation/test identities. Targets are
balanced A/B/C; a memoryless reader of the final distractor/query gets 1/3.
This proves a starter memory mechanism, not mastery of the full curriculum.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import torch
from core.e0_two_state import E0TwoStateCore
from runtime.heart.host import HeartHost
from runtime.heart.host_e0 import E0Loop, PHASE_OBSERVE
from curriculum.generators import generate_episode
from curriculum.schema import validate_episode
from substrate.native import encode_ids


def episodes():
    buckets = {name: [] for name in ('train', 'validation', 'test')}
    # All buckets use the same delay distribution; distractor strings differ.
    decoys = {'train': ('xx', 'yy', 'zz'), 'validation': ('xy', 'yz', 'zx'), 'test': ('xz', 'yx', 'zy')}
    for split, noises in decoys.items():
        for symbol in 'ABC':
            for delay in (1, 3):
                for noise in noises:
                    ep = generate_episode('distracted_recall', 1, {'length': 1, 'delay': delay, 'distractors': 1, 'distractor_kind': 'noise'})
                    ep.update(episode_id=f'memory-{split}-{symbol}-{delay}-{noise}', seed=1,
                              steps=[{'kind': 'input', 'text': symbol, 'control': None}] +
                                    [{'kind': 'tick', 'text': '', 'control': None} for _ in range(delay)] +
                                    [{'kind': 'input', 'text': noise, 'control': None}],
                              query='?', expected={'text': symbol, 'control': None}, facts=[symbol],
                              split_key=f'{symbol}:{delay}:{noise}', split=split)
                    assert not validate_episode(ep), validate_episode(ep)
                    buckets[split].append(ep)
    return buckets


def evaluate(loop, items, intervention=None):
    rows = []
    original = loop._generate_tick
    for ep in items:
        def generate(position):
            if position == 0 and intervention == 'zero':
                loop.reasoning_state = torch.zeros_like(loop.reasoning_state)
                loop.response_state = torch.zeros_like(loop.response_state)
            return original(position)
        loop._generate_tick = generate
        altered = copy.deepcopy(ep)
        if intervention in ('swapped', 'irrelevant'):
            altered['steps'][0]['text'] = {'A': 'B', 'B': 'C', 'C': 'A'}[ep['expected']['text']] if intervention == 'swapped' else 'x'
        # Keep evaluation scoring targets, never pass them into generation.
        row = loop.run_episode(altered, split=ep['split'])
        rows.append({'episode_id': ep['episode_id'], 'expected': ep['expected']['text'],
                     'prediction': row['prediction_text'], 'termination': row['generation']['termination'],
                     'exact': row['metrics']['exact_match']})
    loop._generate_tick = original
    return {'exact_accuracy': sum(r['exact'] for r in rows)/len(rows), 'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-updates', type=int, default=540)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(4242); random.seed(4242)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no fallback')
    buckets = episodes()
    plan = {'schema': 'axon-memory-acceptance-plan-v1', 'device': args.device, 'seed': 4242,
            'max_updates': args.max_updates, 'max_wall_seconds': 600, 'optimizer': 'Adam', 'learning_rate': .003,
            'criteria': {'test_exact_min': .8, 'counterfactual_drop_min': .3, 'validation_selection_min': .9},
            'memoryless_input_aware_baseline': 1/3, 'scope': 'E0 D512 three-symbol delayed/distracted memory mechanism only',
            'splits': buckets}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    host = HeartHost(args.output/'organism', consolidator_ids=('consolidator',)); host.start()
    core = E0TwoStateCore(); loop = E0Loop(args.output/'organism', host, core, sum(buckets.values(), []), device=args.device, response_tick_budget=4)
    optimizer = torch.optim.Adam(core.parameters(), lr=.003); loop.register_optimizer(optimizer)
    began = time.monotonic(); updates = 0; progress = []
    try:
        untrained = evaluate(loop, buckets['validation'])
        selected = False
        while updates < args.max_updates and time.monotonic()-began < 600:
            ordered = buckets['train'][:]; random.shuffle(ordered)
            losses = []
            for ep in ordered:
                if updates >= args.max_updates or time.monotonic()-began >= 600: break
                row = loop.run_episode(ep, train=True, optimizer=optimizer, split='train')
                losses.append(row['training']['objective_loss']); updates += 1
            validation = evaluate(loop, buckets['validation'])
            point = {'updates': updates, 'mean_loss': sum(losses)/len(losses), 'validation_exact': validation['exact_accuracy'], 'elapsed_seconds': time.monotonic()-began}
            progress.append(point); print(json.dumps(point), flush=True)
            if validation['exact_accuracy'] >= .9:
                selected = True; break
        test = evaluate(loop, buckets['test'])
        controls = {kind: evaluate(loop, buckets['test'], kind) for kind in ('zero', 'swapped', 'irrelevant')}
        passed = selected and test['exact_accuracy'] >= .8 and all(test['exact_accuracy']-r['exact_accuracy'] >= .3 for r in controls.values())
        checkpoint = loop.save('memory-mechanism-trial')
        report = {'schema': 'axon-memory-acceptance-v1', 'passed': passed, 'scope': plan['scope'], 'device': args.device,
                  'updates': updates, 'elapsed_seconds': time.monotonic()-began, 'plan_sha256': hashlib.sha256((args.output/'plan.json').read_bytes()).hexdigest(),
                  'untrained_validation': untrained, 'progress': progress, 'test': test, 'counterfactuals': controls,
                  'baseline': 1/3, 'checkpoint': str(checkpoint)}
        (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('progress','test','counterfactuals','untrained_validation')}), flush=True)
        return 0 if passed else 2
    finally: host.stop()


if __name__ == '__main__': raise SystemExit(main())
