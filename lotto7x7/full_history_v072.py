"""GPU-accelerated scenario selection and exhaustive chronological 6/49 replay.

The target draw never enters candidate or ticket construction. Historical
shape-family counts are updated incrementally from prior draws only. CUDA
accelerates matrix-based *independent fair-scenario* candidate scoring; final
reported coverage is recomputed by the existing exact CPU combinatorial DP.
This is a NEW scenario-coverage selector, not the v0.6 exact-greedy selector.
"""
from __future__ import annotations

from bisect import insort
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .baseline_v07 import _int_seed, random_sets
from .candidate_pool_v06 import pool_digest
from .coverage_v06 import exact_coverage, normalized_bets
from .shape_engine_v06 import (enumerate_placements, orbit_size,
                               shape_id, shape_key, validate_images)

SCHEMA = 'lotto7x7.full_history_gpu.v0.7.2'
NAMES = (
    'B0-global-random/random', 'B1-random-pool/random',
    'B1-random-pool/scenario-coverage', 'S1-shape-pool/random',
    'S1-shape-pool/scenario-coverage', 'B2-disjoint/disjoint',
)


def resolve_device(device: str) -> torch.device:
    if device not in ('auto', 'cuda', 'cpu'):
        raise ValueError('device must be auto, cuda or cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False; install a CUDA-enabled PyTorch build')
    return torch.device('cuda' if (device == 'cuda' or
                        (device == 'auto' and torch.cuda.is_available())) else 'cpu')


@lru_cache(maxsize=1024)
def _orbit(key: str):
    return enumerate_placements(key)


class PrefixShapePool:
    """Keeps historical family membership, never looking ahead of cutoff."""
    def __init__(self):
        self.rows = {}
        self.keys = []
        self.last_id = None
        self.count = 0

    def add(self, image, draw_id: int):
        if self.last_id is not None and draw_id <= self.last_id:
            raise ValueError('draw IDs must increase')
        key = shape_key(image)
        if key not in self.rows:
            self.rows[key] = {'shape_key': key, 'shape_id': shape_id(key),
                              'historical_count': 0, 'orbit_size': orbit_size(key)}
            insort(self.keys, key)
        self.rows[key]['historical_count'] += 1
        self.last_id = draw_id
        self.count += 1

    def make(self, *, size: int, families: int, seed: int):
        if len(self.keys) < families:
            raise ValueError('not enough observed shape families at this cutoff')
        rng = np.random.default_rng(seed)
        # Exactly the v0.6 uniform-family and round-robin orbit sampling policy.
        weights = np.full(len(self.keys), 1.0 / len(self.keys), dtype=np.float64)
        chosen = rng.choice(len(self.keys), size=families,
                            replace=False, p=weights)
        selected_keys = [self.keys[int(i)] for i in chosen]
        placements = []
        for key in selected_keys:
            all_positions = _orbit(key)
            placements.append([all_positions[int(i)] for i in
                               rng.permutation(len(all_positions))])
        if sum(map(len, placements)) < size:
            raise ValueError('insufficient legal shape placements')
        cursor = [0] * families
        out = []
        while len(out) < size:
            progress = False
            for i, key in enumerate(selected_keys):
                if len(out) == size:
                    break
                if cursor[i] == len(placements[i]):
                    continue
                out.append(tuple(placements[i][cursor[i]].numbers))
                cursor[i] += 1
                progress = True
            if not progress:
                raise AssertionError('exhausted placement orbits')
        if len(set(out)) != size:
            raise AssertionError('shape pool contains duplicate six-number sets')
        return out, pool_digest([{'numbers': list(v)} for v in out])


def uniform_pool(size: int, seed: int):
    """Exactly the ordered 6/49 draw process used by the v0.7 B1 generator."""
    rng = np.random.default_rng(seed)
    rows, used = [], set()
    while len(rows) < size:
        nums = tuple(sorted(int(v) for v in rng.choice(49, 6, replace=False) + 1))
        if nums not in used:
            used.add(nums)
            rows.append(nums)
    return rows, pool_digest([{'numbers': list(v)} for v in rows])


def _mask(nums):
    return sum(1 << (n - 1) for n in nums)


def _disjoint_option(pool, count: int, order):
    selected, combined = [], 0
    for idx in order:
        item = _mask(pool[int(idx)])
        if item & combined == 0:
            selected.append(int(idx))
            combined |= item
            if len(selected) == count:
                return selected
    return None


@lru_cache(maxsize=5)
def _disjoint_reference(count):
    return exact_coverage(tuple(tuple(range(6*i+1, 6*i+7)) for i in range(count)))


def _theory(bets):
    # Avoid thousands of identical 5-ticket exact DPs when sets are disjoint.
    if len(set(n for ticket in bets for n in ticket)) == 6*len(bets):
        return _disjoint_reference(len(bets))['at_least']
    return exact_coverage(bets)['at_least']


class ScenarioEngine:
    """GPU batched sampled fair-draw scoring; exact DP remains the auditor."""
    def __init__(self, *, device: str, seed: int, scenarios: int):
        self.device = resolve_device(device)
        if type(scenarios) is not int or not 512 <= scenarios <= 262144:
            raise ValueError('scenarios must be an integer between 512 and 262144')
        rng = np.random.default_rng(seed)
        # Uniform 6-of-49 via the six smallest IID uniform values in every row.
        weights = rng.random((scenarios, 49), dtype=np.float64)
        picks = np.argpartition(weights, kth=5, axis=1)[:, :6]
        one_hot = np.zeros((scenarios, 49), dtype=np.float32)
        one_hot[np.arange(scenarios)[:, None], picks] = 1
        self.draws = torch.as_tensor(one_hot, device=self.device)
        self.scenarios = scenarios

    def choose(self, pool, *, count: int, seed: int):
        """Lexicographic sampled >=3/4/5 greedy with exact disjoint check."""
        if not 1 <= count <= 5 or len(pool) < count:
            raise ValueError('invalid ticket count')
        candidates = np.zeros((49, len(pool)), dtype=np.float32)
        for j, ticket in enumerate(pool):
            candidates[np.asarray(ticket, dtype=np.int64) - 1, j] = 1
        with torch.no_grad():
            hits = self.draws @ torch.as_tensor(candidates, device=self.device)
            flags = [(hits >= level) for level in (3, 4, 5)]
            covered = [torch.zeros(self.scenarios, dtype=torch.bool, device=self.device)
                       for _ in range(3)]
            order = np.random.default_rng(seed).permutation(len(pool)).tolist()
            ranking = {int(v): i for i, v in enumerate(order)}
            chosen = []
            for _ in range(count):
                gains = torch.stack([(flag & (~cover[:, None])).sum(dim=0)
                                     for flag, cover in zip(flags, covered)])
                gains_cpu = gains.to('cpu').numpy()
                idx = max((j for j in range(len(pool)) if j not in chosen),
                          key=lambda j: (int(gains_cpu[0, j]), int(gains_cpu[1, j]),
                                         int(gains_cpu[2, j]), -ranking[j]))
                chosen.append(idx)
                covered = [old | flag[:, idx] for old, flag in zip(covered, flags)]
        disjoint = _disjoint_option(pool, count, order)
        if disjoint is not None:
            # Exact postselection safeguard; cannot masquerade sampled score as exact.
            candidate_key = tuple(_theory([pool[j] for j in chosen])[str(h)] for h in (3,4,5))
            disjoint_key = tuple(_disjoint_reference(count)['at_least'][str(h)] for h in (3,4,5))
            if disjoint_key > candidate_key:
                chosen = disjoint
        return chosen


def _draw_id_year(draw_id):
    return str(draw_id // 1_000_000 + 1911)


def _bootstrap_draws(values, seed=2771, reps=2000):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    results = np.empty(reps)
    for k in range(0, reps, 128):
        stop = min(reps, k + 128)
        idx = rng.integers(0, len(arr), size=(stop-k, len(arr)))
        results[k:stop] = arr[idx].mean(axis=1)
    return np.quantile(results, [0.025, 0.975]).tolist()


def _paired(s1, b1, ts1, tb1, *, seed=649):
    a = np.asarray(s1, dtype=np.int8)
    b = np.asarray(b1, dtype=np.int8)
    sf = np.asarray(ts1, dtype=float)
    bf = np.asarray(tb1, dtype=float)
    s_only = int(np.count_nonzero((a == 1) & (b == 0)))
    b_only = int(np.count_nonzero((a == 0) & (b == 1)))
    from scipy.stats import binom
    discordant = s_only + b_only
    p = min(1.0, 2*float(binom.cdf(min(s_only,b_only), discordant, 0.5))) if discordant else 1.0
    diff = a.astype(float) - b
    adjusted = diff - (sf-bf)
    return {'s1_only': s_only, 'b1_only': b_only, 'both': int(np.count_nonzero(a & b)),
            'neither': int(np.count_nonzero((1-a) & (1-b))),
            'difference': float(diff.mean()), 'fair_difference': float((sf-bf).mean()),
            'excess_difference': float(adjusted.mean()), 'exact_mcnemar_p': p,
            'draw_bootstrap_ci95': _bootstrap_draws(diff, seed),
            'excess_bootstrap_ci95': _bootstrap_draws(adjusted, seed+1)}


def score_batch(rows, *, device='auto'):
    """Score all frozen portfolios at once on CUDA, with CPU bitwise parity."""
    chosen_device = resolve_device(device)
    n = len(rows)
    if not n:
        raise ValueError('no rows to score')
    count = len(rows[0]['bets'][NAMES[0]])
    tickets = np.empty((n, len(NAMES), count, 6), dtype=np.int64)
    actual = np.empty((n, 6), dtype=np.int64)
    for t, row in enumerate(rows):
        actual[t] = normalized_bets([row['actual']])[0]
        for j, name in enumerate(NAMES):
            bets = normalized_bets(row['bets'][name])
            if len(bets) != count:
                raise ValueError('inconsistent ticket count')
            tickets[t, j] = bets
    with torch.no_grad():
        bet_tensor = torch.as_tensor(tickets, device=chosen_device)
        actual_tensor = torch.as_tensor(actual, device=chosen_device)
        hits = (bet_tensor[..., None] == actual_tensor[:, None, None, None, :]).sum((-1, -2))
        maxima = hits.max(dim=2).values.to('cpu').numpy()
    return maxima


def run_full_history(images, draw_ids, *, warmup=200, count=5, size=100,
                     families=20, seed=649, scenarios=8192, device='auto',
                     checkpoint: Path | None = None, resume=False,
                     tail_targets: int | None = None,
                     on_progress: Callable[[str], None] | None = None):
    """Replay every eligible chronological draw; resumable and fully audited."""
    x = validate_images(np.asarray(images))
    ids = np.asarray(draw_ids)
    if (ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer) or
        np.any(np.diff(ids) <= 0) or np.any(ids <= 0)):
        raise ValueError('invalid chronological history')
    if (type(warmup) is not int or not 50 <= warmup < len(x) or
        type(count) is not int or not 1 <= count <= 5 or
        type(size) is not int or not count <= size <= 5000 or
        type(families) is not int or not 1 <= families <= size or
        type(seed) is not int or seed < 0):
        raise ValueError('invalid full-history parameters')
    if tail_targets is not None and (type(tail_targets) is not int or tail_targets < 1):
        raise ValueError('tail_targets must be a positive integer')
    start = warmup if tail_targets is None else max(warmup, len(x)-tail_targets)
    active_device = resolve_device(device)
    engine = ScenarioEngine(device=str(active_device), seed=seed+823,
                            scenarios=scenarios)
    archive_hash = hashlib.sha256(x.tobytes()+ids.astype(np.int64).tobytes()).hexdigest()
    params = {'warmup':warmup,'count':count,'size':size,'families':families,
              'seed':seed,'scenarios':scenarios,'tail_targets':tail_targets,
              'archive_sha256':archive_hash}
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    rows = []
    if resume:
        if checkpoint is None or not checkpoint.is_file():
            raise ValueError('resume requires existing checkpoint file')
        for line in checkpoint.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if record['parameters'] != params:
                raise ValueError('checkpoint parameters or archive differ')
            rows.append(record['record'])
        if len(rows) > len(x)-start or any(row['draw_id'] != int(ids[start+i])
                 for i,row in enumerate(rows)):
            raise ValueError('checkpoint target chronology invalid')
    elif checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open('x', encoding='utf-8'):
            pass
    catalog = PrefixShapePool()
    for i in range(start+len(rows)):
        catalog.add(x[i], int(ids[i]))
    for t in range(start+len(rows), len(x)):
        cutoff = int(ids[t-1])
        b0 = random_sets(count, _int_seed(seed, cutoff, 'B0'))
        b2 = random_sets(count, _int_seed(seed, cutoff, 'B2'), disjoint=True)
        b1, b1_sha = uniform_pool(size, _int_seed(seed, cutoff, 'B1-pool'))
        s1, s1_sha = catalog.make(size=size, families=families,
                                   seed=_int_seed(seed, cutoff, 'S1-pool'))
        common_seed = _int_seed(seed, cutoff, 'common-pool-selector')
        order = np.random.default_rng(common_seed).permutation(size).tolist()
        groups = {
            NAMES[0]: b0,
            NAMES[1]: [b1[j] for j in order[:count]],
            NAMES[2]: [b1[j] for j in engine.choose(b1, count=count, seed=common_seed)],
            NAMES[3]: [s1[j] for j in order[:count]],
            NAMES[4]: [s1[j] for j in engine.choose(s1, count=count, seed=common_seed)],
            NAMES[5]: b2,
        }
        groups = {k: [list(v) for v in normalized_bets(vs)] for k,vs in groups.items()}
        fair = {name: _theory(bets) for name, bets in groups.items()}
        row = {'draw_id':int(ids[t]), 'cutoff_draw_id':cutoff,
               'actual': [int(j+1) for j in np.flatnonzero(x[t])],
               'b1_pool_sha256':b1_sha,'s1_pool_sha256':s1_sha,
               'bets':groups, 'fair_coverage':fair}
        rows.append(row)
        if checkpoint is not None:
            with checkpoint.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'parameters':params,'record':row},
                                   sort_keys=True,separators=(',',':'))+'\n')
        catalog.add(x[t], int(ids[t]))
        if on_progress and ((t-start+1) % 25 == 0 or t+1 == len(x)):
            on_progress(f'[FULL {t-start+1}/{len(x)-start}] cutoff={cutoff} -> target={int(ids[t])}')
    maxima = score_batch(rows, device=str(active_device))
    by_arm = {}
    for j,name in enumerate(NAMES):
        vals = maxima[:,j]
        fair = np.asarray([r['fair_coverage'][name]['3'] for r in rows], dtype=float)
        by_arm[name] = {'draws':len(rows), 'max_hits_histogram':[int(np.count_nonzero(vals==k)) for k in range(7)],
                        'mean_max_hits':float(vals.mean()),'realized_ge3':float(np.mean(vals>=3)),
                        'fair_expected_ge3':float(fair.mean()),
                        'ge3_counts':int(np.count_nonzero(vals>=3)),
                        'year':{yr:{'draws':len(ix), 'ge3_count':int(np.count_nonzero(vals[ix]>=3)),
                                    'fair_ge3':float(fair[ix].mean())}
                                for yr,ix in _year_indices(rows).items()}}
    pairs = {}
    for suffix in ('random','scenario-coverage'):
        k1='S1-shape-pool/'+suffix
        k2='B1-random-pool/'+suffix
        pairs[suffix] = _paired(maxima[:,NAMES.index(k1)]>=3, maxima[:,NAMES.index(k2)]>=3,
                                [r['fair_coverage'][k1]['3'] for r in rows],
                                [r['fair_coverage'][k2]['3'] for r in rows], seed=seed)
    # Two predeclared direct contrasts: Holm step-down correction.
    ranks = sorted(pairs, key=lambda label: pairs[label]['exact_mcnemar_p'])
    previous = 0.0
    for i, label in enumerate(ranks):
        previous = max(previous, (len(ranks)-i)*pairs[label]['exact_mcnemar_p'])
        pairs[label]['holm_adjusted_p'] = min(1.0, previous)
    for suffix in pairs:
        k1='S1-shape-pool/'+suffix
        k2='B1-random-pool/'+suffix
        ia=NAMES.index(k1)
        ib=NAMES.index(k2)
        pairs[suffix]['year'] = {year: {
            'targets':len(ix),
            'observed_s1_minus_b1':float(np.mean((maxima[ix,ia]>=3).astype(float)-(maxima[ix,ib]>=3).astype(float))),
            'fair_s1_minus_b1':float(np.mean([rows[int(t)]['fair_coverage'][k1]['3']-rows[int(t)]['fair_coverage'][k2]['3'] for t in ix])),
        } for year, ix in _year_indices(rows).items()}
    return {'schema':SCHEMA,'source_draws':len(x),'evaluated_draws':len(rows),
            'warmup_only_draws':warmup,
            'eligible_draws':len(x)-warmup, 'skipped_eligible_targets':start-warmup,
            'historical_exploratory':True,
            'compute':{'device':str(active_device),
                       'cuda_name':torch.cuda.get_device_name(active_device) if active_device.type=='cuda' else None,
                       'torch_version':torch.__version__,
                       'candidate_optimization':'GPU/CPU Monte Carlo scenario greedy >=3,>=4,>=5; exact disjoint safeguard',
                       'ticket_audit':'exact CPU combinatorial DP or mathematically equivalent disjoint anchor',
                       'scenario_draws':scenarios},
            'parameters':params,'summary':by_arm,'s1_vs_b1_paired':pairs,
            'targets':rows,
            'warnings':['Historical draws were already inspected: exploratory, not blind prediction.',
                        'GPU scenario-coverage is a NEW approximate selector, NOT v0.6 exact-greedy.',
                        'Fair scenarios are independent of future winning draws; final fair coverage is exact.',
                        'Holm correction covers only the two contrasts in this invocation, not past research choices.',
                        'One seed does not establish predictive power; no significance from optimized hindsight.',
                        'No model training is performed; bonus numbers and payouts are not included.']}


def _year_indices(rows):
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        groups[_draw_id_year(row['draw_id'])].append(i)
    return {year:np.asarray(ix, dtype=np.int64) for year,ix in sorted(groups.items())}
