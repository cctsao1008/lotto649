"""Statistics-first Lotto649 candidate selection."""
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from itertools import chain
import json
from math import isfinite
from pathlib import Path
from typing import Iterable

from . import hexagram
from .statistics import score_candidates
from ..portfolio.deficit_compensation_v025 import validate_ticket
from ..portfolio.shape_deficit_v025 import load_exact_baseline

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ("sum", "frequency", "shape")
HistoryRow = tuple[date, int, tuple[int, ...]]


@dataclass(frozen=True)
class SelectionConfig:
    count: int
    pool_size: int
    seed: int
    statistics: tuple[float, float, float]
    epsilon: float
    hexagram_rule: str

    def __post_init__(self) -> None:
        if (type(self.pool_size) is not int or not 5 <= self.pool_size <= 5000
                or type(self.count) is not int or not 1 <= self.count <= self.pool_size):
            raise ValueError("require 5 <= pool_size <= 5000 and 1 <= count <= pool_size")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if len(self.statistics) != len(COMPONENTS):
            raise ValueError("three statistical weights are required")
        for weight in self.statistics:
            if (type(weight) not in (int, float) or weight < 0
                    or (type(weight) is float and not isfinite(weight))):
                raise ValueError("statistical weights must be finite and nonnegative")
        if max(self.statistics) == 0:
            raise ValueError("at least one statistical weight must be positive")
        if type(self.epsilon) not in (float, int) or not 0 <= self.epsilon <= 2:
            raise ValueError("epsilon must be between 0 and 2")
        if self.hexagram_rule != hexagram.RULE:
            raise ValueError(f"unsupported hexagram rule: {self.hexagram_rule}")

    @classmethod
    def load(cls, path: Path) -> "SelectionConfig":
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        expected_keys = {"count", "pool_size", "seed", "statistics", "epsilon", "hexagram_rule"}
        if not isinstance(config, dict) or config.keys() != expected_keys:
            raise ValueError("invalid selection configuration keys")
        weights = config["statistics"]
        if not isinstance(weights, dict) or weights.keys() != set(COMPONENTS):
            raise ValueError("statistics must specify sum, frequency and shape weights")
        config["statistics"] = tuple(weights[name] for name in COMPONENTS)
        return cls(**config)


def load_history(bin_dir: Path = ROOT / "data/bin") -> tuple[HistoryRow, ...]:
    """Use the repository's shared verification of both BINs."""
    from scripts.export_lotto649_csv import read_verified_draws

    years = read_verified_draws(bin_dir)
    return tuple((day, draw_id, main)
                 for day, draw_id, main, _ in chain.from_iterable(years.values()))


def _rank(pool, components, statistics, code, epsilon, count):
    candidates = [
        (tuple(map(int, row)), tuple(map(float, parts)), float(total))
        for row, parts, total in zip(pool, components, statistics, strict=True)
    ]
    candidates.sort(key=lambda item: (-item[2], item[0]))

    results = []
    band = 0
    start = 0
    while len(results) < count:
        band += 1
        floor = candidates[start][2] - epsilon
        stop = start + 1
        while stop < len(candidates) and candidates[stop][2] >= floor:
            stop += 1
        rows = []
        for ticket, parts, total in candidates[start:stop]:
            ticket_code, match_count = hexagram.score(ticket, code)
            rows.append({
                "numbers": ticket,
                "statistical_score": total,
                "statistical_components": dict(zip(COMPONENTS, parts)),
                "statistical_band": band,
                "ticket_hexagram_code": ticket_code,
                "hexagram_score": match_count,
            })
        rows.sort(key=lambda row: (-row["hexagram_score"], -row["statistical_score"], row["numbers"]))
        results.extend(rows[:count - len(results)])
        start = stop
    return results


def select(history: Iterable[HistoryRow], target_date: date, config: SelectionConfig,
           *, resource_root: Path = ROOT / "data") -> dict:
    """Rank a single seeded pool using only draw rows strictly before target_date."""
    prefix = [(day, draw_id, validate_ticket(numbers))
              for day, draw_id, numbers in history if day < target_date]
    if not prefix:
        raise ValueError("no historical draws before the target date")
    for previous, current in zip(prefix, prefix[1:]):
        if current[0] <= previous[0] or current[1] <= previous[1]:
            raise ValueError("history dates and draw IDs must be strictly increasing")
    date_hexagram = hexagram.cast(target_date, resource_root / "astronomy")
    baseline, shape_sha256 = load_exact_baseline(resource_root / "reference/shape_class_counts.json")
    shape_counts = {int(code): count for code, count in baseline["counts"]["edge"].items()}
    largest = max(config.statistics)
    scaled_weights = tuple(weight / largest for weight in config.statistics)
    weight_sum = sum(scaled_weights)
    weights = tuple(weight / weight_sum for weight in scaled_weights)
    pool, parts, statistics = score_candidates(
        [numbers for _, _, numbers in prefix], pool_size=config.pool_size,
        seed=config.seed, weights=weights, shape_counts=shape_counts,
    )
    canonical_history = [(day.isoformat(), draw_id, numbers) for day, draw_id, numbers in prefix]
    encoded_history = json.dumps(canonical_history, separators=(",", ":")).encode("ascii")
    return {
        "schema": "lotto649.selection.v1",
        "target_date": target_date.isoformat(),
        "history": {
            "draws": len(prefix),
            "last_date": prefix[-1][0].isoformat(),
            "last_draw_id": prefix[-1][1],
            "prefix_sha256": sha256(encoded_history).hexdigest(),
        },
        "settings": {
            "count": config.count,
            "pool_size": config.pool_size,
            "seed": config.seed,
            "statistics": dict(zip(COMPONENTS, weights)),
            "epsilon": config.epsilon,
            "shape_reference_sha256": shape_sha256,
        },
        "hexagram": date_hexagram,
        "candidates": _rank(pool, parts, statistics, date_hexagram["code"], config.epsilon, config.count),
    }
