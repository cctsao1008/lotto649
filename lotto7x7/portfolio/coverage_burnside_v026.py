"""Independent Burnside/Mobius orbit-count audit for four 6-of-49 portfolios.

This module does not import the legacy canonicalizer or enumerate the legacy
52,450 membership signatures. Instead, it counts fixed points of S4 ticket
permutations via integer DP on 15 mask-orbits, removes equal-ticket
configurations by the partition-lattice Mobius formula, and applies Burnside.

It certifies *structural classification completeness only*: it does not
independently re-score official prize events, demonstrate forecast skill,
generate tickets, or change any saved portfolio.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import permutations
import json
from math import factorial
from pathlib import Path

SCHEMA = "lotto649.coverage.four_ticket_burnside_independent.v1"
TICKET_COUNT = 4
BALLS_PER_TICKET = 6
LABEL_UNIVERSE = 49
GROUP_SIZE = 24


def apply_ticket_permutation(mask: int, perm: tuple[int, ...]) -> int:
    if (type(mask) is not int or not 1 <= mask <= 15
            or tuple(sorted(perm)) != (0, 1, 2, 3)):
        raise ValueError("invalid mask or S4 permutation")
    # perm[new_ticket] = old_ticket.
    return sum(1 << i for i, old in enumerate(perm) if mask & (1 << old))


def nonzero_mask_orbits(perm: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(1, 16))
    orbits = []
    while remaining:
        first = min(remaining)
        orbit = []
        m = first
        while m not in orbit:
            orbit.append(m)
            m = apply_ticket_permutation(m, perm)
        if m != first or not set(orbit).issubset(remaining):
            raise AssertionError("mask group action is not a partition")
        remaining.difference_update(orbit)
        orbits.append(tuple(orbit))
    if remaining or sum(map(len, orbits)) != 15:
        raise AssertionError("incomplete 15-class mask action")
    return tuple(orbits)


def set_partitions(seq: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Construct the 15 set partitions of four ticket indices independently."""
    if not seq:
        return ((),)
    head, tail = seq[0], seq[1:]
    out = []
    for parts in set_partitions(tail):
        out.append(((head,),) + parts)
        for k in range(len(parts)):
            new_parts = list(parts)
            new_parts[k] = (head,) + new_parts[k]
            out.append(tuple(new_parts))
    return tuple(out)


def partition_mobius(parts: tuple[tuple[int, ...], ...]) -> int:
    """mu(discrete_partition, parts) in the lattice of set partitions."""
    return int(__import__("math").prod(
        (-1) ** (len(b) - 1) * factorial(len(b) - 1) for b in parts
    ))


def count_permutation_fixed_with_equalities(
    perm: tuple[int, ...], parts: tuple[tuple[int, ...], ...],
) -> int:
    """Count 15-class assignments invariant under perm and forced equal rows.

    For each orbit O of nonzero 4-bit membership masks, all c_m for m in O
    must equal one common nonnegative integer q_O. An equality partition
    forbids every O containing a mask with different membership bits inside
    any equality block. Each allowed O contributes to ticket row i exactly
    q_O * sum_{m in O} bit_i(m); integer DP then enforces degree six for
    every row. This is distinct from labeled-signature enumeration and
    lexicographic S4 canonicalization.
    """
    if tuple(sorted(perm)) != (0, 1, 2, 3):
        raise ValueError("invalid ticket permutation")
    if (set(x for block in parts for x in block) != set(range(4))
            or sum(len(block) for block in parts) != 4):
        raise ValueError("partition must contain each ticket index once")
    weights = []
    for orbit in nonzero_mask_orbits(perm):
        forbidden = any(
            any(bool(mask & (1 << block[0])) != bool(mask & (1 << i))
                for i in block[1:])
            for mask in orbit
            for block in parts
        )
        if not forbidden:
            weights.append(tuple(
                sum(bool(mask & (1 << i)) for mask in orbit)
                for i in range(4)
            ))

    @lru_cache(maxsize=None)
    def count(index: int, remaining: tuple[int, int, int, int]) -> int:
        if index == len(weights):
            return int(remaining == (0, 0, 0, 0))
        w = weights[index]
        upper = min(remaining[i] // w[i] for i in range(4) if w[i])
        return sum(
            count(index + 1, tuple(remaining[i] - q * w[i] for i in range(4)))
            for q in range(upper + 1)
        )

    return count(0, (6, 6, 6, 6))


def fixed_valid_signatures(perm: tuple[int, ...]) -> int:
    """Inclusion-exclusion over 15 equality partitions (Möbius inversion)."""
    return sum(
        partition_mobius(parts)
        * count_permutation_fixed_with_equalities(perm, parts)
        for parts in set_partitions((0, 1, 2, 3))
    )


def cycle_type(perm: tuple[int, ...]) -> tuple[int, ...]:
    if tuple(sorted(perm)) != (0, 1, 2, 3):
        raise ValueError("invalid ticket permutation")
    unseen = set(range(4))
    lengths = []
    while unseen:
        v = min(unseen)
        cycle = []
        while v not in cycle:
            cycle.append(v)
            v = perm[v]
        if v != cycle[0]:
            raise AssertionError("not a permutation cycle")
        unseen.difference_update(cycle)
        lengths.append(len(cycle))
    return tuple(sorted(lengths))


def independent_burnside_report() -> dict:
    by_cycle = {}
    for perm in permutations(range(4)):
        kind = cycle_type(perm)
        fixed = fixed_valid_signatures(perm)
        if kind in by_cycle:
            if by_cycle[kind]["fixed_valid"] != fixed:
                raise AssertionError("S4 conjugate permutations have different fixed counts")
            by_cycle[kind]["multiplicity"] += 1
        else:
            by_cycle[kind] = {"cycle_type": list(kind), "multiplicity": 1,
                              "fixed_valid": fixed}
    full = count_permutation_fixed_with_equalities(
        (0, 1, 2, 3), ((0,), (1,), (2,), (3,))
    )
    valid = fixed_valid_signatures((0, 1, 2, 3))
    numerator = sum(row["multiplicity"] * row["fixed_valid"] for row in by_cycle.values())
    if numerator % GROUP_SIZE:
        raise AssertionError("non-integral Burnside orbit count")
    count = numerator // GROUP_SIZE
    return {
        "schema": SCHEMA,
        "method": "independent integer-DP fixed points + partition-lattice Mobius + S4 Burnside",
        "group_order": GROUP_SIZE,
        "total_nonnegative_labeled_mask_structures": full,
        "distinct_ticket_labeled_mask_structures": valid,
        "duplicate_ticket_labeled_mask_structures": full - valid,
        "burnside_fixed_point_numerator": numerator,
        "unlabeled_valid_membership_structures": count,
        "fixed_points_by_conjugacy_class": [
            by_cycle[k] for k in sorted(by_cycle)
        ],
        "expected_legacy_canonical_census": 2957,
        "independent_structural_count_agrees": (
            full == 52450 and valid == 50985 and count == 2957
        ),
        "scientific_decision": "EXACT_STRUCTURE_ONLY_NO_PREDICTIVE_INFERENCE",
        "limit": (
            "Cross-checks the number of membership structures, not the "
            "independent correctness of all official-prize scores in the "
            "original exhaustive certificate; does not establish a lottery edge."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("results/v026_burnside_independent.json"))
    args = p.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((Path.cwd() / "results").resolve()) or out.exists():
        p.error("--out must be a new path inside results/")
    report = independent_burnside_report()
    if not report["independent_structural_count_agrees"]:
        raise AssertionError("independent Burnside census conflicts with old exhaustive certificate")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as fp:
        json.dump(report, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"PASS independent S4 Burnside: full={report['total_nonnegative_labeled_mask_structures']} "
          f"valid={report['distinct_ticket_labeled_mask_structures']} "
          f"orbits={report['unlabeled_valid_membership_structures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
