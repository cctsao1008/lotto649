"""Run registered retrospective research using audited source configuration."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import struct
from pathlib import Path

from .experiment_ledger import (
    DEFAULT_REGISTRY, _canonical, _public_artifacts, _registry_root,
)


def run_experiment(experiment_id: str, registry_path: Path = DEFAULT_REGISTRY,
                   *, output: Path) -> dict:
    """Verify public config, require the pinned history, and create one report."""
    registry_path = registry_path.resolve()
    artifacts = _public_artifacts(registry_path)
    registration = next(((row, config) for row, config in artifacts
                         if row["experiment_id"] == experiment_id
                         and row["role"] == "RESEARCH_CONFIG"), None)
    if registration is None:
        raise ValueError(f"no registered research configuration: {experiment_id}")
    entry, config = registration
    root = _registry_root(registry_path)
    report_root = (root / "results").resolve()
    output = output.resolve()
    if (not output.is_relative_to(report_root) or output == report_root
            or output.suffix != ".json"):
        raise ValueError("experiment output must be a JSON file under results/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")

    # Import numerical dependencies only for a requested run.
    from .geometry.geometry_convergence_v029 import audit_repository_inputs
    from scripts.export_lotto649_csv import HEAD, HISTORY, IMAGE

    bin_root = root / "data/bin"
    source, gate = audit_repository_inputs(root / "data/csv", bin_root)
    expected = config["source"]
    count = expected["draws"]
    if len(source) < count:
        raise ValueError("configured historical snapshot is longer than the verified archive")
    actual = {"draws": count}
    for filename, record, key in (
        ("lotto649_history_v1.bin", HISTORY, "history_bin_sha256"),
        ("lotto649_images_v2.bin", IMAGE, "image_bin_sha256"),
    ):
        raw = (bin_root / filename).read_bytes()
        if sha256(raw).hexdigest() != gate["fingerprints"][key]:
            raise ValueError(f"{filename} changed during source verification")
        prefix = bytearray(raw[:HEAD.size + count * record.size])
        struct.pack_into("<I", prefix, 8, count)
        actual[key] = sha256(prefix).hexdigest()
        if record is HISTORY:
            actual["last_draw_id"] = HISTORY.unpack_from(
                prefix, HEAD.size + (count - 1) * HISTORY.size,
            )[0]
    if actual != expected:
        raise ValueError("configured historical snapshot does not match verified archive")
    source = source[:count]

    if config["engine"] == "incremental-geometry-h2":
        from .geometry.incremental_geometry_v031 import analyze_incremental_geometry

        result = analyze_incremental_geometry(source, **config["parameters"])
    else:
        raise AssertionError("validated experiment engine is missing its implementation")

    csv_hashes = gate["fingerprints"].get("csv_file_sha256", {})
    if csv_hashes:
        gate["fingerprints"]["csv_file_sha256"] = {
            Path(name).relative_to(root).as_posix(): digest
            for name, digest in csv_hashes.items()
        }
    result["source_gate"] = gate
    result["experiment_input_prefix"] = actual
    result["experiment"] = {
        "experiment_id": experiment_id,
        "configuration_sha256": sha256(_canonical(config)).hexdigest(),
        "configuration_git_blob_sha1": entry["git_blob_sha1"],
        "engine": config["engine"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_experiment(args.experiment_id, args.registry, output=args.out)
    print(f"{args.experiment_id}: {report['decision']}")
    print(f"REPORT: {args.out}")


if __name__ == "__main__":
    main()
