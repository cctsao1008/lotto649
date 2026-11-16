"""Compatibility checks for relocated modules and registered experiments."""
from copy import deepcopy
import importlib
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lotto7x7 import experiment_ledger as ledger
from lotto7x7 import experiment_runner as runner

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/configs/v031-h2-geometry-ci.json"


def test_legacy_import_is_canonical_module():
    canonical = importlib.import_module("lotto7x7.legacy.strategy.strategy_verification_v15")
    old = importlib.import_module("lotto7x7.strategy_verification_v15")
    assert old is canonical
    assert canonical.__spec__.name == "lotto7x7.legacy.strategy.strategy_verification_v15"
    assert canonical.SOURCE_CPP.is_file()
    assert importlib.reload(old) is canonical


def test_module_cli_aliases_forward_to_same_parser():
    args = [sys.executable, "-m"]
    old = subprocess.check_output([*args, "lotto7x7.strategy_verification_v15", "--help"])
    new = subprocess.check_output([
        *args, "lotto7x7.legacy.strategy.strategy_verification_v15", "--help",
    ])
    assert old == new


def test_registered_config_rejects_invalid_engine_parameters_and_output(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ledger.validate_research_config(config)
    for patch in ({"engine": "arbitrary.module"}, {"parameters": {**config["parameters"], "seed": True}},
                  {"source": {**config["source"], "history_bin_sha256": "bad"}}):
        with pytest.raises(ValueError):
            ledger.validate_research_config({**deepcopy(config), **patch})
    with pytest.raises(ValueError, match="under results"):
        runner.run_experiment(config["experiment_id"], output=tmp_path / "out.json")
    with pytest.raises(ValueError, match="no registered research"):
        runner.run_experiment("unknown", output=tmp_path / "out.json")


def test_registered_runner_uses_only_pinned_prefix(tmp_path, monkeypatch):
    from lotto7x7.geometry import geometry_convergence_v029, incremental_geometry_v031
    from scripts.export_lotto649_csv import HEAD, HISTORY, IMAGE
    import struct

    experiment_dir = tmp_path / "experiments"
    config_dir = experiment_dir / "configs"
    config_dir.mkdir(parents=True)
    bin_root = tmp_path / "data/bin"
    bin_root.mkdir(parents=True)
    config = json.loads(CONFIG.read_text())
    config["source"]["draws"] = 2498
    config["source"]["last_draw_id"] = 115000104
    fingerprints = {}
    for filename, record, key in (
        ("lotto649_history_v1.bin", HISTORY, "history_bin_sha256"),
        ("lotto649_images_v2.bin", IMAGE, "image_bin_sha256"),
    ):
        full = (ROOT / "data/bin" / filename).read_bytes()
        (bin_root / filename).write_bytes(full)
        prefix = bytearray(full[:HEAD.size + 2498 * record.size])
        struct.pack_into("<I", prefix, 8, 2498)
        config["source"][key] = sha256(prefix).hexdigest()
        fingerprints[key] = sha256(full).hexdigest()
    config_file = config_dir / CONFIG.name
    config_file.write_text(json.dumps(config), encoding="ascii")
    registry = experiment_dir / "ledger_registry_v1.json"
    registry.write_text(json.dumps({"schema": ledger.PUBLIC_SCHEMA, "entries": [{
        "experiment_id": config["experiment_id"], "role": "RESEARCH_CONFIG",
        "artifact": f"experiments/configs/{config_file.name}",
        "git_blob_sha1": ledger._git_blob_sha1(config_file.read_bytes()),
    }]}), encoding="ascii")
    draws = np.zeros((2499, 6), dtype=int)
    monkeypatch.setattr(geometry_convergence_v029, "audit_repository_inputs", lambda *args: (
        draws, {"draws": len(draws), "fingerprints": fingerprints},
    ))

    def analyze(source, **parameters):
        assert len(source) == 2498
        assert parameters == config["parameters"]
        return {"decision":"TEST"}

    monkeypatch.setattr(incremental_geometry_v031, "analyze_incremental_geometry", analyze)
    output = tmp_path / "results/test.json"
    report = runner.run_experiment(config["experiment_id"], registry, output=output)
    assert report["experiment_input_prefix"] == config["source"]
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError):
        runner.run_experiment(config["experiment_id"], registry, output=output)
    config_file.write_text(json.dumps(config) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="differs from frozen Git blob"):
        runner.run_experiment(config["experiment_id"], registry, output=output)
