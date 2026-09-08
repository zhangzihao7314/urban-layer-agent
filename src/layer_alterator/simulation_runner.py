"""C0-C3 execution facade for Layer Simulator-compatible rules."""

import json
import shutil
from pathlib import Path

from src.layer_alterator.percentage import apply_percentage, normalize_fraction_rasters
from src.layer_alterator.router import require_executable
from src.layer_alterator.service import (
    LayerAlteratorResult,
    run_c1_layer_alterator,
    run_c2_layer_alterator,
    run_c3_layer_alterator,
)


def _action(value):
    if isinstance(value, dict):
        return str(value.get("action", "none")).lower()
    return str(value).lower()


def _percentage(value):
    return float(value.get("percentage", 0.0)) if isinstance(value, dict) else 0.0


def run_layer_alterator(
    vector_mask_path, rules_path, ucp_folder, fractions_folder, output_folder
):
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    actions = {name: _action(value) for name, value in rules.items()}
    classification = require_executable(actions)
    if classification.code == "C1":
        return run_c1_layer_alterator(
            vector_mask_path, rules_path, ucp_folder, fractions_folder, output_folder
        )
    if classification.code == "C2":
        return run_c2_layer_alterator(
            vector_mask_path, rules_path, ucp_folder, fractions_folder, output_folder
        )
    if classification.code == "C3" and vector_mask_path and len(rules) == 12:
        return run_c3_layer_alterator(
            vector_mask_path, rules_path, ucp_folder, fractions_folder, output_folder
        )

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for filename, value in rules.items():
        action = _action(value)
        source_dir = fractions_folder if filename.startswith("F_") else ucp_folder
        source = Path(source_dir) / filename
        if not source.exists():
            raise FileNotFoundError(f"Raster layer not found: {source}")
        destination = output_dir / filename
        if action == "pct":
            outputs[Path(filename).stem] = apply_percentage(
                str(source), _percentage(value), str(destination)
            )
        else:
            shutil.copy2(source, destination)
            outputs[Path(filename).stem] = str(destination)
    fraction_outputs = [
        path for name, path in outputs.items() if name.startswith("F_")
    ]
    if classification.code in {"C2", "C3"} and len(fraction_outputs) > 1:
        normalize_fraction_rasters(fraction_outputs)
    return LayerAlteratorResult(str(output_dir), outputs)
