import json
from pathlib import Path


MASK_RULES = {
    "TCH.tif": "mask",
    "IMD.tif": "mask",
    "BH.tif": "mask",
    "BSF.tif": "mask",
    "SVF.tif": "mask",
    "F_AC.tif": "mask",
    "F_S.tif": "mask",
    "F_M.tif": "mask",
    "F_BS.tif": "mask",
    "F_G.tif": "mask",
    "F_TV.tif": "mask",
    "F_W.tif": "mask",
}

PERCENTAGE_RULES = {name: "pct" for name in MASK_RULES}


def generate_mask_rules(output_path: str) -> str:
    """
    Generate C1 all-mask rules.json for Layer Alterator.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(MASK_RULES, f, indent=2)

    return str(output_path)


def generate_percentage_rules(output_path: str) -> str:
    """Generate professor-notebook-compatible C2 all-PCT rules."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(PERCENTAGE_RULES, file, indent=2)
    return str(output_path)


def generate_partial_percentage_rules(output_path: str, pct_predictors) -> str:
    """Generate a C3 mix: selected predictors PCT, all others NONE."""
    selected = set(pct_predictors)
    known = {name.removesuffix(".tif") for name in MASK_RULES}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown C3 predictors: {sorted(unknown)}")
    if not selected or selected == known:
        raise ValueError("C3 needs at least one PCT and at least one NONE predictor.")
    rules = {
        filename: "pct" if filename.removesuffix(".tif") in selected else "none"
        for filename in MASK_RULES
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")
    return str(output)
