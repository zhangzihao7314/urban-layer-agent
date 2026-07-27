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


def generate_mask_rules(output_path: str) -> str:
    """
    Generate C1 all-mask rules.json for Layer Alterator.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(MASK_RULES, f, indent=2)

    return str(output_path)