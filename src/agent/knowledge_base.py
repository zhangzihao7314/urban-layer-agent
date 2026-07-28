"""Table-driven planning typology and LCZ knowledge.

The professor-provided workbook is the authority for planning-type/LCZ
correspondence.  Predictor values remain in the separate CSV table.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import zipfile
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TYPOLOGY_TABLE = PROJECT_ROOT / "data/reference/LCZ_urban_types_match.xlsx"
CORRESPONDENCE_SCORE = {"P": 3.0, "S": 2.0, "O": 1.0, "–": 0.0, "-": 0.0}
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CANONICAL_URBAN_TYPES = {
    "low plants": "Low Plants",
    "bare soil or sand": "Bare Soil or sand",
    "bare rock/paved": "Bare rock or paved",
}


@dataclass(frozen=True)
class LCZCandidate:
    lcz_code: int
    urban_type: str
    correspondence: str
    score: float
    evidence: str


def _base_mark(value: object) -> str:
    text = str(value).strip()
    return text[:1].upper() if text and text.lower() != "nan" else "–"


class TypologyKnowledgeBase:
    def __init__(self, workbook_path: Path | str = DEFAULT_TYPOLOGY_TABLE):
        self.workbook_path = Path(workbook_path)
        if not self.workbook_path.exists():
            raise FileNotFoundError(f"Typology reference workbook not found: {self.workbook_path}")
        raw = _read_first_xlsx_sheet(self.workbook_path)
        self.aggregated_classes = self._parse_aggregated(raw)
        self.matrix = self._parse_matrix(raw)

    @staticmethod
    def _parse_aggregated(raw: pd.DataFrame) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for _, row in raw.iloc[1:9, :2].iterrows():
            name = str(row.iloc[0]).strip()
            if not name or name.lower() == "nan":
                continue
            result[name] = [int(value) for value in re.findall(r"\d+", str(row.iloc[1]))]
        return result

    @staticmethod
    def _parse_matrix(raw: pd.DataFrame) -> pd.DataFrame:
        header_row = next(
            i for i, value in enumerate(raw.iloc[:, 0].astype(str))
            if value.strip() == "LCZ"
        )
        headers = [str(value).strip() for value in raw.iloc[header_row].tolist()]
        records = []
        for _, row in raw.iloc[header_row + 1:].iterrows():
            label = str(row.iloc[0]).strip()
            match = re.match(r"(\d+)\s+(.+)", label)
            if not match:
                continue
            raw_name = match.group(2).strip()
            record = {
                "LCZ": int(match.group(1)),
                "Class Name": CANONICAL_URBAN_TYPES.get(raw_name.lower(), raw_name),
            }
            for column, value in zip(headers[1:], row.iloc[1:]):
                record[column] = str(value).strip()
            records.append(record)
        if not records:
            raise ValueError("No LCZ correspondence matrix found in typology workbook.")
        return pd.DataFrame(records)

    @property
    def planning_typologies(self) -> List[str]:
        return [column for column in self.matrix.columns if column not in {"LCZ", "Class Name"}]

    @property
    def urban_types(self) -> List[str]:
        return self.matrix["Class Name"].tolist()

    def resolve_planning_typology(self, text: str) -> Optional[str]:
        normalized = text.lower()
        aliases = {
            "historic core": "Historic Core",
            "urban core": "Historic Core",
            "perimeter block": "Perimeter Block (19th–early 20th c.)",
            "housing estate": "Modern Midrise Housing Estate",
            "suburban": "Suburban Detached Housing",
            "detached housing": "Suburban Detached Housing",
            "industrial": "Industrial / Logistics",
            "logistics": "Industrial / Logistics",
            "commercial": "Commercial Big-Box",
            "big-box": "Commercial Big-Box",
            "park": "Urban Park",
            "forest": "Urban Forest",
            "agricultural": "Agricultural Fringe",
            "farmland": "Agricultural Fringe",
            "waterfront": "Waterfront",
            "transport": "Transport / Large Paved",
            "parking": "Transport / Large Paved",
            "paved": "Transport / Large Paved",
        }
        matches = [(len(alias), typology) for alias, typology in aliases.items() if alias in normalized]
        return max(matches, default=(0, None))[1]

    def candidates_for(self, planning_typology: str, include_occasional: bool = True) -> List[LCZCandidate]:
        if planning_typology not in self.planning_typologies:
            raise ValueError(f"Unknown planning typology: {planning_typology}")
        candidates = []
        for _, row in self.matrix.iterrows():
            raw = row[planning_typology]
            mark = _base_mark(raw)
            score = CORRESPONDENCE_SCORE.get(mark, 0.0)
            if score <= 0 or (mark == "O" and not include_occasional):
                continue
            candidates.append(LCZCandidate(
                lcz_code=int(row["LCZ"]),
                urban_type=str(row["Class Name"]),
                correspondence=mark,
                score=score,
                evidence=f"{planning_typology} → LCZ {row['LCZ']} {row['Class Name']}: {raw}",
            ))
        return sorted(candidates, key=lambda item: (-item.score, item.lcz_code))

    def aggregated_candidates(self, aggregated_class: str) -> List[LCZCandidate]:
        if aggregated_class not in self.aggregated_classes:
            raise ValueError(f"Unknown aggregated planning class: {aggregated_class}")
        codes = self.aggregated_classes[aggregated_class]
        rows = self.matrix[self.matrix["LCZ"].isin(codes)]
        return [
            LCZCandidate(int(row["LCZ"]), str(row["Class Name"]), "P", 3.0,
                         f"{aggregated_class} contains LCZ {row['LCZ']}")
            for _, row in rows.iterrows()
        ]


@lru_cache(maxsize=4)
def get_knowledge_base(path: str = str(DEFAULT_TYPOLOGY_TABLE)) -> TypologyKnowledgeBase:
    return TypologyKnowledgeBase(path)


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _read_first_xlsx_sheet(path: Path) -> pd.DataFrame:
    """Read values from the first XLSX worksheet without optional dependencies."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XML_NS):
                shared.append("".join(node.text or "" for node in item.findall(".//a:t", XML_NS)))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    cells = {}
    max_row = max_col = 0
    for cell in sheet.findall(".//a:c", XML_NS):
        reference = cell.attrib["r"]
        row_index = int(re.search(r"\d+", reference).group(0)) - 1
        column_index = _column_index(reference)
        value_node = cell.find("a:v", XML_NS)
        inline_node = cell.find("a:is/a:t", XML_NS)
        value = inline_node.text if inline_node is not None else (value_node.text if value_node is not None else None)
        if cell.attrib.get("t") == "s" and value is not None:
            value = shared[int(value)]
        cells[(row_index, column_index)] = value
        max_row, max_col = max(max_row, row_index), max(max_col, column_index)
    matrix = [[cells.get((row, column)) for column in range(max_col + 1)] for row in range(max_row + 1)]
    return pd.DataFrame(matrix)
