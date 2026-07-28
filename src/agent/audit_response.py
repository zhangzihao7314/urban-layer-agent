"""Small, serialisable audit record for an Agent answer."""

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class AuditableAnswer:
    conclusion: str
    reference_facts: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        sections = [self.conclusion]
        for title, values in (
            ("Reference-table facts", self.reference_facts),
            ("Agent reasoning", self.reasoning),
            ("Assumptions", self.assumptions),
            ("Limitations", self.limitations),
            ("Tools used", self.tools_used),
        ):
            if values:
                sections.append(f"**{title}**\n\n" + "\n".join(f"- {v}" for v in values))
        return "\n\n".join(sections)
