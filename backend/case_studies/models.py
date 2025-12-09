from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class CaseStudy:
    id: str
    title: str
    persona_id: Optional[str]
    persona_name: Optional[str]
    problem: str
    why_it_matters: str
    solution: str
    outcome: str
    capspace_angle: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
