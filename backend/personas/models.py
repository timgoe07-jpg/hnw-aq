from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any


@dataclass
class ExperienceRole:
    title: str
    company: str
    description: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class Persona:
    id: str
    name: str
    short_label: str
    age_range: Optional[str] = None
    wealth_range_or_structure: Optional[str] = None
    primary_goal: Optional[str] = None
    key_concern: Optional[str] = None
    investment_behaviour: Optional[str] = None
    why_private_credit_appeals: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Profile:
    full_name: str
    headline: Optional[str] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    profile_url: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    about_summary: Optional[str] = None
    experience: List[ExperienceRole] = field(default_factory=list)
    follower_count: Optional[int] = None
    connections: Optional[int] = None
    photo_url: Optional[str] = None
    is_premium: Optional[bool] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Profile":
        exp = []
        for role in data.get("experience", []):
            if not role:
                continue
            exp.append(
                ExperienceRole(
                    title=role.get("title", ""),
                    company=role.get("company", ""),
                    description=role.get("description", ""),
                    start_date=role.get("start_date"),
                    end_date=role.get("end_date"),
                )
            )
        return Profile(
            full_name=data.get("full_name") or data.get("name", ""),
            headline=data.get("headline"),
            current_title=data.get("current_title"),
            current_company=data.get("current_company"),
            profile_url=data.get("profile_url"),
            location=data.get("location"),
            industry=data.get("industry"),
            about_summary=data.get("about_summary") or data.get("summary"),
            experience=exp,
            follower_count=data.get("follower_count"),
            connections=data.get("connections"),
            photo_url=data.get("photo_url"),
            is_premium=data.get("is_premium"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaMatchScore:
    persona_id: str
    persona_name: str
    score: float
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileMatchResult:
    profile: Profile
    top_persona_matches: List[PersonaMatchScore]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "matches": [m.to_dict() for m in self.top_persona_matches],
        }
