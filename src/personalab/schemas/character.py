"""Character / persona schema for content generation."""

from typing import Any

from pydantic import BaseModel, Field


class Character(BaseModel):
    """Persona used for content ideas, image and video generation."""

    name: str = Field(..., description="Display name of the character")
    vibe: str = Field(..., description="Short vibe or niche (e.g. Lifestyle & Urban Chic)")
    location: str = Field(..., description="Location context (e.g. Lima, Peru)")
    content_pillars: list[str] = Field(
        default_factory=list,
        description="Content pillars or themes",
    )
    physical_description: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured physical description for identity-consistent generation",
    )
    personality: str | None = Field(default=None, description="Optional personality blurb")
    id: str | None = Field(default=None, description="Optional stable id (e.g. daniperez)")

    def to_persona_dict(self) -> dict[str, Any]:
        """Legacy dict shape for code that expects persona dicts."""
        return {
            "name": self.name,
            "vibe": self.vibe,
            "location": self.location,
            "content_pillars": self.content_pillars,
            "physical_description": self.physical_description,
            "personality": self.personality,
            **({"id": self.id} if self.id else {}),
        }
