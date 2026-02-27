"""Pydantic schemas for image and video prompts and content plan."""

from typing import Any

from pydantic import BaseModel, Field


class ScenarioDetails(BaseModel):
    """Single scenario: scene, outfit, framing. Shared by content_plan_schema and user_to_image_prompt LLM output."""

    scene: str = Field(default="", description="Setting / location description")
    outfit: str = Field(default="", description="Clothing and accessories")
    framing: str = Field(default="", description="Shot type (e.g. full-body handheld iPhone, mirror selfie)")


class ContentPlanScenario(BaseModel):
    """One item in planned_content: full details for a planned image/video. Same schema as content_plan_schema items."""

    id: str = Field(default="", description="Unique identifier for the content item")
    type: str = Field(default="image", description="Asset type: image or video")
    persona: str = Field(default="", description="Persona key from personas config")
    title: str = Field(default="", description="Short descriptive title")
    scenario_details: ScenarioDetails = Field(
        default_factory=ScenarioDetails,
        description="Scene, outfit, framing for image/video",
    )
    custom_action: str = Field(default="", description="Details for video action")


class ImagePrompt(BaseModel):
    """Structured image scenario: each field maps to a key in the asset scenario.
    Can be built from a user prompt (PromptBuilder) or from generate_ideas scenario_details."""

    subject_description: str = Field(
        default="",
        description="Fills ${SUBJECT_DESCRIPTION} (e.g. physical_description JSON)",
    )
    scene_description: str = Field(default="", description="Fills ${SCENE_DESCRIPTION}")
    clothing_details: str = Field(default="", description="Fills ${CLOTHING_DETAILS}")
    shot_type: str = Field(default="", description="Fills ${SHOT_TYPE} (framing)")
    policy_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Optional policy overrides for identity/quality",
    )

    def to_variables(self) -> dict[str, str]:
        """Variables dict for render_prompt(scenario_template, variables)."""
        return {
            "SUBJECT_DESCRIPTION": self.subject_description,
            "SCENE_DESCRIPTION": self.scene_description,
            "CLOTHING_DETAILS": self.clothing_details,
            "SHOT_TYPE": self.shot_type,
        }

    @classmethod
    def from_scenario_details(
        cls,
        scenario_details: "ScenarioDetails | dict[str, Any]",
        subject_description: str = "",
    ) -> "ImagePrompt":
        """Build ImagePrompt from scenario (ScenarioDetails or content plan scenario_details dict)."""
        if isinstance(scenario_details, ScenarioDetails):
            s = scenario_details
        else:
            s = ScenarioDetails(
                scene=scenario_details.get("scene", ""),
                outfit=scenario_details.get("outfit", ""),
                framing=scenario_details.get("framing", ""),
            )
        return cls(
            subject_description=subject_description,
            scene_description=s.scene,
            clothing_details=s.outfit,
            shot_type=s.framing,
        )


class ContentPlan(BaseModel):
    """Content plan: list of scenarios. Matches content_plan_schema (planned_content)."""

    planned_content: list[ContentPlanScenario] = Field(
        default_factory=list,
        description="List of planned image/video items with scenario_details",
    )


class VideoPrompt(BaseModel):
    """Structured video scenario: each field maps to a video_scenarios template variable.
    Can be built from a content plan scenario or constructed directly."""

    subject_description: str = Field(
        default="",
        description="Fills ${SUBJECT_DESCRIPTION} (e.g. physical_description JSON)",
    )
    action_details: str = Field(default="", description="Fills ${ACTION_DETAILS}")
    location_details: str = Field(default="", description="Fills ${LOCATION_DETAILS}")
    mood_and_expression: str = Field(
        default="", description="Fills ${MOOD_AND_FACE_EXPRESSION}",
    )
    policy_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Optional policy overrides for identity/quality",
    )

    def to_variables(self) -> dict[str, str]:
        """Variables dict for render_prompt(scenario_template, variables)."""
        return {
            "SUBJECT_DESCRIPTION": self.subject_description,
            "ACTION_DETAILS": self.action_details,
            "LOCATION_DETAILS": self.location_details,
            "MOOD_AND_FACE_EXPRESSION": self.mood_and_expression,
        }

    @classmethod
    def from_scenario_details(
        cls,
        scenario_details: "ScenarioDetails | dict[str, Any]",
        subject_description: str = "",
        custom_action: str = "",
    ) -> "VideoPrompt":
        """Build VideoPrompt from a content plan scenario."""
        if isinstance(scenario_details, ScenarioDetails):
            scene = scenario_details.scene
        else:
            scene = scenario_details.get("scene", "")
        action = custom_action or (
            scenario_details.get("custom_action", "")
            if isinstance(scenario_details, dict)
            else ""
        )
        expression = (
            scenario_details.get("expression", "")
            if isinstance(scenario_details, dict)
            else ""
        )
        return cls(
            subject_description=subject_description,
            action_details=action,
            location_details=scene,
            mood_and_expression=expression,
        )


class ImageGenMeta(BaseModel):
    """Metadata for a single image generation call."""

    model: str = ""
    aspect_ratio: str = ""
    prompt_hash: str = ""
    anchors_used: list[str] = Field(default_factory=list)


class VideoGenMeta(BaseModel):
    """Metadata for a video generation call."""

    model: str = ""
    resolution: str = ""
    aspect_ratio: str = ""
    prompt_hash: str = ""
    anchors_used: list[str] = Field(default_factory=list)
    reference_images_count: int = 0
