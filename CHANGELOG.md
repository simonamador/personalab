# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-24

### Added

- **Core library** (`personalab`): config, LLM abstraction, prompts, schemas, generation, identity, orchestration, tools.
- `LLMClient` protocol with `TextPart`/`BytesPart` provider-agnostic content parts.
- `GeminiAdapter` implementing `LLMClient` for Google Gemini.
- `IdeaGenerator` for content plan generation with Google Search grounding.
- `ImageGenerator` for identity-consistent image generation with anchor conditioning.
- `VideoGenerator` for scenario-based video generation.
- `PromptBuilder` contract: `PassThroughPromptBuilder` and `GeminiPromptBuilder`.
- `IdentityPolicy` deterministic retry policy and `IdentityEvaluator` stub.
- `AnchorPack` for managing persona reference images.
- Instagram realism post-processing pipeline (`apply_instagram_realism`).
- `RegistryWriter` for append-only JSONL orchestration logs.
- Pydantic schemas: `Character`, `ImagePrompt`, `VideoPrompt`, `ContentPlan`, `ScenarioDetails`.
- YAML-based configuration with typed `ProjectConfig` and package defaults.
- Unified `${VAR}` template syntax across all YAML files.
- CLI scripts: `gen_anchors`, `gen_ideas`, `gen_images`, `gen_videos`, `auto_generate_image`, `evaluate_asset`.
- 93 unit tests covering all domains.
- Example: `examples/quickstart.py`.
