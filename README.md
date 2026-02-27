# Personalab

**Synthetic identity infrastructure with embedding-based drift control.**

Personalab is a framework for creating and maintaining synthetic personas whose visual identity remains measurably consistent across hundreds of content generations. The core problem it solves is not image generation, but **quantifying and enforcing identity stability over time** through face embeddings, geometric landmark analysis, and longitudinal drift tracking.

Every generated image is evaluated against a canonical anchor using ArcFace 512-d cosine similarity and 106-point facial landmark ratios. A composite score drives deterministic accept/retry/reject decisions. Drift is tracked per persona over time. Without this layer, repeated generation produces faces that look "similar enough" to a human but diverge measurably; with it, you get reproducible, auditable identity consistency.

### What this is

- An **identity consistency engine** with pluggable generation backends.
- A **measurement system** for synthetic face stability (embeddings, landmarks, drift).
- A **retry orchestrator** that uses evaluation scores to improve generations automatically.
- A **longitudinal tracker** that detects if a persona's identity is degrading over sessions.

### What this is not

- A wrapper for any single provider. Gemini, OpenAI, Runware, Replicate, and Runway are built-in adapters; swap or combine them via `config.yml`. The `LLMClient` protocol accepts any backend.
- An image generation library. Generation is a means to produce candidates; the value is in evaluating and controlling them.
- A real-time system. Each generation-evaluation cycle takes 20-40s (generation-bound, not evaluation-bound).

## When to use Personalab


| Use case                                                                                      | Fit                                              |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| You need a synthetic persona that looks like the **same person** across 50+ pieces of content | Yes                                              |
| You need **auditable metrics** proving identity consistency (not just visual inspection)      | Yes                                              |
| You're building a content pipeline where identity drift would break brand trust               | Yes                                              |
| You need **longitudinal drift monitoring** to detect degradation over weeks/months            | Yes                                              |
| You want a quick one-off AI image with no consistency requirements                            | No -- use the Gemini/DALL-E API directly         |
| You need real-time face verification (< 100ms)                                                | No -- use a dedicated FR service                 |
| You want to generate faces without anchors or evaluation                                      | No -- that defeats the purpose of this framework |


## Observed metrics

Measured during end-to-end runs with Gemini image generation and the `buffalo_l` ArcFace model (CPU inference):


| Metric                                         | Typical range      | Notes                                                     |
| ---------------------------------------------- | ------------------ | --------------------------------------------------------- |
| ArcFace cosine similarity (same persona)       | 0.49 -- 0.75       | Headshots score higher than full-body shots               |
| ArcFace cosine similarity (different personas) | 0.10 -- 0.35       | Confirms discriminative power                             |
| Geometric normalized error (headshot)          | 0.02 -- 0.06       | 106-point landmarks, well-framed faces                    |
| Geometric normalized error (full-body)         | 0.10 -- 0.20       | Smaller face bbox reduces landmark precision              |
| Composite score (ACCEPT range)                 | >= 0.55            | Default threshold; configurable                           |
| Composite score (RETRY range)                  | 0.35 -- 0.55       | Triggers escalating prompt patches                        |
| Video embedding std (stable identity)          | < 0.06             | Low std = consistent face across frames                   |
| Video geometric variance (stable identity)     | < 0.002            | Low variance = no landmark drift across frames            |
| Sharpness ratio (frame / anchor)               | 0.4 -- 1.2         | Below 0.4 = excessive smoothing                           |
| Histogram distance (consecutive frames)        | < 0.35             | Above threshold = flicker or tonal drift                  |
| Anchor generation                              | ~20s per candidate | Gemini image model, 2 candidates                          |
| Asset image generation                         | ~20s per image     | Includes anchor injection                                 |
| Identity evaluation                            | ~200ms per image   | CPU-only, cached anchor embedding                         |
| Video quality evaluation                       | ~50ms per video    | Laplacian + histogram on extracted frames                 |
| Instagram realism post-processing              | ~220ms per image   | Noise + aberration + JPEG compression, 70% size reduction |
| FaceRuntime initialization                     | ~3s                | One-time; shared across evaluators                        |


## Features

**Identity layer:**

- ArcFace 512-d embeddings with cosine similarity scoring.
- 106-point geometric landmark ratios (7 unique ratios) with 5-point fallback.
- Adaptive geometric thresholds that relax for full-body/distant shots based on face bbox size.
- Composite scoring with configurable weights (default 70% embedding / 30% geometric).
- Deterministic accept/retry/reject policy with escalating prompt patches.
- Longitudinal drift tracking per persona (JSONL timelines, trend detection via linear regression).
- Automated anchor selection from candidates (`central_similarity` strategy).
- Shared `FaceRuntime` (single InsightFace model load for all evaluators).
- **Video identity verification**: per-frame cosine similarity against the anchor (not frame-to-frame), with temporal statistics (mean, std, min) and geometric temporal variance to detect identity drift across frames.

**Visual quality layer (separate from identity):**

- Laplacian variance sharpness evaluation per frame, compared against anchor as a ratio.
- Illumination consistency checks via histogram distance (Bhattacharyya, chi-squared, or correlation) between consecutive key frames to detect flicker or tonal drift.
- `PostProcessingGate`: advisory recommendations (sharpening, super-resolution) only when identity is verified stable and degradation is purely textural. Identity thresholds are never adjusted based on visual quality.

**Generation layer:**

- Anchor reference generation (frontal, 3/4, smile, silhouette, natural light).
- Content plan generation grounded in Google Search trends.
- Identity-conditioned image generation with anchor injection.
- Scenario-based video generation with identity + quality dual evaluation.
- User prompt to structured `ImagePrompt`/`VideoPrompt` conversion.
- Instagram realism post-processing (noise, chromatic aberration, JPEG artifacts).

**Orchestration:**

- `GenerationOrchestrator`: reusable generate -> evaluate -> retry loop for images.
- `VideoOrchestrator`: video generation with frame extraction, identity evaluation, quality evaluation, and post-processing gate -- all on the same extracted frames.
- Structured JSONL metrics logging per generation attempt.
- All components injectable; no global state.

**Runtime (controlled parallelism):**

- `Runner` abstraction with two implementations: `SyncRunner` (sequential baseline, fully reproducible) and `AsyncRunner` (concurrent via asyncio + ProcessPoolExecutor).
- `JobSpec` / `JobResult` with `job_id`, `seed`, and `config_hash` for deterministic tracking and audit.
- `ProviderSemaphore`: per-provider `asyncio.Semaphore` to limit simultaneous in-flight API requests.
- `RateLimiter`: token-bucket algorithm per provider to smooth bursts and prevent 429 responses.
- `EvaluationPool`: `ProcessPoolExecutor` with per-worker `FaceRuntime` initialization for CPU/GPU-bound identity evaluation.
- `RetryPolicy`: explicit retry governance with mandatory logging of the full prompt applied on every attempt.
- Monotonic `sequence_number` on all JSONL records for stable ordering under concurrency.
- Thread-safe logging across `GenerationMetricsLogger`, `RegistryWriter`, and `DriftTracker`.

**Multi-provider LLM layer:**

- `LLMClient` protocol with normalized response types (`LLMTextResponse`, `LLMImageResponse`, `LLMVideoResponse`).
- Dual sync/async API on every adapter: `generate_image()` + `generate_image_async()`, `generate_video()` + `generate_video_async()`, `generate_text_json()` + `generate_text_json_async()`.
- Built-in adapters: `GeminiAdapter`, `OpenAIAdapter` (text), `RunwareAdapter` (image), `ReplicateAdapter` (image), `RunwayAdapter` (video).
- Per-modality provider selection in `config.yml` (`models.text.provider`, `models.image.provider`, `models.video.provider`).
- `create_client(config)` factory with `RoutingClient` that composes different providers per modality behind a single `LLMClient` interface.

**Extensibility:**

- `LLMClient` protocol -- implement six methods (3 sync + 3 async) and plug it in; or use a built-in adapter.
- `IdentityEvaluator` protocol -- plug in custom scoring.
- `Runner` abstraction -- implement `run(jobs)` for custom execution strategies.
- Graceful degradation: runs without InsightFace using `StubEvaluator`.

## Installation

```bash
git clone <repo-url>
cd influencer
pip install -e .

# Identity evaluation (ArcFace, InsightFace)
pip install -e ".[eval]"

# Alternative LLM providers (install only what you need)
pip install -e ".[openai]"      # OpenAI text generation
pip install -e ".[runware]"     # Runware image generation
pip install -e ".[replicate]"   # Replicate (Stable Diffusion / Flux)
pip install -e ".[runway]"      # RunwayML video generation
pip install -e ".[all-providers]"  # All of the above

# Development
pip install -e ".[dev]"

# Everything
pip install -e ".[dev,eval,all-providers]"
```

### Requirements

**Core:**

- Python >= 3.10
- `pydantic >= 2.0`, `pyyaml >= 6.0`, `python-dotenv >= 1.0`
- `numpy >= 1.24`, `opencv-python >= 4.8`
- `google-genai >= 1.0` (default LLM provider; replaceable)

**Evaluation (optional):**

- `insightface >= 0.7`, `onnxruntime >= 1.16`

**Alternative providers (optional):**

- `openai >= 1.0` -- for `OpenAIAdapter` (text/JSON)
- `runware >= 0.5`, `httpx >= 0.27` -- for `RunwareAdapter` (image)
- `replicate >= 0.25`, `httpx >= 0.27` -- for `ReplicateAdapter` (image)
- `runwayml >= 0.5`, `httpx >= 0.27` -- for `RunwayAdapter` (video)

### Environment

```bash
# .env file (recommended) -- set only the keys for providers you use
GOOGLE_API_KEY=your_key_here        # Gemini (default)
OPENAI_API_KEY=your_key_here        # OpenAI
RUNWARE_API_KEY=your_key_here       # Runware
REPLICATE_API_TOKEN=your_key_here   # Replicate
RUNWAY_API_KEY=your_key_here        # RunwayML
```

## Quick start

```python
from personalab import (
    Character, load_config, create_client,
    IdeaGenerator, ImageGenerator, ImagePrompt,
    select_anchor, build_evaluator,
    IdentityPolicy, GenerationOrchestrator,
    DriftTracker, GenerationMetricsLogger,
)
from personalab.prompts import (
    load_system_prompts, load_asset_scenarios, load_anchor_templates,
)

# 1. Config + LLM client (reads providers from config.yml automatically)
config = load_config()
client = create_client(config)

# 2. Define a persona
character = Character(
    name="Daniela 'Dani' Perez",
    vibe="Lifestyle & Urban Chic",
    location="Lima, Peru",
    content_pillars=["Sustainable Andean Fashion", "Peruvian Gastronomy"],
    physical_description={"age_range": "23", "hair": {"base_color": "dark brown"}},
    id="daniperez",
)

# 3. Generate anchor references
img_gen = ImageGenerator(
    client=client,
    model_name=config.model_name("image"),
    aspect_ratio=config.generation["image"]["aspect_ratio"],
    references_root=config.paths["references"],
)
img_gen.generate_reference_anchors(
    persona_name="daniperez",
    character_prompt=load_anchor_templates(config),
    physical_description=character.physical_description,
    frontal_candidates_k=4,
)

# 4. Auto-select the best anchor
select_anchor("daniperez", config.paths["references"])

# 5. Generate with identity-aware retry
evaluator = build_evaluator(config.evaluation)
policy = IdentityPolicy.from_config(config.evaluation)

orchestrator = GenerationOrchestrator(
    image_generator=img_gen,
    evaluator=evaluator,
    policy=policy,
    drift_tracker=DriftTracker(log_dir=config.evaluation["drift"]["log_dir"]),
    metrics_logger=GenerationMetricsLogger(output_dir=config.paths["output"]),
    output_dir=config.paths["output"],
    references_root=config.paths["references"],
)

# Generate a content plan and produce the first asset
plan = IdeaGenerator(
    client=client, config=config, persona=character,
    system_prompts=load_system_prompts(config),
).generate_and_save("./generated_content/daniperez/content_plan.yml")

scenario = plan["planned_content"][0]["scenario_details"]
result = orchestrator.run(
    persona="daniperez",
    scenario_template=load_asset_scenarios(config)["image_scenarios"],
    image_prompt=ImagePrompt.from_scenario_details(scenario, subject_description="..."),
)
# result.accepted, result.final_score, result.attempts
```

See `[examples/quickstart.py](examples/quickstart.py)` for a complete runnable version.

## Project structure

```
src/personalab/
├── identity/        # FaceRuntime, ArcFace/Geometric evaluators, anchor selector, policy, drift
│   ├── video_evaluator.py    # Per-frame identity scoring with temporal stats
│   ├── frame_extractor.py    # Uniform / face-detected frame sampling
│   └── ...
├── quality/         # Visual quality evaluation (separate from identity)
│   ├── sharpness.py              # Laplacian variance evaluation
│   ├── illumination.py           # Histogram distance between key frames
│   ├── video_quality_evaluator.py # Composed sharpness + illumination
│   ├── post_processing_gate.py   # Advisory post-processing recommendations
│   └── schemas.py                # FrameSharpness, VideoQualityResult, PostProcessingAdvice
├── runtime/         # Controlled parallelism: runners, concurrency, eval pool
│   ├── runner.py         # Runner ABC, SyncRunner, AsyncRunner
│   ├── job.py            # JobSpec, JobResult, compute_config_hash
│   ├── concurrency.py    # ProviderSemaphore, RateLimiter
│   ├── eval_pool.py      # EvaluationPool (ProcessPoolExecutor)
│   └── retry_policy.py   # Explicit retry governance with prompt logging
├── orchestration/   # GenerationOrchestrator, VideoOrchestrator, RegistryWriter
├── observability/   # GenerationMetricsLogger (structured JSONL, thread-safe)
├── generation/      # IdeaGenerator, ImageGenerator, VideoGenerator
├── llm/             # LLMClient protocol (sync + async), response types, factory, adapters
│   ├── client.py         # Protocol + LLMTextResponse / LLMImageResponse / LLMVideoResponse
│   ├── factory.py        # create_client(), RoutingClient
│   ├── gemini.py         # GeminiAdapter (text, image, video -- sync & async)
│   ├── openai_adapter.py # OpenAIAdapter (text/JSON -- sync & async)
│   ├── runware_adapter.py    # RunwareAdapter (image -- sync & async)
│   ├── replicate_adapter.py  # ReplicateAdapter (image -- sync & async)
│   └── runway_adapter.py     # RunwayAdapter (video -- sync & async)
├── prompts/         # Template rendering, hashing, PromptBuilder, data loading
├── schemas/         # Pydantic models: Character, ImagePrompt, evaluation results, meta
├── config/          # Typed config loader, YAML utilities
├── tools/           # Post-processing: Instagram realism filter
└── data/            # Package YAML defaults (config, prompts, examples, references)

scripts/             # CLI entry points
examples/            # Runnable quickstart
tests/               # 278 unit tests across all domains
```

### Architecture

1. **Identity first** -- `identity/` is the central domain. Generators produce candidates; the identity layer decides if they're acceptable.
2. **Protocol-driven** -- `LLMClient` and `IdentityEvaluator` are Protocols (structural typing). Implementations are swappable without inheritance.
3. **Dependency inversion** -- Each adapter imports only its vendor SDK (`google.genai`, `openai`, `runware`, etc.); only `FaceRuntime` imports `insightface`. Everything else depends on abstractions and normalized response types.
4. **Multi-provider routing** -- `create_client()` reads per-modality provider settings and returns either a single adapter or a `RoutingClient` that composes different providers behind a unified `LLMClient` interface.
5. **No global state** -- All components are instantiated with explicit dependencies (constructor injection). The `GenerationOrchestrator` composes them into a loop.
6. **Graceful degradation** -- `build_evaluator()` returns a `StubEvaluator` when InsightFace is absent. The generation pipeline still works; you just lose real scoring.
7. **Parallelism without drift** -- The `runtime/` domain separates I/O-bound concurrency (`asyncio` for API calls) from CPU/GPU-bound parallelism (`ProcessPoolExecutor` for identity evaluation). `SyncRunner` preserves fully sequential execution for debugging and reproducibility; `AsyncRunner` accelerates throughput without changing metrics or accept/reject decisions.

## Configuration

### Paths


| Key                | Default               | Purpose                                    |
| ------------------ | --------------------- | ------------------------------------------ |
| `paths.output`     | `./generated_content` | Plans, images, videos, metrics, drift logs |
| `paths.references` | `./references`        | Anchor images per persona                  |
| `paths.prompts`    | `./prompts`           | Optional user overrides for YAML templates |


### Evaluation

```yaml
evaluation:
  embedding:
    backend: "arcface"
    model_name: "buffalo_l"
    similarity_threshold: 0.55       # Cosine similarity floor
  geometric:
    enabled: true
    max_normalized_error: 0.08       # Base threshold (auto-relaxed for small faces)
  scoring:
    weights:
      embedding: 0.7
      geometric: 0.3
    accept_threshold: 0.55           # Composite score to ACCEPT
    retry_threshold: 0.35            # Below this -> REJECT_FINAL immediately
    policy_mode: "composite_only"    # "composite_only" | "strict"
  retry:
    max_attempts: 3
  drift:
    enabled: true
    log_dir: "./generated_content/drift"
```

**Policy modes:**

- `composite_only` (default) -- Accept based solely on the weighted composite score.
- `strict` -- Requires composite >= threshold AND all sub-evaluators pass individually.

### Quality evaluation

```yaml
evaluation:
  quality:
    enabled: true
    sharpness:
      min_ratio: 0.4                # frame laplacian / anchor laplacian
    illumination:
      max_histogram_distance: 0.35  # Bhattacharyya distance threshold
      method: "bhattacharyya"       # "bhattacharyya" | "chi_squared" | "correlation"
    post_processing:
      auto_sharpen: false
      auto_super_res: false
      max_embedding_std: 0.06       # Max embedding std for identity stability
      max_geometric_variance: 0.002 # Max geometric ratio variance for stability
```

Quality evaluation runs in parallel with identity evaluation on the same extracted frames. Its results are advisory and do not affect accept/reject decisions.

### Models (per-modality provider selection)

```yaml
models:
  text:
    provider: "gemini"              # gemini | openai
    model_name: "gemini-3-flash-preview"
  image:
    provider: "gemini"              # gemini | runware | replicate
    model_name: "gemini-3-pro-image-preview"
  video:
    provider: "gemini"              # gemini | runway
    model_name: "veo-3.1-fast-generate-preview"
```

Each modality can use a different provider. The `create_client()` factory reads these settings and returns a `RoutingClient` when providers differ, or a single adapter when all three match.

**Available providers:**

| Modality | Providers                          | Notes                                      |
| -------- | ---------------------------------- | ------------------------------------------ |
| text     | `gemini`, `openai`                 | JSON structured output                     |
| image    | `gemini`, `runware`, `replicate`   | Replicate runs Stable Diffusion / Flux     |
| video    | `gemini`, `runway`                 | Runway polls for async task completion      |

**Backward compatibility:** plain-string model names (e.g., `text: "gemini-3-flash-preview"`) are still accepted and default to `gemini` as the provider.

### Runtime (parallelism controls)

```yaml
runtime:
  max_concurrent_jobs: 4          # asyncio tasks running in parallel
  eval_pool_workers: 2            # ProcessPoolExecutor workers for identity eval
  provider_limits:                # per-provider semaphore (max in-flight API calls)
    gemini: 5
    runware: 10
    replicate: 3
    runway: 2
    openai: 5
  rate_limits:                    # per-provider token-bucket (requests/second)
    gemini: 5.0
    runware: 10.0
    replicate: 2.0
    runway: 1.0
    openai: 5.0
  retry:
    max_retries: 3
    backoff_base: 1.0             # seconds
    backoff_max: 30.0
    backoff_factor: 2.0           # exponential multiplier
```

All values have sensible defaults if omitted. The runtime section is only consumed by `AsyncRunner` -- `SyncRunner` ignores it entirely and delegates to the existing orchestrators.

## Identity evaluation

### Composite scoring

```
composite = w_embedding * cosine_similarity + w_geometric * (1 - normalized_error)
```

The composite drives all policy decisions. Individual sub-evaluator pass/fail flags are available for inspection but do not gate acceptance in `composite_only` mode.

### ArcFace embeddings

- 512-dimensional vectors from InsightFace's `buffalo_l` recognition model.
- Cosine similarity between anchor and candidate, clamped to [0, 1].
- Anchor embedding cached across retry attempts.

### Geometric landmarks

- **106-point** (preferred): 7 unique normalized ratios -- inter-eye to nose tip, nose length, mouth width, jaw width, nose width, upper lip to chin, eye height.
- **5-point** (fallback): 2 unique ratios -- nose-to-mouth center, mouth width.
- All ratios normalized by inter-eye distance (scale-invariant).

### Adaptive thresholds

The geometric error threshold relaxes automatically when the face occupies a small portion of the image:


| Face bbox / image diagonal | Effective threshold  |
| -------------------------- | -------------------- |
| >= 30%                     | Base (0.08)          |
| 10% -- 30%                 | Linear interpolation |
| <= 10%                     | Base x 2.5 (0.20)    |


This prevents false rejections on full-body or street photography shots.

### Drift tracking

Per-persona JSONL timeline at `drift/{persona}_drift.jsonl`. Each record contains composite score, embedding similarity, geometric error, decision, and timestamp. `DriftTracker.summary()` returns rolling mean, std, min, max, and trend direction (improving/degrading/stable via simple linear regression over the last N records).

### Video identity verification

Video evaluation extracts key frames (uniform or face-detected sampling) and evaluates **each frame against the original anchor image** -- never frame-to-frame. The result includes per-frame `CandidateResult` scores plus temporal statistics:

- **`EmbeddingTemporalStats`**: mean, std, and min of cosine similarities across all frames. A low std indicates the model maintained a stable identity throughout the video. A high std suggests intermittent identity loss even if the mean looks acceptable.
- **`GeometricTemporalStats`**: per-ratio variance of landmark deltas across frames. Detects geometric drift (e.g., changing jaw width or nose proportions) that the mean error would mask.
- **`std_composite_score`**: standard deviation of the weighted composite score across frames.

These statistics are **informational only** -- they do not change the accept/reject thresholds. The `IdentityPolicy` still decides based on the mean composite score, same as for images.

### Visual quality evaluation

Visual quality is evaluated separately from identity and lives in the `quality/` domain. The two axes are independent: a video can have perfect identity preservation but blurry frames, or sharp frames but identity drift.

**Sharpness (Laplacian variance)**

Each frame's sharpness is measured via the variance of the Laplacian operator and compared against the anchor's Laplacian variance as a ratio:

```
sharpness_ratio = frame_laplacian_variance / anchor_laplacian_variance
```

A ratio near 1.0 means the frame is as sharp as the anchor; a ratio below the `min_ratio` threshold (default 0.4) indicates excessive smoothing or blur.

**Illumination consistency (histogram distance)**

Normalized grayscale histograms are compared between consecutive key frames using Bhattacharyya distance (default), chi-squared, or correlation. High distances between adjacent frames indicate flicker or tonal drift.

**Post-processing gate**

The `PostProcessingGate` combines identity and quality verdicts:

| Identity stable | Quality degraded | Degradation type | Recommendation                          |
| --------------- | ---------------- | ---------------- | --------------------------------------- |
| Yes             | No               | none             | No action needed                        |
| Yes             | Yes              | textural         | Recommend sharpening + super-resolution |
| Yes             | Yes              | illumination     | No auto-fix; manual review              |
| Yes             | Yes              | both             | Manual review                           |
| No              | Any              | Any              | No post-processing (identity unstable)  |

The gate **never suggests adjusting identity thresholds** based on visual quality. Post-processing is only recommended when identity is verified stable and the degradation is purely textural.

### FaceRuntime

A single `FaceRuntime` instance loads all InsightFace ONNX models once (~3s, ~500MB) and is injected into both evaluators via the factory. No duplicate model loading.

## Usage guide

### 1. Generate anchors

```bash
python scripts/gen_anchors.py daniperez
```

Creates frontal candidates in `references/daniperez/anchors/candidates/`.

### 2. Select anchor

```python
from personalab import select_anchor
select_anchor("daniperez", "./references")  # central_similarity strategy
```

Or manually copy your preferred candidate to `references/daniperez/anchors/daniperez_anchor_frontal_neutral.png`.

### 3. Generate with evaluation

```bash
python scripts/auto_generate_image.py daniperez
```

Runs the full orchestrator: generate -> evaluate -> retry -> log.

### 4. Inspect metrics

```python
from personalab import GenerationMetricsLogger
metrics = GenerationMetricsLogger(output_dir="./generated_content")
for record in metrics.read_log("daniperez"):
    print(record["scores"], record["decision"])
```

### 5. Monitor drift

```python
from personalab import DriftTracker
drift = DriftTracker(log_dir="./generated_content/drift")
summary = drift.summary("daniperez")
print(f"mean={summary.mean_score:.4f}, trend={summary.trend}")
```

### 6. Video generation with quality verification

```python
from personalab import (
    VideoGenerator, VideoPrompt, FrameExtractor,
    VideoIdentityEvaluator, build_evaluator, IdentityPolicy,
    VideoOrchestrator, DriftTracker, GenerationMetricsLogger,
    SharpnessEvaluator, IlluminationEvaluator,
    VideoQualityEvaluator, PostProcessingGate,
)

config = load_config()
client = create_client(config)
eval_cfg = config.evaluation

# Identity evaluator (reused from image pipeline)
evaluator = build_evaluator(eval_cfg)
policy = IdentityPolicy.from_config(eval_cfg)

# Video identity evaluator
frame_extractor = FrameExtractor()
video_evaluator = VideoIdentityEvaluator(evaluator, frame_extractor)

# Quality evaluator (independent of identity)
quality_cfg = eval_cfg.get("quality", {})
quality_evaluator = VideoQualityEvaluator(
    sharpness=SharpnessEvaluator(
        min_sharpness_ratio=quality_cfg.get("sharpness", {}).get("min_ratio", 0.4),
    ),
    illumination=IlluminationEvaluator(
        max_histogram_distance=quality_cfg.get("illumination", {}).get("max_histogram_distance", 0.35),
        method=quality_cfg.get("illumination", {}).get("method", "bhattacharyya"),
    ),
)

# Post-processing gate
pp_cfg = quality_cfg.get("post_processing", {})
gate = PostProcessingGate(
    identity_accept_threshold=eval_cfg["scoring"]["accept_threshold"],
    max_embedding_std=pp_cfg.get("max_embedding_std", 0.06),
    max_geometric_variance=pp_cfg.get("max_geometric_variance", 0.002),
)

# Video orchestrator with both axes
vid_gen = VideoGenerator(
    client=client,
    model_name=config.model_name("video"),
    resolution=config.generation["video"]["resolution"],
    aspect_ratio=config.generation["video"]["aspect_ratio"],
)

video_orch = VideoOrchestrator(
    video_generator=vid_gen,
    video_evaluator=video_evaluator,
    policy=policy,
    frame_extractor=frame_extractor,
    quality_evaluator=quality_evaluator,
    post_processing_gate=gate,
    drift_tracker=DriftTracker(log_dir=eval_cfg["drift"]["log_dir"]),
    metrics_logger=GenerationMetricsLogger(output_dir=config.paths["output"]),
    output_dir=config.paths["output"],
    references_root=config.paths["references"],
)

result = video_orch.run(
    persona="daniperez",
    scenario_template=load_asset_scenarios(config)["video_scenarios"],
    video_prompt=VideoPrompt(
        subject_description="young Peruvian woman, 23",
        action_details="walking through a market",
        location_details="Barranco, Lima",
    ),
)

# Inspect results
for attempt in result.attempts:
    identity = attempt.eval_result
    if identity:
        emb = identity.embedding_stats
        print(f"  Embedding: mean={emb.mean_cosine:.4f}, std={emb.std_cosine:.4f}, min={emb.min_cosine:.4f}")
        geo = identity.geometric_stats
        if geo:
            print(f"  Geometric variance: mean={geo.mean_ratio_variance:.6f}, max={geo.max_ratio_variance:.6f}")

    quality = attempt.quality_result
    if quality:
        print(f"  Sharpness: mean_ratio={quality.mean_sharpness_ratio:.4f}, ok={quality.sharpness_ok}")
        print(f"  Illumination: max_dist={quality.max_histogram_distance:.4f}, ok={quality.illumination_ok}")

    advice = attempt.post_processing_advice
    if advice:
        print(f"  Gate: identity_stable={advice.identity_stable}, recommend_sharpen={advice.recommend_sharpening}")
```

### 7. Parallel generation with AsyncRunner

Run multiple generation jobs concurrently while keeping evaluation deterministic and JSONL logs ordered:

```python
from personalab import (
    load_config, create_client,
    ImageGenerator, ImagePrompt,
    build_evaluator, IdentityPolicy,
    DriftTracker, GenerationMetricsLogger,
)
from personalab.runtime import (
    JobSpec, AsyncRunner, RetryPolicy,
    EvaluationPool, ProviderSemaphore, RateLimiter,
)
import hashlib, json

config = load_config()
client = create_client(config)
eval_cfg = config.evaluation
rt_cfg = config.runtime

# Build the async runner with concurrency controls
runner = AsyncRunner(
    client=client,
    image_generator=ImageGenerator(
        client=client,
        model_name=config.model_name("image"),
        aspect_ratio=config.generation["image"]["aspect_ratio"],
        references_root=config.paths["references"],
    ),
    policy=IdentityPolicy.from_config(eval_cfg),
    retry_policy=RetryPolicy(
        max_retries=rt_cfg.get("retry", {}).get("max_retries", 3),
    ),
    eval_pool=EvaluationPool(
        max_workers=rt_cfg.get("eval_pool_workers", 2),
        eval_cfg=eval_cfg,
    ),
    provider_semaphore=ProviderSemaphore(rt_cfg.get("provider_limits")),
    rate_limiter=RateLimiter(rt_cfg.get("rate_limits")),
    drift_tracker=DriftTracker(log_dir=eval_cfg["drift"]["log_dir"]),
    metrics_logger=GenerationMetricsLogger(output_dir=config.paths["output"]),
    output_dir=config.paths["output"],
    references_root=config.paths["references"],
    provider_key=config.generation["image"].get("provider", "gemini"),
)

# Define jobs -- each one has a unique job_id, seed, and config_hash
scenarios = [
    {"name": "cafe_selfie", "prompt": ImagePrompt(subject_description="...", ...)},
    {"name": "street_walk", "prompt": ImagePrompt(subject_description="...", ...)},
    {"name": "gym_mirror",  "prompt": ImagePrompt(subject_description="...", ...)},
]

jobs = []
for i, s in enumerate(scenarios):
    cfg_bytes = json.dumps(s["prompt"].model_dump(), sort_keys=True).encode()
    jobs.append(JobSpec(
        job_id=f"batch-001-{s['name']}",
        seed=42 + i,
        config_hash=hashlib.sha256(cfg_bytes).hexdigest()[:16],
        persona="daniperez",
        scenario_template={},
        prompt=s["prompt"],
        asset_type="image",
    ))

# Run all jobs concurrently -- I/O via asyncio, evaluation via ProcessPoolExecutor
results = runner.run(jobs)

for r in results:
    status = "ACCEPTED" if r.accepted else "REJECTED"
    print(f"[{r.job_id}] {status}  score={r.final_score:.4f}  attempts={r.final_attempt}")

# Clean up the evaluation process pool
runner._eval_pool.shutdown()
```

**How it works:**

- `AsyncRunner.run()` creates an asyncio event loop and dispatches all `JobSpec`s as concurrent tasks.
- Each task acquires a `ProviderSemaphore` slot and passes through the `RateLimiter` before calling `generate_image_async()`.
- Identity evaluation is offloaded to a `ProcessPoolExecutor` via `EvaluationPool`, which initializes an independent `FaceRuntime` in each worker process (avoiding pickling issues).
- `RetryPolicy` governs retry decisions and logs the full prompt applied on every attempt.
- Every JSONL record receives a monotonic `sequence_number` so logs can be correctly reordered regardless of concurrent write timing.
- `SyncRunner` offers the same `run(jobs)` interface but delegates to the existing synchronous orchestrators, producing identical results for debugging and regression.

## Extending

### Custom LLM provider

Implement the six `LLMClient` protocol methods (3 sync + 3 async) and return the normalized response types:

```python
from personalab.llm import LLMClient, LLMTextResponse, LLMImageResponse, LLMVideoResponse

class MyAdapter:
    # --- Synchronous ---
    def generate_text_json(self, *, system_instruction, user_prompt, schema, use_search, model_name) -> LLMTextResponse:
        ...
        return LLMTextResponse(parsed=data, text=raw)

    def generate_image(self, *, parts, aspect_ratio, model_name) -> LLMImageResponse:
        ...
        return LLMImageResponse(images=[img_bytes])

    def generate_video(self, *, prompt, resolution, aspect_ratio, model_name) -> LLMVideoResponse:
        ...
        return LLMVideoResponse(video_data=video_bytes, operation_name="op-123")

    # --- Asynchronous (used by AsyncRunner) ---
    async def generate_text_json_async(self, *, system_instruction, user_prompt, schema, use_search, model_name) -> LLMTextResponse:
        ...

    async def generate_image_async(self, *, parts, aspect_ratio, model_name) -> LLMImageResponse:
        ...

    async def generate_video_async(self, *, prompt, resolution, aspect_ratio, model_name) -> LLMVideoResponse:
        ...
```

The async methods mirror the sync API exactly. If your SDK only provides a synchronous client, wrap calls with `asyncio.get_event_loop().run_in_executor(None, ...)` as the built-in `ReplicateAdapter` and `RunwayAdapter` do.

To register it with the factory, add an entry to `_REGISTRY` in `llm/factory.py` or instantiate it directly and pass it to any component that accepts an `LLMClient`.

### Custom evaluator

```python
from personalab import EvaluationResult, CandidateResult, EmbeddingScore

class MyEvaluator:
    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult:
        ...
```

Both plug directly into `GenerationOrchestrator` via constructor injection.

## Development

```bash
pip install -e ".[dev,eval]"
pytest tests/ -v   # ~278 tests, ~9s
```

Tests cover all domains with deterministic fakes (no API calls). The `FakeLLMClient` (with both sync and async methods) and fake evaluators ensure unit tests run in seconds. Each adapter has its own test file with mocked vendor SDK calls. The `quality/` domain has dedicated tests for sharpness, illumination, the composed evaluator, and the post-processing gate. The `runtime/` domain has dedicated tests for `JobSpec`/`JobResult`, `ProviderSemaphore`, `RateLimiter`, `EvaluationPool`, `RetryPolicy`, and both `SyncRunner`/`AsyncRunner` (using `pytest-asyncio`).

### Conventions

- `${VAR}` syntax for YAML template substitution.
- Pydantic schemas for all data contracts (no loose dicts at boundaries).
- One test directory per domain: `tests/{domain}/`.
- Protocols for all cross-domain contracts (`LLMClient`, `IdentityEvaluator`, `Runner`).

## License

MIT -- see [LICENSE](LICENSE).