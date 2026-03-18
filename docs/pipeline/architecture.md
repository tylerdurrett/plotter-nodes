# Technical Architecture

## System Overview

The Portrait Map Lab pipeline implements a modular, extensible architecture for generating importance maps from portrait images. The system follows functional programming principles with pure functions, immutable data structures, and clear separation of concerns.

## Architecture Principles

### Design Philosophy

1. **Modularity**: Each module has single responsibility
2. **Composability**: Functions combine into pipelines
3. **Immutability**: Data flows through transformations without mutation
4. **Extensibility**: Easy to add new map types or processing stages
5. **Testability**: Pure functions with deterministic outputs
6. **Interoperability**: ComfyUI-ready, API-first design

### Data Flow Architecture

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       v
┌──────────────────────────────────────────────┐
│           Pipeline Orchestrator               │
│                                               │
│  ┌────────────┐  ┌────────────┐             │
│  │ Landmarks  │→ │   Masks    │             │
│  └────────────┘  └──────┬─────┘             │
│                         │                    │
│  ┌────────────┐  ┌──────v─────┐             │
│  │  Distance  │← │  Regions   │             │
│  │   Fields   │  └────────────┘             │
│  └──────┬─────┘                              │
│         │                                    │
│  ┌──────v─────┐  ┌────────────┐             │
│  │ Influence  │→ │  Combine   │             │
│  │    Maps    │  └──────┬─────┘             │
│  └────────────┘         │                    │
│                         │                    │
│  ┌─────────────┐  ┌─────v──────┐            │
│  │ Complexity  │→ │ Flow Speed │            │
│  │    Map      │  │ Derivation │            │
│  └─────────────┘  └──────┬─────┘            │
│                          │                   │
│  ┌─────────────┐  ┌──────v─────┐            │
│  │     ETF     │→ │ Flow Field │            │
│  └─────────────┘  │  Blending  │            │
│                   └──────┬─────┘            │
└──────────────────────────┼───────────────────┘
                          │
                    ┌──────v─────┐
                    │   Output   │
                    │  Bundle    │
                    └────────────┘
```

## Module Architecture

### Core Modules

| Module | Responsibility | Dependencies | Outputs |
|--------|---------------|--------------|---------|
| `landmarks.py` | Face detection | MediaPipe | LandmarkResult |
| `face_regions.py` | Region definitions | None | Polygon coordinates |
| `face_contour.py` | Face boundary extraction | SciPy, OpenCV | Contour masks, signed distance |
| `segmentation.py` | Semantic segmentation | MediaPipe, OpenCV | Category masks, contour polygons |
| `masks.py` | Mask rasterization | OpenCV | Binary masks |
| `distance_fields.py` | Distance transforms | SciPy | Distance arrays |
| `remap.py` | Influence mapping | NumPy | Influence arrays |
| `combine.py` | Map combination | NumPy | Combined map |
| `pipelines.py` | Orchestration | All above | PipelineResult, ContourResult |
| `luminance.py` | Luminance & CLAHE | OpenCV | Tonal targets |
| `compose.py` | Map composition | NumPy | Density targets |
| `etf.py` | Edge Tangent Field | OpenCV, SciPy | Tangent fields, coherence |
| `flow_fields.py` | Flow field blending | NumPy | Combined flow fields |
| `lic.py` | LIC visualization | SciPy | Flow textures |
| `complexity_map.py` | Local complexity measurement | OpenCV, SciPy | ComplexityResult |
| `flow_speed.py` | Speed derivation from complexity | NumPy | Speed arrays |

### Export Module

| Module | Responsibility | Dependencies | Outputs |
|--------|---------------|--------------|---------|
| `export.py` | Cross-language export | NumPy, JSON | ExportBundle, manifest.json, .bin files |

### Support Modules

| Module | Responsibility | Usage |
|--------|---------------|-------|
| `models.py` | Data structures | Type definitions |
| `storage.py` | I/O operations | Save/load data |
| `viz.py` | Visualization | Debug/display |

## Data Structures

### Core Models

```python
@dataclass(frozen=True)
class LandmarkResult:
    landmarks: np.ndarray          # (478, 2) coordinates
    image_shape: tuple[int, int]   # (H, W)
    confidence: float              # 0.0-1.0

@dataclass(frozen=True)
class PipelineResult:
    landmarks: LandmarkResult
    masks: dict[str, np.ndarray]
    distance_fields: dict[str, np.ndarray]
    influence_maps: dict[str, np.ndarray]
    combined: np.ndarray

@dataclass(frozen=True)
class ContourResult:
    landmarks: LandmarkResult | None  # None when using segmentation
    contour_polygon: np.ndarray
    contour_mask: np.ndarray
    filled_mask: np.ndarray
    signed_distance: np.ndarray
    directional_distance: np.ndarray
    influence_map: np.ndarray

@dataclass(frozen=True)
class ComplexityResult:
    raw_complexity: np.ndarray     # Unnormalized metric output
    complexity: np.ndarray          # Normalized [0, 1] complexity
    metric: str                     # Which metric was used

@dataclass(frozen=True)
class FlowResult:
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    coherence: np.ndarray
    flow_speed: np.ndarray | None = None  # Speed modulation from complexity

@dataclass(frozen=True)
class ComposedResult:
    feature_result: PipelineResult | None
    contour_result: ContourResult | None
    density_result: DensityResult | None
    flow_result: FlowResult | None
    complexity_result: ComplexityResult | None  # New field
```

### Configuration Models

```python
@dataclass
class RemapConfig:
    curve: str = "gaussian"
    sigma: float = 80.0
    radius: float = 150.0
    tau: float = 60.0
    clamp_distance: float = 300.0

@dataclass
class PipelineConfig:
    regions: list[RegionDefinition]
    remap: RemapConfig
    weights: dict[str, float]

@dataclass
class ContourConfig:
    contour_method: str = "landmarks"  # landmarks, segmentation_face, segmentation_head, average
    remap: RemapConfig
    direction: str = "inward"
    band_width: float | None = None
    contour_thickness: int = 1
    epsilon_factor: float = 0.005
    smooth_contour: bool = True  # Gaussian blur on SDF for rounder contours
    output_dir: str = "output"

@dataclass
class ComplexityConfig:
    metric: str = "gradient"  # gradient, laplacian, multiscale_gradient
    sigma: float = 3.0
    scales: list[float] = field(default_factory=lambda: [1.0, 3.0, 8.0])
    scale_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    normalize_percentile: float = 99.0
    output_dir: str = "output"

@dataclass
class FlowSpeedConfig:
    speed_min: float = 0.3  # Speed in complex areas
    speed_max: float = 1.0  # Speed in smooth areas
```

## Pipeline Orchestration

### Execution Flow

```python
def run_feature_distance_pipeline(image, config):
    # 1. Detect landmarks
    landmarks = detect_landmarks(image)

    # 2. Build masks
    masks = build_region_masks(landmarks, config.regions)

    # 3. Compute distance fields
    distance_fields = {}
    for name, mask in get_feature_masks(masks):
        distance_fields[name] = compute_distance_field(mask)

    # 4. Remap to influence
    influence_maps = {}
    for name, field in distance_fields.items():
        influence_maps[name] = remap_influence(field, config.remap)

    # 5. Combine
    combined = combine_maps(influence_maps, config.weights)

    return PipelineResult(...)
```

### Error Handling

```python
try:
    landmarks = detect_landmarks(image)
except ValueError as e:
    if "No face detected" in str(e):
        return EmptyResult()
    raise

# Graceful degradation
if "mouth" not in masks:
    logger.warning("Mouth not detected, using default")
    masks["mouth"] = np.zeros(shape)
```

## Extension Points

### Adding New Features

1. **Define region** in `face_regions.py`:
```python
RegionDefinition(
    name="nose",
    landmark_indices=[1, 2, 5, 6, ...]
)
```

2. **Update weights** in config:
```python
weights = {
    "eyes": 0.4,
    "mouth": 0.3,
    "nose": 0.3
}
```

### Adding New Map Types

The contour pipeline demonstrates how to add new map types:

1. **Create processor** module:
```python
# face_contour.py
def get_face_oval_polygon(landmarks: LandmarkResult) -> np.ndarray:
    # Extract face boundary using convex hull
    hull = ConvexHull(landmarks.landmarks[:, :2])
    return landmarks.landmarks[hull.vertices]

def compute_signed_distance(contour_mask, filled_mask) -> np.ndarray:
    # Compute signed distance field
    distance = compute_distance_field(contour_mask)
    signed = np.where(filled_mask > 0, -distance, distance)
    return signed
```

2. **Create pipeline function**:
```python
# In pipelines.py
def run_contour_pipeline(image, config=None):
    landmarks = detect_landmarks(image)
    polygon = get_face_oval_polygon(landmarks)
    contour_mask = rasterize_contour_mask(polygon, image.shape[:2])
    filled_mask = rasterize_filled_mask(polygon, image.shape[:2])
    signed_distance = compute_signed_distance(contour_mask, filled_mask)
    directional = prepare_directional_distance(signed_distance, config.direction)
    influence = remap_influence(directional, config.remap)
    return ContourResult(...)
```

3. **Add CLI subcommand**:
```python
# In run_pipeline.py
subparsers = parser.add_subparsers()
contour_parser = subparsers.add_parser('contour')
contour_parser.add_argument('--direction', choices=['inward', 'outward', 'both', 'band'])
```

### Custom Remapping Curves

```python
# In remap.py
elif config.curve == "sigmoid":
    k = config.steepness
    m = config.midpoint
    influence = 1 / (1 + np.exp(-k * (d - m)))
```

## Performance Architecture

### Memory Management

- **Lazy allocation**: Arrays allocated on-demand
- **In-place operations**: Where possible
- **View sharing**: NumPy views avoid copies
- **Garbage collection**: Intermediate results released

### Computational Optimization

- **Vectorization**: All operations use NumPy
- **Broadcasting**: Efficient element-wise operations
- **Parallelization**: Independent operations parallelizable
- **Caching**: Model files cached locally

### Scalability

| Image Size | Memory Usage | Processing Time |
|------------|-------------|-----------------|
| 640×480 | ~10 MB | ~100ms |
| 1920×1080 | ~60 MB | ~300ms |
| 4K (3840×2160) | ~250 MB | ~800ms |

## Integration Architecture

### ComfyUI Integration

The architecture is designed for easy ComfyUI node creation:

```python
class FaceImportanceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "eye_weight": ("FLOAT", {"default": 0.6}),
                "mouth_weight": ("FLOAT", {"default": 0.4}),
                "curve": (["linear", "gaussian", "exponential"],),
            }
        }

    def process(self, image, eye_weight, mouth_weight, curve):
        config = PipelineConfig(
            weights={"eyes": eye_weight, "mouth": mouth_weight},
            remap=RemapConfig(curve=curve)
        )
        result = run_feature_distance_pipeline(image, config)
        return (result.combined,)
```

### API Architecture

```python
# FastAPI example
@app.post("/process")
async def process_portrait(
    file: UploadFile,
    eye_weight: float = 0.6,
    mouth_weight: float = 0.4
):
    image = await load_upload(file)
    result = run_feature_distance_pipeline(image)
    return {
        "importance_map": encode_base64(result.combined),
        "metadata": {
            "confidence": result.landmarks.confidence,
            "shape": result.combined.shape
        }
    }
```

## Testing Architecture

### Unit Testing

Each module has corresponding test file:
```
tests/
    test_landmarks.py     # Mock MediaPipe, test conversions
    test_masks.py         # Test rasterization
    test_distance.py      # Verify distance calculations
    test_remap.py         # Test curve functions
    test_combine.py       # Test weighted combination
    test_pipeline.py      # Integration tests
```

### Test Strategy

1. **Pure functions**: Deterministic input/output tests
2. **Edge cases**: Empty masks, missing features
3. **Numerical**: Verify mathematical correctness
4. **Visual regression**: Compare against reference outputs
5. **Performance**: Benchmark critical paths

## Deployment Architecture

### Package Structure

```
portrait_map_lab/
    __init__.py       # Public API exports
    *.py             # Implementation modules
    models/          # Downloaded model files

scripts/
    run_pipeline.py   # CLI interface

tests/
    test_*.py        # Test modules
    fixtures/        # Test data
```

### Dependency Management

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "opencv-python>=4.8",
    "scipy>=1.10",
    "mediapipe>=0.10",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.1"]
api = ["fastapi>=0.100", "pillow>=10.0"]
comfyui = ["torch>=2.0"]
```

## Future Architecture Considerations

### Planned Extensions

1. **GPU Acceleration**: CUDA kernels for distance transforms
2. **Batch Processing**: Process multiple images simultaneously
3. **Video Support**: Temporal consistency between frames
4. **Neural Features**: Deep learning feature extraction
5. **Web Assembly**: Browser-based processing

### Architectural Patterns

- **Plugin System**: Dynamic loading of processors
- **Pipeline DSL**: Declarative pipeline definitions
- **Streaming**: Process large images in tiles
- **Distributed**: Multi-machine processing for batch jobs

## Conclusion

The architecture prioritizes modularity, extensibility, and integration readiness while maintaining high performance and code quality. The functional design enables easy testing, debugging, and enhancement of individual components without affecting the overall system.