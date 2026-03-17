# Portrait Map Lab

## README / PRD

## 1. Overview

**Portrait Map Lab** is a Python-based research and development repository for generating **portrait analysis maps** that will later support an algorithmic continuous-line drawing system for SVG and pen-plotter output.

The long-term vision is a system that takes an image, associated depth/normal information, and eventually 3D geometry, then generates a line-based drawing whose accumulated line density creates tone while also feeling structurally tied to the form of the subject.

This repository is the **Stage 1 map-generation lab** for that larger system.

Its immediate focus is on creating reusable, testable pipelines for deriving density and importance maps from portraits. The first such map family is centered on **distance and influence relative to the eyes and mouth**.

---

## 2. Purpose

This repo exists to provide a clean and modular environment for experimenting with and implementing the first layer of portrait-analysis logic before any heavy investment in ComfyUI integration or downstream line-generation systems.

It is intended to answer questions like:

* How should important facial regions be detected?
* How should those regions be turned into masks?
* How should masks become distance fields?
* How should raw distance fields be remapped into useful influence maps?
* What intermediate outputs are worth preserving for later stages?
* How should the code be structured so it can later be wrapped in ComfyUI nodes?

---

## 3. Product Vision

The broader project aims to create a **continuous-line SVG generation pipeline** for portrait imagery. That future system will likely use:

* original image data
* depth maps
* normal maps
* reconstructed 3D geometry
* feature importance maps
* target density maps
* directional flow or contour-following maps
* visibility and plotting constraints

The conceptual behavior is that a line or particle moves through or around the subject, leaving behind a trail. The density of the trail creates shading, while the orientation and movement of the trail reflect the structure of the subject.

This repo is responsible for building the **upstream maps** that later systems will consume.

---

## 4. Current Scope

The current scope is intentionally narrow and focused.

### In scope

* Standalone Python repo for portrait map experimentation
* Face landmark extraction
* Semantic face-region definition
* Region mask generation
* Distance field generation
* Distance remapping into influence / density helper maps
* Visualization and debugging outputs
* Architecture that supports later ComfyUI wrapping

### Out of scope for now

* Final SVG generation
* Continuous-line optimization
* Hidden-surface removal
* Mesh geodesic computation
* Full plotter simulation
* Building the entire pipeline directly inside ComfyUI

---

## 5. Architectural Decisions Already Made

### 5.1 Start with a standalone Python repo

We decided to begin with a normal Python repository rather than starting directly in ComfyUI.

**Reasoning:**

* The immediate work is algorithmic experimentation, not workflow graph design.
* We need flexibility for testing, iteration, and debugging.
* The core logic should be stable and reusable before any node wrappers are built.

### 5.2 Use MediaPipe directly first

We decided to begin by using **MediaPipe directly from Python** for face landmark detection.

**Reasoning:**

* Better visibility into what the detection code is doing
* Easier debugging and validation
* Less dependency on third-party ComfyUI node maintenance
* Cleaner foundation for future wrappers

### 5.3 ComfyUI remains a planned integration target

We do plan to later expose this repo’s capabilities as **ComfyUI custom nodes**.

**Important architectural implication:**

* Core logic should live in the Python package
* Future ComfyUI nodes should be thin wrappers around the core package
* Core processing code should remain UI-agnostic and Comfy-agnostic

### 5.4 Start with 2D image-plane distance fields

We decided that the first implementation of the eye/mouth importance map should use **2D Euclidean image-space distance**, not geodesic or mesh-surface distance.

**Reasoning:**

* Much simpler to implement and validate
* Correct first step for the current stage
* Good enough to test the usefulness of these maps
* Surface-aware variants can come later

### 5.5 Use region masks, not only feature-center points

We decided the main implementation should use **full eye and mouth regions**, not just center points.

**Reasoning:**

* The eyes and mouth are extended structures, not single points
* Distance to regions gives more natural spatial behavior
* Region-based maps will be more useful downstream

### 5.6 Preserve both raw and remapped outputs

We want to preserve:

* raw masks
* raw distance fields
* remapped influence fields
* combined outputs
* debug visualizations

**Reasoning:**

* Intermediate outputs will matter for later experimentation
* We do not want a black-box one-off script
* The repo should function as a map-generation lab

---

## 6. Initial Feature Focus

## 6.1 Eye and Mouth Distance / Influence Maps

The first concrete output family is a set of maps that describe proximity to the **eyes** and **mouth**.

These maps are intended to function as portrait importance helpers. Since eyes and mouth are highly important to identity and visual recognition, these maps may later be used to:

* bias target density
* prioritize residual error
* modulate line density
* weight optimization objectives
* influence path allocation in continuous-line generation

---

## 7. First Pipeline Definition

The first end-to-end pipeline should look like this:

1. Load portrait image
2. Run face landmark detection
3. Convert landmarks into semantic facial regions
4. Rasterize regions into binary masks
5. Compute Euclidean distance transforms from the masks
6. Remap raw distance into influence fields
7. Combine influence fields into a unified feature-importance image
8. Save intermediate and final outputs

---

## 8. First Output Set

For a single portrait input, the repo should be able to produce at least:

* Original input image
* Landmark overlay image
* Left eye mask
* Right eye mask
* Combined eye mask
* Mouth mask
* Raw distance-to-eyes image
* Raw distance-to-mouth image
* Remapped eye influence image
* Remapped mouth influence image
* Combined feature influence image
* Optional debug contact sheet showing all outputs together

---

## 9. Functional Requirements

### 9.1 Landmark extraction

The system must support extracting face landmarks from a portrait image.

### 9.2 Region definition

The system must define semantic facial regions in a centralized way, starting with:

* left eye
* right eye
* mouth

### 9.3 Mask generation

The system must convert semantic regions into rasterized binary masks.

### 9.4 Distance transform generation

The system must compute Euclidean distance fields from the generated masks.

### 9.5 Influence remapping

The system must support remapping raw distances into influence images via configurable falloff functions.

### 9.6 Feature-map combination

The system must support combining eye and mouth influence maps with configurable weighting.

### 9.7 Visualization and debugging

The system must support clear visualization of intermediate and final outputs.

### 9.8 Repeatable execution

The system must support a repeatable script-driven workflow for running the pipeline on one or more images.

---

## 10. Non-Functional Requirements

### 10.1 Modularity

Code should be organized into reusable modules rather than large monolithic scripts.

### 10.2 Comfy-wrap readiness

Core logic should be designed so future ComfyUI custom nodes can wrap it with minimal duplication.

### 10.3 Debuggability

Intermediate products should be easy to inspect.

### 10.4 Extensibility

The architecture should support adding new map types later.

### 10.5 Minimal UI assumptions

The processing code should not depend on notebooks, Gradio, or ComfyUI. Those should be optional layers on top.

---

## 11. Proposed Repository Structure

```text
portrait-map-lab/
  pyproject.toml
  README.md
  src/
    portrait_map_lab/
      __init__.py
      landmarks.py
      face_regions.py
      masks.py
      distance_fields.py
      remap.py
      pipelines.py
      viz.py
      io.py
      types.py
  scripts/
    run_landmarks.py
    run_feature_distance_map.py
  notebooks/
    feature_distance_experiments.ipynb
  tests/
```

This structure is intended to separate:

* reusable package code
* execution/testing scripts
* optional exploratory notebooks
* tests

---

## 12. Proposed Module Responsibilities

### `landmarks.py`

Responsibilities:

* run MediaPipe face landmark detection
* normalize outputs
* expose image-space coordinates in a stable internal format

### `face_regions.py`

Responsibilities:

* define semantic face-region presets
* map landmark outputs into polygons for left eye, right eye, and mouth

### `masks.py`

Responsibilities:

* rasterize polygons into binary masks
* support combined and per-feature masks

### `distance_fields.py`

Responsibilities:

* compute Euclidean distance transforms
* preserve raw distance outputs

### `remap.py`

Responsibilities:

* remap distance into influence values
* support multiple curve styles such as:

  * exponential
  * gaussian
  * capped linear

### `pipelines.py`

Responsibilities:

* define end-to-end processing pipelines
* keep script entry points thin and consistent

### `viz.py`

Responsibilities:

* generate overlays
* preview masks
* visualize distance fields
* produce debug contact sheets

### `types.py`

Responsibilities:

* define structured internal data objects where useful
* keep interfaces stable and explicit

---

## 13. Design Guidance for the Developer

### 13.1 Think in terms of a map-generation framework

Even though we are starting with one map family, the architecture should support future additions such as:

* silhouette distance maps
* nose distance maps
* depth-based importance maps
* curvature approximations
* saliency maps
* target density maps

### 13.2 Centralize semantic region definitions

Landmark region groupings should live in one dedicated place. Do not scatter landmark index lists throughout the repo.

### 13.3 Preserve intermediate outputs

Raw distance maps and masks are not just debug artifacts. They are part of the product’s research value.

### 13.4 Keep visualization first-class

This repo is exploratory. Good visual debugging will matter almost as much as the processing code itself.

### 13.5 Keep the core package interface clean

The package should be easy to call:

* from scripts
* from notebooks
* from a future light UI
* from future ComfyUI nodes

---

## 14. Suggested First Milestone

## Milestone 1: Eye / Mouth Feature Distance Pipeline

### Goal

Build a reliable first pipeline that takes a portrait image and outputs robust eye and mouth masks, raw distance maps, remapped influence maps, and a combined feature-importance map.

### Acceptance Criteria

A developer should be able to run a command and obtain:

* landmark detection
* eye and mouth region extraction
* mask generation
* distance transforms
* influence remapping
* combined feature map
* saved debug outputs

### Tunable Parameters

The first implementation should support tuning at least:

* eye weight
* mouth weight
* eye falloff radius or decay
* mouth falloff radius or decay
* max distance clamp
* output normalization mode

---

## 15. Future Integration Plan

Once the initial Python implementation is stable, the next phase will be to wrap selected capabilities as **ComfyUI custom nodes**.

Likely first candidate nodes:

* Detect face landmarks
* Build face-region masks
* Compute eye/mouth distance fields
* Combine influence maps

The intended pattern is:

* core package remains the source of truth
* ComfyUI wrappers call into the package
* no duplication of core algorithm logic in the wrappers

---

## 16. Future Expansion Areas

This repo should be architected with the expectation that future phases may add:

* additional semantic face regions
* multiple portrait feature groups
* depth-derived maps
* normal-derived maps
* curvature proxies
* relit target density generation
* batch processing
* mesh-aware and geodesic-aware map generation
* downstream line-generation support

---

## 17. Success Definition

This repo will be successful in its first phase if it becomes a clean and reusable environment for developing portrait importance maps, beginning with eye and mouth distance/influence maps, while preserving the flexibility to grow into a broader portrait map-generation system and later serve as the backend for ComfyUI custom nodes.

---

## 18. Developer Handoff Summary

The repo should be built around the following already-made decisions:

* Start with a standalone Python repo
* Use MediaPipe directly first
* Focus initially on 2D image-space distance fields
* Use region masks rather than only eye/mouth center points
* Preserve both raw and remapped outputs
* Design the processing code so it can later be wrapped in ComfyUI custom nodes
* Treat this as the beginning of a broader map-generation framework, not a single throwaway script

The first job is not to solve the whole drawing system.
The first job is to build a solid, extensible, inspectable **portrait map lab**.