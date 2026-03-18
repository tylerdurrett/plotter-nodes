# Vision & Roadmap

## Overall Repo Goal

This repo is a **Python-based map and analysis lab** that supports a broader creative-coding practice centered on generative drawing for a plotter environment maintained in a separate TypeScript repo.

Its purpose is to help develop and test the **upstream visual intelligence** needed for plotter-oriented algorithms: portrait importance maps, density targets, structural guidance maps, geometry-derived fields, and other analysis layers that can later drive line, shading, and related mark-making systems.

This repo is **not** the final plotter runtime or drawing engine. Instead, it is a research and prototyping environment for building the inputs and supporting logic that more advanced creative-code algorithms will depend on. Over time, outputs from this repo may be wrapped as ComfyUI nodes and/or exported for use in the separate TypeScript plotter system.

## Relationship to the Larger Plotter System

The larger goal is to build a family of **creative code drawing algorithms** for a plotter environment, including but not limited to:

* Continuous-line portrait rendering
* Density-based pen shading systems
* Form-following stroke systems
* Geometry-aware drawing behaviors
* Other experimental plotting algorithms beyond line drawing

This Python repo exists to make it easier to explore and develop the kinds of **maps, signals, and derived representations** that those algorithms need, especially when those representations are easier to prototype in Python using tools such as MediaPipe, image-processing libraries, scientific Python workflows, and eventually geometry tooling.

The TypeScript plotter repo remains the home for the actual plotter-oriented drawing environment. This repo supports that work by generating reusable analysis outputs and by helping define the logic that future drawing systems will consume.

## Specific Algorithm Direction: Continuous Line Portrait System

One of the major target algorithms for the broader plotter environment is a **continuous-line portrait drawing system**.

The high-level idea is to generate a line, or conceptually a moving particle leaving behind a line, that builds a portrait through accumulated mark density. The line should not only recreate light and shadow, but should also feel tied to the 3D structure of the subject, as if it is flowing around or wrapping across the form.

This system is intended for pen plotting, so it must balance several goals at once:

* Represent tone through accumulated line activity
* Preserve important facial structure and recognizable features
* Follow the form in a visually meaningful way
* Remain coherent as a drawable plotting path
* Eventually support SVG or other vector-like output suitable for the plotter workflow

The current Python repo is focused on the **early-stage supporting layers** for that system rather than the final line generator itself.

## Development Stages

### Stage 1: Build portrait analysis and importance maps

Create the foundational maps that describe what parts of the image or subject matter most and how importance or density should be distributed.

Examples include:

* Eye and mouth influence maps
* Face contour maps
* Future silhouette maps
* Future density target maps
* Future geometry-derived maps from depth, normals, or reconstructed surface data

**This is the current focus of the repo.**

### Stage 2: Compose higher-level target and guidance maps

Move from isolated maps to **composed map systems** that combine:

* Focal feature importance
* Structural support
* Future tonal goals
* Future gating or masking logic

These combined maps will act as higher-level targets or constraints for later drawing algorithms.

### Stage 3: Develop direction and flow fields

Create maps or fields that describe how a line should want to move across the image or surface.

These may eventually incorporate:

* Image structure
* Face contour logic
* Depth/normal-derived structure
* Curvature-like cues
* Projected or surface-based flow directions

This stage is where the system begins shifting from "what is important" to "how the line should travel."

### Stage 4: Build accumulation and density logic

Define how drawn line segments deposit visual density.

This stage will model how repeated passes, spacing, and overlap create darker or lighter regions, so the eventual line generator can compare:

* Desired density
* Already deposited density
* Remaining residual

This provides the feedback loop needed for controlled mark-making.

### Stage 5: Generate the line behavior

Develop the actual continuous-line algorithm that uses the maps and fields from earlier stages.

The intended behavior is a line that:

* Prefers important regions where needed
* Follows structural or geometric guidance
* Accumulates more in darker or higher-priority regions
* Remains visually coherent and plotter-friendly

This may be implemented initially in 2D image space and later expanded toward more geometry-aware behavior.

### Stage 6: Promote toward full geometric / 3D awareness

As depth maps, normals, and reconstructed 3D geometry become part of the pipeline, the algorithm can become more surface-aware.

This is the stage where the "line wrapping around form" idea becomes more literal, potentially using reconstructed geometry and more advanced visibility or surface-distance logic.

### Stage 7: Integrate with the plotter environment

Once the map logic and line behavior are mature enough, the relevant outputs and algorithmic ideas can be transferred or adapted into the separate TypeScript plotter repo.

At that point, this Python repo continues to function as:

* A research lab
* A map-generation backend
* A place for prototyping new visual-analysis techniques
* A possible source for ComfyUI node wrappers
