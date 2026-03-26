# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VectorScale is a RetroArch Slang shader that implements the Kopf-Lischinski pixel-art vectorization algorithm. It converts pixel art into smooth vector boundaries with analytical anti-aliasing, running entirely on the GPU as a 10-pass fragment shader pipeline. There is no build system, test suite, or application code — the deliverable is the shader preset file and its associated `.slang` shaders.

## How to Test

Load `retroarch-shaders/vibeboy-vectorize.slangp` as a shader preset in RetroArch with a Vulkan, Metal, or Direct3D 12 backend. All testing is visual — run pixel art content and inspect the output for correct boundary smoothing, color accuracy, and anti-aliasing artifacts.

## Architecture

### Pipeline (9 passes, defined in `vibeboy-vectorize.slangp`)

All shaders are in `retroarch-shaders/shaders/`. Data flows linearly through passes, with later passes reading earlier outputs via RetroArch texture aliases.

1. **similarity-graph** — Builds a 2x-scaled connectivity graph encoding pixel colors (odd,odd), horizontal edges (even,odd), vertical edges (odd,even), and diagonal corner flags (even,even) as bitfields.
2. **resolve-crossings** — Resolves diagonal crossings where both diagonals are present, using curve-length, sparse-pixel BFS, and island heuristics to vote.
3. **cell-graph** — Creates B-spline control points (CPs) at grid corners. Each CP has 3 texture rows: position+flags+directions, prev neighbor, next neighbor. Handles diagonal splits (2 CPs per corner), T-junctions, and valence-4 nodes.
4. **init-positions** — Initializes the position texture from cell graph original positions.
5. **optimize-energy** (x2) — 2D Newton-Raphson minimizing curvature + positional energy with analytical gradient and Hessian (3 iterations). Stores absolute positions in F32 framebuffers.
6. **update-tjunction** — Weighted average smoothing for T-junction CPs and their stems.
7. **cell-rasterizer** — Finds nearest B-spline boundary per output pixel via Newton-Raphson, resolves color from edge data, applies analytical AA. Multi-hit resolution with single-neighbor endpoint rejection.
8. **passthrough** — Identity sample for correct RetroArch viewport mapping.

### Key Data Encoding Conventions

- **Similarity graph** (passes 0-1): 2W×2H texture. Parity of (x,y) determines content type. Diagonal flags are 2-bit integers stored as float, decoded with `uint(val + 0.5)`.
- **Cell graph** (pass 2): Width = W+1 (corners), height = 6×(H+1) (3 rows × 2 slots per corner). Flags and direction packs are stored as raw floats in F32 textures, decoded with `uint(val + 0.5)`. Neighbor indices encoded as `(cx, cy, slot)`.
- **Positions** (passes 3-6): Width = W+1, height = 2×(H+1) (2 slots per corner). Stores absolute `(x, y)` positions in F32 framebuffers.
- Texture sizes are over-allocated using source-relative scales in the `.slangp` file. The shaders guard against out-of-bounds reads.

### Directions Convention

Cardinal directions are encoded as integers: 0=N, 1=E, 2=S, 3=W. The "left" and "right" edge colors are defined relative to the direction of travel along a boundary (see `get_edge_colors`). The rasterizer swaps left/right for the next-edge because the tangent reverses at t=1.

### CP Flags

- `1` = pinned (border or endpoint)
- `2` = active interior (has both prev and next neighbors)
- `16` (IS_CORNER) = sharp corner detected by angle heuristic
- `32` (IS_TJUNCTION) = T-junction node
