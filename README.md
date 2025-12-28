# RayTracer Engine - Hybrid CUDA/OpenMP Implementation

A high-performance ray tracing engine that leverages hybrid CPU-GPU computing using CUDA and OpenMP for parallel rendering. This project implements a real-time interactive ray tracer with path tracing capabilities.

## What's in the Program

This ray tracer renders 3D scenes using physically-based path tracing. The engine shoots rays from a virtual camera through each pixel, simulating how light bounces off surfaces to create realistic images with soft shadows, color bleeding, and reflections.

The program features:
- **Interactive camera controls** - Orbit around the scene, zoom in/out, and explore from any angle
- **Real-time rendering** - See changes instantly as you move the camera
- **Monte Carlo sampling** - Multiple samples per pixel for smooth, noise-reduced images
- **Gamma correction** - Proper color space handling for realistic output

### Available Scenes

When you launch the program, you can choose between two scene types:

#### Scene 1: Custom Shape Builder
An interactive scene creator where you build your own 3D world:
- Specify the number of objects you want (up to the maximum limit)
- For each object, choose:
  - **Shape**: Sphere or Cube
  - **Material**: Diffuse (matte), Metal (reflective), or Light (emissive)
- Objects are automatically arranged in a grid pattern with slight random offsets
- Light sources become the camera's pivot point for easy viewing
- Great for experimenting with different material combinations

#### Scene 2: Forest Environment
A procedurally generated natural outdoor scene featuring:
- **Trees**: Configurable number (1-15) with brown trunks (cubes) and green canopy (clustered spheres)
- **Ground plane**: Large olive-green sphere simulating terrain
- **Sun/Light source**: Emissive sphere providing scene illumination
- **Logs**: Randomly placed fallen logs on the ground
- **Rocks**: Clusters of grey spheres scattered throughout
- Trees are randomly positioned across the scene for a natural look
- Camera pivots around the first tree for easy scene exploration

## Features

- **Hybrid Rendering**: Combines CUDA GPU acceleration with OpenMP CPU parallelization
- **Real-time Interactive Viewing**: SDL2-based window with mouse and keyboard controls
- **Path Tracing**: Monte Carlo path tracing with multiple samples per pixel
- **Multiple Materials**: Support for Lambertian (diffuse), Metal (reflective), and Emissive materials
- **Geometric Primitives**: Spheres and cubes with efficient intersection testing
- **Tile-based Rendering**: Divides the image into tiles for efficient parallel processing
- **Performance Analyzer**: Built-in tool for benchmarking and comparing different configurations

## Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 5.0 or higher
- Multi-core CPU (for OpenMP parallelization)

### Software
- CUDA Toolkit (nvcc compiler)
- SDL2 library
- OpenMP (typically included with GCC)
- GNU Make

## Installation

### Install Dependencies (Ubuntu/Debian)

```bash
make install-deps
```

Or manually:

```bash
sudo apt-get update
sudo apt-get install libsdl2-dev
```

### Build

Build both the raytracer and performance analyzer:

```bash
make all
```

Or build individually:

```bash
make bin/raytracer        # Build only the raytracer
make bin/hybrid_analyzer  # Build only the performance analyzer
```

## Usage

### Running the Ray Tracer

```bash
make run-raytracer
# or
./bin/raytracer
```

### Controls

| Input | Action |
|-------|--------|
| **Mouse Drag** | Rotate camera around pivot point |
| **Mouse Wheel** | Zoom in/out |
| **W / Up Arrow** | Pitch camera up |
| **S / Down Arrow** | Pitch camera down |
| **A / Left Arrow** | Rotate camera left |
| **D / Right Arrow** | Rotate camera right |
| **+ / =** | Zoom in |
| **-** | Zoom out |
| **F** | Toggle fullscreen |

### Running the Performance Analyzer

```bash
make run-analyzer
# or
./bin/hybrid_analyzer
```

The analyzer benchmarks different CPU thread counts and GPU block configurations, outputting results to `hybrid_performance_analysis.csv`.

## Project Structure

```
.
├── bin/                    # Compiled executables
│   ├── raytracer
│   └── hybrid_analyzer
├── include/                # Header files
│   ├── camera_shared.h     # Camera structure definitions
│   ├── common_math.h       # Vector math operations
│   ├── cuda_utils.h        # CUDA error checking utilities
│   ├── hittables.h         # Hit record and primitive definitions
│   ├── hybrid_render.h     # Hybrid rendering declarations
│   ├── kernels.h           # CUDA kernel declarations
│   ├── materials.h         # Material types and functions
│   └── scene.h             # Scene management
├── src/                    # Source files
│   ├── main.cu             # Main application entry point
│   ├── hybrid_render.cu    # Hybrid CPU/GPU rendering implementation
│   ├── kernels.cu          # CUDA rendering kernels
│   ├── scene.cu            # Scene setup and management
│   └── hybrid_analyzer_cuda.cu  # CUDA wrapper for analyzer
├── obj/                    # Compiled object files
├── singleversion/          # Single-file version reference
├── hybrid_performance_analyzer.cpp  # Performance benchmarking tool
├── hybrid_performance_analysis.csv  # Benchmark results
└── Makefile
```

## Architecture

### Hybrid Rendering Approach

The engine uses a tile-based hybrid rendering approach:

1. **Tile Division**: The frame is divided into 64x64 pixel tiles
2. **GPU Rendering**: CUDA kernels process tiles using multiple streams
3. **CPU Fallback**: OpenMP threads can process tiles when GPU is saturated
4. **Load Balancing**: Dynamic scheduling distributes work between CPU and GPU

### CUDA Configuration

- Default block size: Configurable (tested configurations from 1x1 to 32x32)
- Multiple CUDA streams for concurrent kernel execution
- Stack size set to 16KB for recursive ray tracing

### Performance

Based on benchmark results (800x600, 4 samples per pixel):

| Configuration | Speedup vs Serial |
|---------------|-------------------|
| 1 CPU thread, 8x8 GPU blocks | 2.83x |
| 4 CPU threads, 8x16 GPU blocks | 6.44x |
| 8 CPU threads, 8x8 GPU blocks | 8.47x |

## Building for Different Architectures

Modify the `-arch` flag in the Makefile to target your GPU:

```makefile
NVCC_FLAGS = -std=c++11 -O3 -arch=sm_XX
```

Common values:
- `sm_50`: Maxwell (GTX 900 series)
- `sm_61`: Pascal (GTX 1000 series)
- `sm_75`: Turing (RTX 2000 series)
- `sm_86`: Ampere (RTX 3000 series)
- `sm_89`: Ada Lovelace (RTX 4000 series)

## Cleaning Build Artifacts

```bash
make clean    # Remove obj/ and bin/ directories
make rebuild  # Clean and rebuild everything
```

## License

This project is provided for educational and research purposes.

## Acknowledgments

- SDL2 for window management and rendering
- NVIDIA CUDA Toolkit for GPU acceleration
- OpenMP for CPU parallelization
