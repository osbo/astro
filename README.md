# Astro: High-Performance Barnes-Hut N-Body Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated N-body gravitational simulation written in Swift and Metal, featuring advanced GPU optimizations, custom radix sorting, and innovative octree data structures. This project demonstrates high-performance computing techniques optimized for Apple Silicon Macs.

## Performance Highlights

- **500,000+ particles** with real-time simulation
- **O(n log n)** complexity via Barnes-Hut algorithm
- **Custom GPU radix sorting** without Metal Performance Shaders
- **Optimized octree storage** for maximum GPU residency
- **120 FPS target** with advanced lighting effects

## Architecture Overview

### Core Simulation Engine

The simulation uses the **Barnes-Hut algorithm** to reduce computational complexity from O(n²) to O(n log n). The implementation features:

- **Morton Code-based spatial partitioning** for efficient octree construction
- **Custom GPU radix sorting** with prefix sums for optimal memory access patterns
- **Layer-by-layer octree construction** with collision-aware storage allocation
- **Floating-point origin system** using double precision for large-scale simulations

### Innovative Octree Storage Pattern

The octree implementation uses an unconventional storage pattern to maximize GPU residency. Rather than using traditional pointer-based tree structures, each layer is stored sequentially from leaves to root, with collision space reserved at the end of each layer. This design allows each layer to mathematically determine the next layer's starting position, even with collisions and empty cells, eliminating the need for pointer-based tree structures.

The layer offsets and sizes are pre-calculated based on the maximum expected particle count, ensuring optimal memory allocation and access patterns. This approach maximizes GPU residency by keeping all octree data in contiguous memory regions, enabling efficient parallel processing across the entire tree structure.

### Morton Code Optimization

The Morton code implementation supports up to 21 bits per spatial direction (63 total bits), providing extremely fine spatial resolution. However, extensive testing with various particle counts revealed that using only 7 layers (21 bits total) provides the optimal balance between spatial partitioning and individual particle isolation. At this depth, most individual particles maintain their own leaf nodes, ensuring maximum accuracy in the Barnes-Hut approximation while minimizing computational overhead.

This optimization was determined through empirical testing with particle counts ranging from 10,000 to 500,000, where 7 layers provided the best performance characteristics for the target simulation scale.

### GPU-Optimized Data Structures

All data structures are carefully designed for optimal GPU caching and memory access patterns. The PositionMass struct combines position and mass data in a single cache line, while the OctreeNode struct includes all necessary information for force calculations, lighting, and tree traversal in a compact format. These structures are shared between Swift and Metal through the bridging header, ensuring zero-copy operations and optimal memory bandwidth utilization.

## Technical Implementation

### Custom Radix Sort Implementation

The project implements a complete GPU radix sort from scratch, providing full control over the sorting process without relying on Metal Performance Shaders. This custom implementation features a multi-pass prefix sum algorithm with auxiliary buffers designed specifically for large datasets. The split-based radix sort supports 64-bit keys and uses memory-efficient ping-pong buffers for in-place sorting operations.

The implementation is optimized for Apple Silicon with 512-thread workgroups, providing optimal occupancy and memory bandwidth utilization. The radix sort processes all 64 bits of the Morton codes in 64 separate passes, ensuring complete spatial ordering for optimal octree construction.

### Barnes-Hut Force Calculation

The force calculation kernel implements a sophisticated stack-based traversal algorithm for optimal performance. Each particle traverses the octree using a depth-first approach, maintaining a stack of nodes to visit. The algorithm applies the theta criterion to determine when to approximate distant particle groups as single masses, significantly reducing computational complexity while maintaining physical accuracy.

The kernel includes advanced optimizations such as early termination for self-interactions, force clamping to prevent numerical instability, and softening parameters to handle close particle encounters. The stack-based approach minimizes memory access patterns and maximizes GPU occupancy during force calculations.

### Advanced Lighting System

The current version prioritizes maximum particle count and simulation performance, but the codebase retains the infrastructure for advanced lighting effects from the legacy version. The lighting system includes real-time calculations from all light sources, implementing a Blinn-Phong shading model for realistic surface lighting. While post-processing effects such as bloom, chromatic aberration, and film grain are currently disabled to maximize particle count, the underlying shader infrastructure remains intact for future use.

## Performance Optimizations

### Memory Management

- **Shared memory buffers** between Swift and Metal for zero-copy operations
- **Memoryless depth textures** for efficient depth testing
- **Private storage buffers** for GPU-only data
- **Optimized buffer layouts** for coalesced memory access

### Compute Pipeline Optimization

- **64-thread workgroups** for optimal occupancy
- **Memory barriers** for correct synchronization
- **Pipeline state caching** to minimize state changes
- **Instance rendering** for efficient sphere drawing

### Algorithmic Optimizations

- **Theta criterion** (θ = 0.7) for Barnes-Hut approximation
- **Softening parameter** to prevent numerical instability
- **Force clamping** to prevent extreme accelerations
- **Early termination** for self-interactions

## User Interface

The simulation features an interactive camera system:

- **WASD movement** for camera positioning
- **Mouse look** for camera orientation
- **Real-time parameter adjustment** for simulation tuning
- **Floating-point origin** system for large-scale exploration

## Legacy Graphics Features

The previous version of this simulation prioritized visual fidelity and advanced graphics techniques over particle count. This legacy version showcased sophisticated rendering techniques that demonstrated the full capabilities of modern GPU graphics programming.

### Advanced Gravitational Effects

The legacy version implemented physically-based gravitational effects that went beyond simple Newtonian gravity. Vertex-shifting gravitational lensing used definite integrals over gravitational fields to simulate the bending of light around massive objects, creating realistic lensing effects without requiring ray tracing. The simulation also included gravitational redshift effects, where light from stars appeared redder when viewed from regions of stronger gravitational fields, and velocity-based Doppler effects that simulated relativistic corrections for high-speed particles.

### Comprehensive Post-Processing Pipeline

The legacy version featured a full post-processing pipeline designed to achieve cinematic visual quality. Bloom effects created realistic light diffusion around bright objects, while film grain added subtle noise patterns for a more organic appearance. Chromatic aberration simulated lens imperfections, creating color fringing effects that enhanced the sense of depth and realism. Custom planet shaders included atmospheric scattering calculations, creating realistic atmospheric effects around planetary bodies.

These advanced graphics features were disabled in the current version to prioritize simulation performance and maximum particle count, but the underlying shader infrastructure and mathematical implementations remain in the codebase for future reactivation.

## Build and Run

### Requirements
- macOS 12.0+
- Xcode 14.0+
- Apple Silicon Mac (M1/M2/M3) for optimal performance

### Building
```bash
git clone <repository>
cd Astro
open Astro.xcodeproj
# Build and run in Xcode
```

### Configuration

The simulation parameters can be adjusted in the Renderer.swift file to optimize for different performance targets. The current configuration prioritizes maximum particle count with 5 stars, 200 planets, and 500,000 dust particles. Post-processing effects are disabled to maximize simulation performance, but can be re-enabled by setting the usePostProcessing flag to true for enhanced visual quality at the cost of particle count.

## Technical Achievements

This project demonstrates:

1. **Advanced GPU Programming**: Custom Metal compute kernels without relying on Metal Performance Shaders
2. **Algorithm Optimization**: Barnes-Hut implementation with innovative storage patterns
3. **Memory Efficiency**: Optimized data structures for GPU caching and bandwidth utilization
4. **Real-time Performance**: 500,000+ particle simulation at interactive frame rates
5. **Cross-platform Design**: Shared structs in bridging headers for Swift/Metal interoperability

## Future Enhancements

- **Multi-GPU support** for even larger simulations
- **Adaptive time stepping** based on particle density
- **Particle collision detection** and response
- **Real-time parameter adjustment** UI
- **Export capabilities** for scientific analysis

---

*This project represents a comprehensive implementation of high-performance computing techniques, demonstrating expertise in GPU programming, algorithm optimization, and real-time graphics programming.*
