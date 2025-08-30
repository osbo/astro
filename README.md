# Astro: High-Performance Barnes-Hut N-Body Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated N-body gravitational simulation written in Swift and Metal, featuring advanced GPU optimizations, custom radix sorting, and optimized octree data structures. This project demonstrates high-performance computing techniques optimized for Apple Silicon Macs, achieving real-time simulation of **500,000+ particles** through algorithmic optimization rather than brute force.

https://github.com/osbo/astro/blob/main/Astro%20new.mov

## Performance Benchmarks

| Metric | Value | Context |
|--------|-------|---------|
| **Particle Count** | 500,000+ | Real-time simulation |
| **Frame Rate** | 120 FPS | Target performance |
| **Algorithm Complexity** | O(n log n) | Barnes-Hut optimization |
| **GPU Memory Efficiency** | Contiguous storage | Zero pointer overhead |
| **Spatial Resolution** | 21 bits per axis | Morton code optimization |

**[BENCHMARK: Performance comparison across different layer configurations]**

## Core Technical Innovations

### Layer-by-Layer Octree Storage

The octree implementation uses a storage pattern where each layer is stored sequentially from leaves to root, with collision space pre-allocated at the end of each layer. Layer offsets and sizes are mathematically pre-calculated based on maximum particle count, allowing each layer to determine the next layer's starting position without pointer traversal.

This design eliminates pointer-based tree structures, keeping all octree data in contiguous memory regions that maximize GPU residency and enable efficient parallel processing across the entire tree structure.

### Custom GPU Radix Sort

The project implements a complete GPU radix sort from scratch, providing full control over the sorting process without relying on Metal Performance Shaders. This custom implementation features:

- **Multi-pass prefix sum algorithm** with auxiliary buffers designed for large datasets
- **Split-based radix sort** supporting 64-bit keys with memory-efficient ping-pong buffers
- **Optimized for Apple Silicon** with 64-thread workgroups for optimal occupancy
- **64 separate passes** processing all bits of Morton codes for complete spatial ordering

This custom implementation was necessary because Metal Performance Shaders don't provide the level of control needed for the specific Morton code sorting requirements of the Barnes-Hut algorithm.

### Morton Code Optimization

The Morton code implementation supports up to 21 bits per spatial direction (63 total bits), providing extremely fine spatial resolution. Testing with particle counts from 10,000 to 500,000 revealed that using only **7 layers (21 bits total)** provides the optimal balance between spatial partitioning and individual particle isolation.

At this depth, most individual particles maintain their own leaf nodes, ensuring maximum accuracy in the Barnes-Hut approximation while minimizing computational overhead. This optimization was determined through empirical testing rather than theoretical analysis.

## Implementation Details

### Barnes-Hut Force Calculation

The force calculation kernel implements a **stack-based traversal algorithm** for optimal performance. Each particle traverses the octree using a depth-first approach, maintaining a stack of nodes to visit. The algorithm applies the theta criterion (θ = 0.7) to determine when to approximate distant particle groups as single masses.

**Key optimizations include:**
- **Early termination** for self-interactions
- **Force clamping** to prevent numerical instability
- **Softening parameters** to handle close particle encounters
- **Stack-based approach** that minimizes memory access patterns

### GPU-Optimized Data Structures

All data structures are designed for optimal GPU caching and memory access patterns:

- **PositionMass struct** combines position and mass data in a single cache line
- **OctreeNode struct** includes all necessary information for force calculations, lighting, and tree traversal in a compact format
- **Shared structs** between Swift and Metal through the bridging header for zero-copy operations
- **Optimized buffer layouts** for coalesced memory access

## Engineering Trade-offs: Performance vs. Visual Fidelity

This project demonstrates deliberate engineering decision-making by prioritizing simulation scale over visual effects. The current version focuses entirely on core simulation performance, achieving 500,000+ particles through algorithmic optimization.

### Previous Version: Visual Fidelity Focus

The previous version prioritized visual fidelity and advanced graphics techniques, showcasing sophisticated rendering capabilities:

https://github.com/osbo/astro/blob/main/Astro%20old.mov

**Advanced Gravitational Effects:**
- **Vertex-shifting gravitational lensing** using definite integrals over gravitational fields
- **Gravitational redshift effects** where light from stars appeared redder in stronger gravitational fields
- **Velocity-based Doppler effects** simulating relativistic corrections for high-speed particles

**Comprehensive Post-Processing Pipeline:**
- **Bloom effects** creating realistic light diffusion around bright objects
- **Film grain** adding subtle noise patterns for organic appearance
- **Chromatic aberration** simulating lens imperfections for enhanced depth perception
- **Custom planet shaders** with atmospheric scattering calculations

### Current Version: Performance Focus

The current version represents a deliberate engineering trade-off, removing advanced graphics features to achieve unprecedented simulation scale. This demonstrates:

- **Clear product vision** and ability to make difficult technical decisions
- **Focus on core value proposition** rather than feature creep
- **Understanding of performance bottlenecks** and optimization priorities
- **Engineering discipline** to remove code that doesn't serve the primary goal

**[BENCHMARK: Performance comparison between legacy and current versions]**

## Current Configuration

The simulation is currently configured for maximum particle count:

- **5 stars** with high mass and luminosity
- **200 planets** with moderate mass
- **500,000 dust particles** for realistic space environment
- **Post-processing disabled** to maximize simulation performance

The configuration can be adjusted in `Renderer.swift` to optimize for different performance targets.

## Summary

This project demonstrates expertise in:

### Technical Skills
- **High-Performance Computing**: Real-time simulation of complex physical systems
- **GPU Architecture Optimization**: Custom algorithms designed for GPU parallel processing
- **Algorithm Design**: Novel approaches to classic computational problems
- **Memory System Optimization**: Contiguous data structures for maximum bandwidth utilization
- **Cross-Platform Development**: Seamless integration between Swift and Metal
- **Performance Profiling**: Empirical optimization based on real-world testing

### Engineering Judgment
- **Engineering Decision-Making**: Ability to make difficult trade-offs between competing technical goals
- **Product Focus**: Prioritizing core value proposition over feature completeness
- **Performance Optimization**: Understanding of bottlenecks and optimization priorities
- **Code Quality**: Discipline to remove code that doesn't serve the primary goal

The combination of algorithmic optimization, GPU optimization, and performance engineering makes this project a compelling demonstration of advanced software engineering skills in high-performance computing and graphics programming.

---

*This project represents a comprehensive implementation of high-performance computing techniques, demonstrating expertise in GPU programming, algorithm optimization, and real-time graphics programming.*
