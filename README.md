# Astro: High-Performance Barnes-Hut N-Body Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated N-body gravitational simulation written in Swift and Metal, featuring advanced GPU optimizations, custom radix sorting, and innovative octree data structures. This project demonstrates high-performance computing techniques optimized for Apple Silicon Macs, achieving real-time simulation of **500,000+ particles** through algorithmic innovation rather than brute force.

## 🚀 Performance Highlights

- **500,000+ particles** with real-time simulation at interactive frame rates
- **O(n log n)** complexity via Barnes-Hut algorithm with custom optimizations
- **Custom GPU radix sorting** without Metal Performance Shaders
- **Optimized octree storage** for maximum GPU residency
- **120 FPS target** with advanced lighting effects

## 🔬 What Makes This Unique

This project represents a complete reimagining of N-body simulation architecture, moving beyond traditional approaches to achieve unprecedented performance. Rather than simply implementing the Barnes-Hut algorithm, I developed several key innovations that make this simulation possible at scale:

### 🏗️ Innovative Octree Storage Architecture

The most significant technical achievement is the **layer-by-layer octree construction** with collision-aware storage allocation. Traditional octree implementations use pointer-based tree structures that scatter data across memory, causing poor GPU cache performance. This implementation uses a revolutionary storage pattern where:

- Each octree layer is stored sequentially from leaves to root
- Collision space is pre-allocated at the end of each layer
- Layer offsets and sizes are mathematically pre-calculated based on maximum particle count
- Each layer can determine the next layer's starting position without pointer traversal

This design eliminates the need for pointer-based tree structures entirely, keeping all octree data in contiguous memory regions that maximize GPU residency and enable efficient parallel processing across the entire tree structure.

### 🔄 Custom GPU Radix Sort Implementation

The project implements a complete GPU radix sort from scratch, providing full control over the sorting process without relying on Metal Performance Shaders. This custom implementation features:

- **Multi-pass prefix sum algorithm** with auxiliary buffers designed for large datasets
- **Split-based radix sort** supporting 64-bit keys with memory-efficient ping-pong buffers
- **Optimized for Apple Silicon** with 64-thread workgroups for optimal occupancy
- **64 separate passes** processing all bits of Morton codes for complete spatial ordering

This custom implementation was necessary because Metal Performance Shaders don't provide the level of control needed for the specific Morton code sorting requirements of the Barnes-Hut algorithm.

### 📊 Morton Code Optimization Through Empirical Testing

The Morton code implementation supports up to 21 bits per spatial direction (63 total bits), providing extremely fine spatial resolution. However, extensive testing with particle counts from 10,000 to 500,000 revealed that using only **7 layers (21 bits total)** provides the optimal balance between spatial partitioning and individual particle isolation.

At this depth, most individual particles maintain their own leaf nodes, ensuring maximum accuracy in the Barnes-Hut approximation while minimizing computational overhead. This optimization was determined through empirical testing rather than theoretical analysis, demonstrating the importance of performance profiling in algorithm design.

## 🏛️ Technical Architecture

### Core Simulation Engine

The simulation uses the **Barnes-Hut algorithm** to reduce computational complexity from O(n²) to O(n log n). The implementation features:

- **Morton Code-based spatial partitioning** for efficient octree construction
- **Custom GPU radix sorting** with prefix sums for optimal memory access patterns
- **Layer-by-layer octree construction** with collision-aware storage allocation
- **Floating-point origin system** using double precision for large-scale simulations

### Barnes-Hut Force Calculation

The force calculation kernel implements a sophisticated **stack-based traversal algorithm** for optimal performance. Each particle traverses the octree using a depth-first approach, maintaining a stack of nodes to visit. The algorithm applies the theta criterion (θ = 0.7) to determine when to approximate distant particle groups as single masses.

**Key optimizations include:**
- **Early termination** for self-interactions
- **Force clamping** to prevent numerical instability
- **Softening parameters** to handle close particle encounters
- **Stack-based approach** that minimizes memory access patterns

### GPU-Optimized Data Structures

All data structures are carefully designed for optimal GPU caching and memory access patterns:

- **PositionMass struct** combines position and mass data in a single cache line
- **OctreeNode struct** includes all necessary information for force calculations, lighting, and tree traversal in a compact format
- **Shared structs** between Swift and Metal through the bridging header for zero-copy operations
- **Optimized buffer layouts** for coalesced memory access

## 💡 What This Demonstrates

### Advanced GPU Programming
- Custom Metal compute kernels without relying on Metal Performance Shaders
- Sophisticated memory management and buffer optimization
- Parallel algorithm design optimized for GPU architecture

### Algorithm Innovation
- Novel octree storage pattern that eliminates pointer-based tree structures
- Empirical optimization of spatial partitioning parameters
- Custom radix sort implementation tailored to specific use case

### Performance Engineering
- Real-time simulation of 500,000+ particles through algorithmic innovation
- Memory bandwidth optimization through contiguous data structures
- GPU cache efficiency through careful data layout design

### Cross-Platform Design
- Shared structs in bridging headers for Swift/Metal interoperability
- Zero-copy operations between CPU and GPU memory
- Efficient data flow between different programming paradigms

## 🎨 Legacy Graphics Features

The previous version of this simulation prioritized visual fidelity and advanced graphics techniques over particle count. This legacy version showcased sophisticated rendering techniques that demonstrated the full capabilities of modern GPU graphics programming.

### Advanced Gravitational Effects

The legacy version implemented physically-based gravitational effects that went beyond simple Newtonian gravity:

- **Vertex-shifting gravitational lensing** using definite integrals over gravitational fields
- **Gravitational redshift effects** where light from stars appeared redder in stronger gravitational fields
- **Velocity-based Doppler effects** simulating relativistic corrections for high-speed particles

### Comprehensive Post-Processing Pipeline

The legacy version featured a full post-processing pipeline designed to achieve cinematic visual quality:

- **Bloom effects** creating realistic light diffusion around bright objects
- **Film grain** adding subtle noise patterns for organic appearance
- **Chromatic aberration** simulating lens imperfections for enhanced depth perception
- **Custom planet shaders** with atmospheric scattering calculations

These advanced graphics features were disabled in the current version to prioritize simulation performance and maximum particle count, but the underlying shader infrastructure and mathematical implementations remain in the codebase for future reactivation.

## ⚙️ Current Configuration

The simulation is currently configured for maximum particle count:

- **5 stars** with high mass and luminosity
- **200 planets** with moderate mass
- **500,000 dust particles** for realistic space environment
- **Post-processing disabled** to maximize simulation performance

The configuration can be adjusted in `Renderer.swift` to optimize for different performance targets or to re-enable advanced graphics features.

## 🏆 Technical Achievements Summary

This project demonstrates expertise in:

1. **High-Performance Computing**: Real-time simulation of complex physical systems
2. **GPU Architecture Optimization**: Custom algorithms designed for GPU parallel processing
3. **Algorithm Design**: Novel approaches to classic computational problems
4. **Memory System Optimization**: Contiguous data structures for maximum bandwidth utilization
5. **Cross-Platform Development**: Seamless integration between Swift and Metal
6. **Performance Profiling**: Empirical optimization based on real-world testing

The combination of algorithmic innovation, GPU optimization, and performance engineering makes this project a compelling demonstration of advanced software engineering skills in high-performance computing and graphics programming.

---

*This project represents a comprehensive implementation of high-performance computing techniques, demonstrating expertise in GPU programming, algorithm optimization, and real-time graphics programming.*
