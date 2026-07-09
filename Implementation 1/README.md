![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations
---

# Emperical Evaluation of Spacial Locality in CSR and Pointer Based Graph Structures

## 📌 Project Summary
This project studies how different memory layouts affect CPU cache performance during graph traversal by analyzing cache hits/miss 
depending on the vertex degree. I compared a pointer-based vs Compressed Sparse Row (CSR) based adjacency list during a BFS (and eventually DFS) search on both a uniform and power law graph. Rather than treating the graph as uniform, this project evaluates memory layout performance on a vertex degree basis to determine whether performance improvements derive from the graph as a whole or primarily from a small number of high degree vertices, and measures the percentage improvement that the degree of a vertex is responsible for.  

## 🎯 Motivation
It is well established that CSR is a more cache efficient layout compared to traditional pointer based adjacency lists. CSR stores data contiguously, meaning data is stores in an uninterrupted block in memory without using pointers. Pointer-based on the other hand, stores data randomly across memory, resulting in frequent cache misses. Most real world graphs such as social networks or internet infrastructure are typically not uniform and instead follow a power law distribution, where a small number of verticies have significantly more connections compared to the rest of the graph. Traditional benchmarking for graph traversal evaluates efficiency by aggregating metrics such as total execution time or cache miss rates. While these measurments demonstrate that CSR is faster, they do not reveal where performance is gained in the graph on a vertex level. This motivated this project to study if CSR's locality advantage is distributed evenly across all verticies among both a uniform and power law graph, or if it is driven by a small number of high degree verticies.

## 🧩 Novelty
Prior work has shown that cache miss rate can be measured as a function of vertex degree, and this has been used to explain how graph reordering algorithms improve locality. Other studies have designed systems around the assumption that high degree vertices are responsible for most of the cache behavior in power law graphs, and use this assumption to justify storing vertices differently based on degree. However, degree based measurement has not been applied to memory layout itself. To my knowledge, no existing work directly measures how much of the performance gap between CSR and pointer based adjacency lists is attributable to high degree vertices, as opposed to being spread evenly across the graph. This project applies degree based cache analysis to that comparison directly, to determine whether the layout assumptions made by existing systems are supported by a measurable effect.

## 🧠 Methodology
1. **Dataset**: Uses the [X](link here) dataset of x+
2. **Architecture**: Specific model used
   - Generate two graph topologies with 1k veritices each (Uniform and power law)
   - Build logically equivalant graph representations adjacency lists using CSR and pointer based 
   - BFS traversal is ran and each traversal is logged with layout used, degree of vertex being processed, memory behavior, spatial lcoality proxy, and cache-miss measurement
   - Every access is assigned a low/medium/high degree bucket based on degree of vertex
   - Results are compared by analyzing memory jump for each degree bucket
   - Future methodolodgy needs to be refined

3. **Evaluation**:
   - 
   - 
   - 
4. **Metrics**:
   - power_law_graph_results, unifor_graph_results

#### Additional Methodology:
- **Something optional**: Sentence

## 🌍 Impact
What is the impact of your research project?

#### Future Work
- **Something optional**: Sentence

**Additional Sources:**
- Optional
