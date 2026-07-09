![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations

---

# Headliner

## 📌 Project Summary
This project extends a graph based circuit diagram retrieval framework by improving the scalability of the retrieval pipeline. The original framework converts circuit diagrams into graph representations and performs retreval using Graph Edit Distance (GED) to measure similarity. This achieves high accuracy, but the computational demands of GED limits the scalability for larger and more complex databases. This research aims to investigate alternative strategies to perserve retreival quality and accuracy while reduciing the computational cost.

## 🎯 Motivation
Modern circuit repositories continue to grow as engineer reuse existing designs, search for prior work, and verify potential plagerisim conflicts. Unlike regular images, circuit diagmras contain complex structural relationships, making graph representations an appropriate retrevial solution rather than pure image based approaches.

The original paper demonstrates that graph based retreival outperforms traditional image techniques by preserving circuit topology. However, because the retrieval system relies on GED, a method for finding how similar two graphs are by finding the number of edits reqruired to transform one graph to another (an NP hard problem), this introduces expensive computation that introduces a bottleneck when scaling to more complex graphs. This provided motivation for this project to develop a practical graph based circuit retrieval without sacrificing accuracy. 

## 🧩 Novelty
Although there have been numerous studies proposing graph similarity models and applying them to neural networks and electronic design, to my knowledge no work has evaluated modern graph similarity methods specifically for circuit diagram retreival and instead primarily use GED. This project aims to bridge the two areas by implementing different graph similarity methods for circuit diagram retreival.

## 🧠 Methodology
1. **Dataset**: No dataset yet
2. **Architecture**: Compare multiple graph similarity methods
   - Base: Graph Edit Distance (GED)
   - SimGNN
   - GraphSim
   - ISONET
   - Additional graph embedding/transformer methods
3. **Evaluation**: Each method will be evaluated using the same datasets and tasks.
   - Measure retrieval latency and scalability as size increases
   - Computational cost (runtime and memory usage)
   - Retrieval quality and accuracy
4. **Metrics**:
   - Precision@K
   - Average retrieval latency
   - Memory usage
   - Scalability

## 🌍 Impact
Efficient graph-based circuit retrieval is useful in electrical design automation, patent analysis, reverse engineering, and reusing engineering knowledge. Reducing retrieval time while keeping structural accuracy would allow engineers to search much larger circuit databases, improve the reuse of old designs, speed up design verification, and support large circuit repositories where exact graph matching is not practical. Additionally, the findings may apply to other graph retrieval problems where structural similarity matters more than visual appearance.

#### Future Work
GED is only one area that has potential for improvement. 
- Circuit components could be analyzed to measure how they contribute to circuit functionality so future algorithims can be developed to weigh circuit components according to their importance. 
- Currently, graph representations are undirected, ignore edge labels, and use only node types as lables. This could be extended to involve directed graphs and more information about the circuit such as pin type information, component values, hierarchical circuit information, etc.
- This framework could be extended to support multimodal retreival across circuit diagrams, netlists, and text circuit descriptions.

**Additional Sources:** https://arxiv.org/pdf/2503.11658