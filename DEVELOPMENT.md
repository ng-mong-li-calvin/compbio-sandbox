# Development Journal & Design Spec: GNN Lineage Engine

This document tracks the architectural decisions, math-to-code translations, and personal learning milestones for the GNN Lineage project.

---

## 🎯 Personal Engineering Objectives
* [ ] Master `PyTorch Geometric` (`torch_geometric`) data structures (`Data`, `DataLoader`, message passing).
* [ ] Implement a clean, modular stochastic simulation pipeline for multi-generational pedigree trees.
* [ ] Bridge population genetics (allele frequencies, linkage covariance) with geometric deep learning.
* [ ] Keep the entire pipeline locally executable on a consumer GPU with zero cloud/API dependencies.

---

## 🗺️ Implementation Roadmap

### Phase 1: Stochastic Pedigree & Trait Simulation (`/src/simulation`)
* **Goal:** Generate synthetic generational data with known ground-truth covariance and recessive traits.
* **Tasks:**
  * [ ] Build a pedigree graph generator (defining parent-child directed edges, population size, and generation depth).
  * [ ] Implement multiallelic inheritance rules (tracking dominant/recessive allele propagation).
  * [ ] Add environmental noise and trait covariance functions to simulate complex polygenic phenotypes.

### Phase 2: Graph Topology & Tensors (`/src/data`)
* **Goal:** Convert raw pedigree data into PyTorch-ready geometric graph structures.
* **Tasks:**
  * [ ] Define node feature vectors (e.g., historical traits, generation depth, marker states).
  * [ ] Construct edge indices (`edge_index` tensor format required by `torch_geometric`).
  * [ ] Write a custom `Dataset` wrapper to batch multi-generation graphs for training.

### Phase 3: GNN Architecture & Message-Passing (`/src/models`)
* **Goal:** Build the neural network that learns how traits propagate through lineage.
* **Tasks:**
  * [ ] Implement a baseline Graph Convolutional Network (`GCNConv` or `SAGEConv`) layer stack.
  * [ ] Design custom prediction heads for phenotype classification and recessive carrier probability regression.
  * [ ] Set up the local training loop with loss tracking (MSE / Cross-Entropy) using standard PyTorch optimizers.

### Phase 4: Evaluation & Visualization (`/notebooks`)
* **Goal:** Verify whether the model actually learned genetic patterns or is just fitting noise.
* **Tasks:**
  * [ ] Plot predicted vs. actual trait covariance heatmaps.
  * [ ] Analyze feature attribution to see how heavily the GNN relies on parent nodes vs. distant ancestors.

---

## 🧠 Design Decisions & Trade-Offs

* **Why PyTorch Geometric over raw NetworkX for training?** 
  * While `NetworkX` is great for topological graph manipulation and visualization, `PyTorch Geometric` integrates natively with PyTorch tensor operations, enabling GPU acceleration and efficient mini-batching for message-passing layers.
* **Why synthetic simulation over real-world VCF files (for now)?**
  * Real-world genomic VCF files carry massive parsing overhead, missing data, and complex batch effects that can derail a project before the core architecture is built. Starting with a controlled stochastic simulator lets us validate the math first.

---

## 📝 Debugging & Notes Log
* *[Date] - Note on edge indexing:* Make sure directed edges point from parent to child so message passing flows forward through generations correctly.
