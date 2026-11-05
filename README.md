# Graph-Based Trait Inheritance Engine (GNN-Lineage)

A high-performance graph neural network (GNN) and message-passing pipeline designed to model complex phenotypic inheritance, multi-allele interactions, and trait covariance across structured lineage graphs. 

This project bridges population genetics with modern geometric deep learning, treating pedigree structures and gene interactions as dynamic, message-passing topological spaces rather than flat tabular data.

---

## Architecture & Core Mechanics

* **Graph Representation:** Lineage trees and generational pedigrees are modeled as directed graphs using `PyTorch Geometric`, where nodes represent individuals/loci and edges encode descent links and linkage disequilibrium (LD) relationships.
* **Message-Passing Engine:** Implements custom Graph Convolutional layers (`GCNConv` / spatial message-passing) to propagate latent genetic states and recurrent trait probabilities across complex multi-generation graphs.
* **Covariance & Prediction Heads:** Evaluates multi-variate feature dependencies to predict latent phenotypic expressions and recessive carrier probabilities without relying on black-box external cloud APIs.

---

## Technical Stack

* **Core Framework:** Python, PyTorch, PyTorch Geometric (`torch_geometric`)
* **Graph Topology & Processing:** NetworkX, SciPy, NumPy
* **Execution:** 100% locally executable on consumer hardware with zero external cloud dependencies or API rate limits.

---

## Project Structure

```text
├── src/
│   ├── data/           # Pedigree graph constructors and adjacency matrix builders
│   ├── models/         # Custom GNN architecture and message-passing layers
│   ├── simulation/     # Stochastic trait inheritance and allele frequency generators
│   └── training/       # Local training loops, evaluation metrics, and loss tracking
├── notebooks/          # Exploratory covariance heatmaps and topology visualizations
└── README.md
