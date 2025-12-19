# Pairwise-TRIX: Local Pairwise Entity Interaction for Relation Graph Reasoning

## 1. Background and Motivation

TRIX models relation interactions by constructing a relation graph whose edges are induced by shared entities.
Specifically, TRIX refines relation representations by aggregating contributions from individual entities and their roles
(head/tail) in relation occurrences.

While TRIX significantly improves expressivity over relation-only models, it treats entity contributions independently.
This independent aggregation limits its ability to capture **joint relational patterns** that arise from **co-occurrence
and coordination among multiple entities**.

In many knowledge graph scenarios, relational semantics depend not only on which entities participate,
but also on **which entities participate together**.
To address this limitation, we extend TRIX with **local pairwise entity interactions**, introducing a minimal higher-order
structural refinement.

---

## 2. Notation

Let:

- \( \mathcal{G} = (\mathcal{V}, \mathcal{R}, \mathcal{E}) \) be a knowledge graph
- \( v \in \mathcal{V} \): entity
- \( r \in \mathcal{R} \): relation
- Each triple \( (h, r, t) \in \mathcal{E} \)

We follow TRIX and distinguish entity roles:
- head role (H)
- tail role (T)

Let:
- \( \mathbf{x}_r^{(k)} \in \mathbb{R}^d \): relation embedding at layer \( k \)
- \( \mathbf{x}_v^{(k)} \in \mathbb{R}^d \): entity embedding at layer \( k \)

---

## 3. Review: Relation Graph Construction in TRIX

TRIX constructs a **relation graph** where relations are nodes.
Edges between relations are induced by shared entities.

### 3.1 Entity–Relation Incidence Matrices

For each entity \( v \), TRIX defines four incidence indicators:
- \( E_{HH}(v, r_i, r_j) \)
- \( E_{HT}(v, r_i, r_j) \)
- \( E_{TH}(v, r_i, r_j) \)
- \( E_{TT}(v, r_i, r_j) \)

Each indicator specifies whether entity \( v \) participates in relations \( r_i \) and \( r_j \) with corresponding roles.

### 3.2 Relation Adjacency Tensor

The TRIX relation adjacency tensor is defined as:

\[
A^{\text{TRIX}}_{r_i r_j \alpha} = \sum_{v \in \mathcal{V}} E_{\alpha}(v, r_i, r_j)
\]

where \( \alpha \in \{HH, HT, TH, TT\} \).

This formulation aggregates **independent entity contributions**.

---

## 4. Limitation of Independent Entity Aggregation

The above formulation cannot distinguish between:

- multiple entities participating independently, and
- multiple entities consistently co-occurring as a group

since all entity contributions are aggregated independently and symmetrically.

Formally, the aggregation is invariant to any permutation of entity indices and discards **entity co-occurrence structure**.

---

## 5. Pairwise-TRIX: Modeling Local Pairwise Entity Interactions

### 5.1 Core Idea

Instead of treating entities independently, we explicitly model **pairwise entity co-occurrence** in relation interactions.

We introduce **entity pairs** as intermediate structural units, capturing whether two entities jointly participate in a
relation pair \( (r_i, r_j) \).

This corresponds to a **local second-order (2-WL) refinement**, while preserving inductive and equivariant properties.

---

## 6. Pairwise Entity Construction

For each relation pair \( (r_i, r_j) \), define:

\[
\mathcal{V}_{ij} = \{ v \mid v \text{ participates in both } r_i \text{ and } r_j \}
\]

We construct a set of **local entity pairs**:

\[
\mathcal{P}_{ij} = \{ (v_a, v_b) \mid v_a, v_b \in \mathcal{V}_{ij},\; a < b \}
\]

To control complexity, only **local pairs within the same relation interaction** are considered.
No global \( O(|\mathcal{V}|^2) \) enumeration is performed.

---

## 7. Pairwise Relation Adjacency

### 7.1 Pairwise Incidence Function

For each entity pair \( (v_a, v_b) \), define a pairwise incidence:

\[
P_{\alpha}((v_a, v_b), r_i, r_j) = E_{\alpha}(v_a, r_i, r_j) \cdot E_{\alpha}(v_b, r_i, r_j)
\]

This indicates that **both entities jointly contribute** to the same role-specific relation interaction.

---

### 7.2 Pairwise Relation Adjacency Tensor

We define the pairwise relation adjacency as:

\[
A^{\text{pair}}_{r_i r_j \alpha} =
\sum_{(v_a, v_b) \in \mathcal{P}_{ij}} P_{\alpha}((v_a, v_b), r_i, r_j)
\]

This tensor captures **co-occurrence patterns of entity pairs**, which are invisible to TRIX.

---

## 8. Pairwise Message Passing

### 8.1 Message Computation

For each relation \( r_i \), messages from neighboring relations \( r_j \) are computed as:

\[
\mathbf{m}_{r_i}^{(k)} =
\sum_{r_j \in \mathcal{N}(r_i)} \sum_{\alpha}
A^{\text{pair}}_{r_i r_j \alpha} \cdot
\mathbf{W}_{\alpha}^{(k)} \mathbf{x}_{r_j}^{(k)}
\]

where:
- \( \mathbf{W}_{\alpha}^{(k)} \) are role-specific transformation matrices

---

### 8.2 Relation Update

The relation embedding update follows TRIX:

\[
\mathbf{x}_{r_i}^{(k+1)} =
\text{MLP}\left(
\mathbf{x}_{r_i}^{(k)} \;\Vert\; \mathbf{m}_{r_i}^{(k)}
\right)
\]

---

## 9. Optional: Hybrid TRIX + Pairwise Aggregation

For stability and ablation, the original TRIX adjacency can be combined:

\[
A^{\text{hybrid}} = A^{\text{TRIX}} + \lambda A^{\text{pair}}
\]

where \( \lambda \) controls the contribution of pairwise interactions.

---

## 10. Expressivity Discussion

- TRIX corresponds to a **1-WL-style refinement**, operating on individual entity contributions.
- Pairwise-TRIX introduces **pairwise entity interactions**, corresponding to a **localized 2-WL refinement**.
- The model can distinguish relational structures that are indistinguishable under independent aggregation.

---

## 11. Computational Considerations

- Pairwise construction is **local per relation pair**
- Worst-case complexity per relation pair: \( O(|\mathcal{V}_{ij}|^2) \)
- In practice, \( |\mathcal{V}_{ij}| \) is small due to relation sparsity
- No global entity-pair graph is constructed

---

## 12. Summary

Pairwise-TRIX enhances TRIX by introducing minimal higher-order structure:
- Captures entity co-occurrence patterns
- Preserves inductive generalization
- Improves expressivity beyond entity-wise aggregation
- Maintains interpretability and computational feasibility
