# Pithos Mathematical Foundations: Distance Preservation & Error Bounds

This document provides a formal mathematical analysis of why Pithos's projection pipeline preserves angular (cosine) similarity during binary quantization.

---

## 1. The Pithos Ingestion Pipeline

Given a high-dimensional vector $\mathbf{x} \in \mathbb{R}^d$, Pithos applies a three-step transformation prior to sign-quantization:
1. **LoRA Weight Modulation / SVD Projection** (if applicable) to align semantic dimensions.
2. **Rademacher Preconditioning ($D$):** Multiplication by a diagonal matrix $D \in \mathbb{R}^{d \times d}$ where diagonal entries are i.i.d. random signs:
   $$D_{ii} \sim \text{Uniform}(\{-1, +1\})$$
3. **Normalized Fast Walsh-Hadamard Transform ($H$):** Multiplication by the orthogonal Hadamard matrix $H = \frac{1}{\sqrt{d}} H_d$, defined recursively as:
   $$H_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad H_{2^k} = H_2 \otimes H_{2^{k-1}}$$

The projected vector is:
$$\mathbf{z} = H D \mathbf{x}$$

---

## 2. Energy Distribution & Flattening

Real-world embeddings (e.g., from DINOv3) are often highly sparse or contain extreme coordinate spikes (high peak-to-average power ratio). Sign binarization on raw vectors loses significant information if a few large components dominate.

### Theorem 1 (Energy Flattening / Sub-Gaussian Bounds)
For any vector $\mathbf{x} \in \mathbb{R}^d$ and random diagonal sign matrix $D$, the transformed vector $\mathbf{z} = H D \mathbf{x}$ has its maximum absolute coordinate bounded by:
$$P\left( \|\mathbf{z}\|_{\infty} \ge t \|\mathbf{x}\|_2 \right) \le 2 d \exp\left( -\frac{d \cdot t^2}{2} \right)$$

Setting $\delta = 2 d \exp(-d \cdot t^2 / 2)$ yields:
$$\|\mathbf{z}\|_{\infty} \le \sqrt{\frac{2 \ln(2d / \delta)}{d}} \|\mathbf{x}\|_2$$
with probability at least $1 - \delta$.

**Implication:** The Walsh-Hadamard transform spreads the energy uniformly across all $d$ coordinates. No single coordinate dominates, making the sign-quantization step robust and preventing catastrophic information loss.

---

## 3. Preservation of Cosine Similarity

After projection, Pithos applies the sign activation function:
$$A(\mathbf{x}) = \text{sign}(H D \mathbf{x}) \in \{-1, +1\}^d$$

Let $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ be two unit vectors ($\|\mathbf{x}\|_2 = \|\mathbf{y}\|_2 = 1$), and let $\theta = \arccos(\langle \mathbf{x}, \mathbf{y} \rangle)$ be the angle between them.

### Theorem 2 (Grothendieck / Charikar Sign-Quantization relation)
For a random rotation matrix $R \in \mathbb{R}^{d \times d}$ whose rows are independent spherically symmetric Gaussian vectors, the probability that the binarized signs of coordinate $i$ match is:
$$P(\text{sign}((R\mathbf{x})_i) = \text{sign}((R\mathbf{y})_i)) = 1 - \frac{\theta}{\pi}$$

In Pithos, the combination $H D$ acts as a **Fast Johnson-Lindenstrauss Transform (FJLT)** (Ailon & Chazelle, 2006). Because the Rademacher preconditioning $D$ randomizes the signs and $H$ mixes all coordinates, each coordinate of $H D \mathbf{x}$ behaves asymptotically as an independent random projection.

Therefore, the expected Hamming distance $d_H$ (the fraction of differing bits) over $d$ dimensions is:
$$\mathbb{E}\left[ \frac{d_H(A(\mathbf{x}), A(\mathbf{y}))}{d} \right] = \frac{\theta}{\pi} = \frac{\arccos(\langle \mathbf{x}, \mathbf{y} \rangle)}{\pi}$$

### Corollary (Monotonicity)
Since the mapping $\theta \mapsto \frac{\theta}{\pi}$ is strictly increasing for $\theta \in [0, \pi]$, sorting database items by their Hamming distance to a query vector is mathematically equivalent to sorting them by their original cosine similarity, up to a small approximation error bounded by the concentration of measure.

---

## 4. Concentration of Measure (Error Bounds)

By applying Hoeffding's inequality to the sum of independent-like coordinates:

$$P\left( \left| \frac{d_H(A(\mathbf{x}), A(\mathbf{y}))}{d} - \frac{\theta}{\pi} \right| \ge \epsilon \right) \le 2 \exp(-2 \epsilon^2 d)$$

For a database of size $N$ and a tolerance $\epsilon$, to ensure that all pairwise distance relations are preserved with high probability $1 - \gamma$, we require:
$$d \ge O\left( \epsilon^{-2} \log\left(\frac{N}{\gamma}\right) \right)$$

This is the classic **Johnson-Lindenstrauss bound**, proving that Pithos's binarized representation preserves neighbor search structure with logarithmic scaling in database size $N$.
