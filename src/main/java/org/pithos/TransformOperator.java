package org.pithos;

import java.util.Arrays;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/// # TransformOperator
///
/// Handles isometric transformations and binarization for vector embeddings:
/// 1. **Rademacher Preconditioning (D_pre):** Deterministic random sign flipping (±1)
///    to eliminate spatial burstiness and flatten peak activations across embedding dimensions.
/// 2. **Block-Diagonal Walsh-Hadamard Transform (H_BD):** In-place fast orthogonal rotation
///    with O(D · log D) complexity, spreading embedding energy uniformly across coordinates.
/// 3. **Kronecker Fallback (H_u ⊗ Ω_v):** For non-power-of-two block widths, decomposes the block
///    into a power-of-two component u and a residual component v using a Discrete Cosine Transform (DCT-II) basis.
/// 4. **Adaptive Quantization:** 1-bit sign binarization and 2-bit ternary quantization with noise thresholding.
/// 5. **SVD Spectral Energy Profiling:** Computes the cumulative energy distribution Φ(k) from projection/LoRA weights
///    using the Jacobi eigenvalue algorithm, enabling dynamic Matryoshka tier budgeting.
///
/// All dense vector computations leverage the **Java Vector API (`jdk.incubator.vector`)** to compile
/// into native SIMD instructions (AVX2, AVX-512, or ARM NEON).
public final class TransformOperator {

    /// Preferred hardware SIMD vector shape for 32-bit floats.
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int dimension;
    private final int[] tiers;
    private final float[] signs;

    /// Constructs a `TransformOperator` using a deterministic Rademacher seed (`42`).
    ///
    /// @param dimension the total vector dimensionality (D)
    /// @param tiers cumulative Matryoshka tier step boundaries
    public TransformOperator(int dimension, int[] tiers) {
        this.dimension = dimension;
        this.tiers = Arrays.copyOf(tiers, tiers.length);
        this.signs = new float[dimension];

        // Generate Rademacher signs deterministically using seed 42 (matching Python client)
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            signs[i] = rand.nextBoolean() ? 1.0f : -1.0f;
        }
    }

    /// Constructs a `TransformOperator` with explicit custom preconditioning signs.
    ///
    /// @param dimension the total vector dimensionality (D)
    /// @param tiers cumulative Matryoshka tier step boundaries
    /// @param customSigns preconditioning sign array of length `dimension`
    public TransformOperator(int dimension, int[] tiers, float[] customSigns) {
        this.dimension = dimension;
        this.tiers = Arrays.copyOf(tiers, tiers.length);
        if (customSigns == null || customSigns.length != dimension) {
            throw new IllegalArgumentException("Signs length must match dimension: expected " + dimension);
        }
        this.signs = Arrays.copyOf(customSigns, customSigns.length);
    }

    /// Computes the cumulative spectral energy distribution Φ(k) from frozen projection/LoRA weights W ∈ ℝ^(D × D₀).
    ///
    /// ### Mathematical Foundation:
    /// 1. Form the symmetric Gram matrix:
    ///    `A = W · Wᵀ ∈ ℝ^(D × D)`
    /// 2. Compute eigenvalues λᵢ using the Jacobi eigenvalue algorithm with SIMD vectorization.
    /// 3. Singular values are σᵢ = √(max(0, λᵢ)).
    /// 4. Sort σᵢ descending and calculate the cumulative energy ratio:
    ///    `Φ(k) = (∑_{i=1}^k σᵢ²) / (∑_{i=1}^D σᵢ²)`
    ///
    /// @param flatW row-major flattened weight matrix of size D × D₀
    /// @param D output dimension (rows of W)
    /// @param D0 bottleneck/LoRA rank dimension (columns of W)
    /// @return cumulative energy array Φ of length D, where Φ[D-1] = 1.0

    public static float[] computeCumulativeEnergy(float[] flatW, int D, int D0) {
        // Construct symmetric covariance matrix A = W * W^T
        float[][] A = new float[D][D];
        for (int i = 0; i < D; i++) {
            int rowOffsetI = i * D0;
            for (int j = 0; j < D; j++) {
                int rowOffsetJ = j * D0;
                float sum = 0.0f;
                int k = 0;
                int kBound = SPECIES.loopBound(D0);
                FloatVector vSum = FloatVector.zero(SPECIES);
                for (; k < kBound; k += SPECIES.length()) {
                    FloatVector va = FloatVector.fromArray(SPECIES, flatW, rowOffsetI + k);
                    FloatVector vb = FloatVector.fromArray(SPECIES, flatW, rowOffsetJ + k);
                    vSum = va.fma(vb, vSum);
                }
                sum = vSum.reduceLanes(VectorOperators.ADD);
                for (; k < D0; k++) {
                    sum += flatW[rowOffsetI + k] * flatW[rowOffsetJ + k];
                }
                A[i][j] = sum;
            }
        }

        // Compute eigenvalues using SIMD Jacobi rotation
        float[] eigenvalues = jacobiEigenvalues(A, D);

        // Singular values are square roots of eigenvalues
        float[] sigmas = new float[D];
        float sumSigmasSq = 0.0f;
        for (int i = 0; i < D; i++) {
            sigmas[i] = (float) Math.sqrt(Math.max(0.0, eigenvalues[i]));
            sumSigmasSq += sigmas[i] * sigmas[i];
        }

        // Sort singular values in descending order
        Arrays.sort(sigmas);
        for (int i = 0; i < D / 2; i++) {
            float temp = sigmas[i];
            sigmas[i] = sigmas[D - 1 - i];
            sigmas[D - 1 - i] = temp;
        }

        // Compute cumulative spectral energy ratio Phi
        float[] phi = new float[D];
        float runningSum = 0.0f;
        for (int i = 0; i < D; i++) {
            runningSum += sigmas[i] * sigmas[i];
            phi[i] = sumSigmasSq > 0 ? (runningSum / sumSigmasSq) : 0.0f;
        }
        return phi;
    }

    /// Computes all eigenvalues of a real symmetric matrix A ∈ ℝ^(n × n) using Jacobi rotations.
    ///
    /// At each iteration, finds the maximal off-diagonal element A_pq and applies an orthogonal
    /// plane rotation R(p, q, θ) parameterized by:
    /// `tan(2θ) = 2 · A_pq / (A_qq - A_pp)`
    /// Row updates are vectorized via `FloatVector`.
    private static float[] jacobiEigenvalues(float[][] A, int n) {

        int maxIterations = 100;
        for (int iter = 0; iter < maxIterations; iter++) {
            int p = 0;
            int q = 1;
            float maxVal = Math.abs(A[0][1]);
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    float absVal = Math.abs(A[i][j]);
                    if (absVal > maxVal) {
                        maxVal = absVal;
                        p = i;
                        q = j;
                    }
                }
            }

            if (maxVal < 1e-6f) {
                break;
            }

            float apq = A[p][q];
            float app = A[p][p];
            float aqq = A[q][q];

            float theta = 0.5f * (aqq - app) / apq;
            float t = (float) (1.0 / (Math.abs(theta) + Math.sqrt(1.0 + theta * theta)));
            if (theta < 0) {
                t = -t;
            }

            float c = (float) (1.0 / Math.sqrt(1.0 + t * t));
            float s = t * c;

            // Perform Jacobi rotation using Java Vector API on rows A[p] and A[q]
            int i = 0;
            int upper = SPECIES.loopBound(n);
            for (; i < upper; i += SPECIES.length()) {
                FloatVector vp = FloatVector.fromArray(SPECIES, A[p], i);
                FloatVector vq = FloatVector.fromArray(SPECIES, A[q], i);

                FloatVector vpNew = vp.mul(c).sub(vq.mul(s));
                FloatVector vqNew = vp.mul(s).add(vq.mul(c));

                vpNew.intoArray(A[p], i);
                vqNew.intoArray(A[q], i);
            }
            for (; i < n; i++) {
                float ap = A[p][i];
                float aq = A[q][i];
                A[p][i] = c * ap - s * aq;
                A[q][i] = s * ap + c * aq;
            }

            // Update diagonal and annihilate intersection
            A[p][q] = 0.0f;
            A[q][p] = 0.0f;
            A[p][p] = app - t * apq;
            A[q][q] = aqq + t * apq;

            // Maintain symmetry: synchronize columns
            for (int j = 0; j < n; j++) {
                if (j != p && j != q) {
                    A[j][p] = A[p][j];
                    A[j][q] = A[q][j];
                }
            }
        }

        float[] eigenvalues = new float[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = A[i][i];
        }
        return eigenvalues;
    }

    /// Preconditions an input vector with Rademacher signs, rotates it via block-diagonal Hadamard,
    /// and binarizes it into packed 64-bit words (1 bit per dimension).
    ///
    /// @param x raw input float vector of length `dimension`
    /// @return packed 64-bit `long[]` array of length ⌈D / 64⌉
    public long[] transformAndQuantize(float[] x) {
        if (x.length != dimension) {
            throw new IllegalArgumentException("Input vector size " + x.length + " must match dimension " + dimension);
        }

        // 1. Rademacher Preconditioning (Sign-flip)
        float[] z = new float[dimension];
        int i = 0;
        int upperBound = SPECIES.loopBound(dimension);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, x, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, signs, i);
            va.mul(vb).intoArray(z, i);
        }
        for (; i < dimension; i++) {
            z[i] = x[i] * signs[i];
        }

        // 2. Block-Diagonal Hadamard Rotation
        int start = 0;
        for (int tier : tiers) {
            int width = tier - start;
            rotateBlock(z, start, width);
            start = tier;
        }

        // 3. 1-Bit Binarization & Packing into longs (64 bits per long)
        int numLongs = (dimension + 63) / 64;
        long[] packed = new long[numLongs];
        for (int j = 0; j < dimension; j++) {
            if (z[j] >= 0.0f) {
                int longIdx = j / 64;
                int bitIdx = j % 64;
                packed[longIdx] |= (1L << bitIdx);
            }
        }
        return packed;
    }

    /// Preconditions an input vector with Rademacher signs and rotates it via block-diagonal Hadamard.
    ///
    /// @param x raw input float vector of length `dimension`
    /// @return rotated float vector z = H_BD · D_pre · x
    public float[] preconditionAndRotate(float[] x) {
        if (x.length != dimension) {
            throw new IllegalArgumentException("Input vector size " + x.length + " must match dimension " + dimension);
        }

        // 1. Rademacher Preconditioning (Sign-flip)
        float[] z = new float[dimension];
        int i = 0;
        int upperBound = SPECIES.loopBound(dimension);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, x, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, signs, i);
            va.mul(vb).intoArray(z, i);
        }
        for (; i < dimension; i++) {
            z[i] = x[i] * signs[i];
        }

        // 2. Block-Diagonal Hadamard Rotation
        int start = 0;
        for (int tier : tiers) {
            int width = tier - start;
            rotateBlock(z, start, width);
            start = tier;
        }
        return z;
    }

    /// Binarizes a rotated vector z using 2-bit (ternary) quantization:
    /// - Sign bit: 1 if z_j ≥ 0, 0 if z_j < 0
    /// - Mask bit: 1 if |z_j| ≥ threshold (active coordinate), 0 if |z_j| < threshold (noise)
    ///
    /// @param z rotated float vector
    /// @param threshold noise suppression threshold
    /// @return `long[2][]` where index 0 is `signPacked` and index 1 is `maskPacked`
    public long[][] quantize2Bit(float[] z, float threshold) {
        int numLongs = (dimension + 63) / 64;
        long[] signPacked = new long[numLongs];
        long[] maskPacked = new long[numLongs];

        for (int j = 0; j < dimension; j++) {
            float val = z[j];
            float absVal = Math.abs(val);
            if (absVal >= threshold) {
                int longIdx = j / 64;
                int bitIdx = j % 64;
                maskPacked[longIdx] |= (1L << bitIdx);
                if (val >= 0.0f) {
                    signPacked[longIdx] |= (1L << bitIdx);
                }
            }
        }
        return new long[][]{signPacked, maskPacked};
    }

    /// Calculates the percentile threshold for absolute values of z, used for 2-bit noise cutoff.
    ///
    /// @param z input vector
    /// @param percentile cutoff percentile in [0, 1] (e.g. 0.20 for bottom 20% noise)
    /// @return threshold value
    public static float calculatePercentileThreshold(float[] z, float percentile) {
        float[] absValues = new float[z.length];
        int i = 0;
        int upper = SPECIES.loopBound(z.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector vz = FloatVector.fromArray(SPECIES, z, i);
            vz.abs().intoArray(absValues, i);
        }
        for (; i < z.length; i++) {
            absValues[i] = Math.abs(z[i]);
        }
        Arrays.sort(absValues);
        int index = Math.min(absValues.length - 1, (int) (z.length * percentile));
        return absValues[index];
    }

    /// Computes the exact Euclidean L2 squared distance ||q - d||² between two float vectors
    /// using hardware SIMD lanes via `FloatVector`.
    ///
    /// @param query query float vector
    /// @param db database float vector
    /// @return squared L2 distance
    public float computeL2Float(float[] query, float[] db) {
        int n = Math.min(query.length, db.length);
        float sum = 0.0f;
        int i = 0;
        int upper = SPECIES.loopBound(n);
        FloatVector vSum = FloatVector.zero(SPECIES);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector vq = FloatVector.fromArray(SPECIES, query, i);
            FloatVector vd = FloatVector.fromArray(SPECIES, db, i);
            FloatVector diff = vq.sub(vd);
            vSum = diff.fma(diff, vSum);
        }
        sum = vSum.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float diff = query[i] - db[i];
            sum += diff * diff;
        }
        return sum;
    }

    /// Back-projects a transformed vector z to raw input space x.
    /// Since H_BD and D_pre are orthogonal, symmetric, and self-inverse:
    /// `x = D_pre · H_BD · z`
    ///
    /// @param z transformed vector
    /// @return reconstructed raw vector x
    public float[] backProject(float[] z) {
        if (z.length != dimension) {
            throw new IllegalArgumentException("Target vector size must match dimension: expected " + dimension);
        }
        float[] x = new float[dimension];
        System.arraycopy(z, 0, x, 0, dimension);

        // 1. Rotate by block Hadamard
        int start = 0;
        for (int tier : tiers) {
            int width = tier - start;
            rotateBlock(x, start, width);
            start = tier;
        }

        // 2. Precondition (sign-flip)
        int idx = 0;
        int upper = SPECIES.loopBound(dimension);
        for (; idx < upper; idx += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, x, idx);
            FloatVector vb = FloatVector.fromArray(SPECIES, signs, idx);
            va.mul(vb).intoArray(x, idx);
        }
        for (; idx < dimension; idx++) {
            x[idx] = x[idx] * signs[idx];
        }
        return x;
    }

    /// Dispatches block rotation: uses O(D · log D) Fast Walsh-Hadamard Transform (FWHT) if
    /// block width is a power of two, otherwise applies Kronecker factorized rotation.
    private void rotateBlock(float[] z, int start, int width) {
        if ((width & (width - 1)) == 0) {
            fwht(z, start, width);
        } else {
            // Factorize width = u * v where u is the largest power of two
            int u = 1;
            while (width % (u * 2) == 0) {
                u *= 2;
            }
            int v = width / u;
            kroneckerRotate(z, start, u, v);
        }
    }

    /// In-place SIMD Fast Walsh-Hadamard Transform (FWHT) normalized by 1 / √N.
    ///
    /// Evaluates the Hadamard recursive butterfly network:
    /// `H_{2k} = (1 / √2) · [ H_k   H_k ;  H_k  -H_k ]`
    private void fwht(float[] a, int start, int length) {

        for (int len = 1; len < length; len <<= 1) {
            for (int i = 0; i < length; i += (len << 1)) {
                if (len >= SPECIES.length()) {
                    int j = 0;
                    int lenBound = SPECIES.loopBound(len);
                    for (; j < lenBound; j += SPECIES.length()) {
                        FloatVector vu = FloatVector.fromArray(SPECIES, a, start + i + j);
                        FloatVector vv = FloatVector.fromArray(SPECIES, a, start + i + len + j);
                        FloatVector vAdd = vu.add(vv);
                        FloatVector vSub = vu.sub(vv);
                        vAdd.intoArray(a, start + i + j);
                        vSub.intoArray(a, start + i + len + j);
                    }
                    for (; j < len; j++) {
                        float u = a[start + i + j];
                        float v = a[start + i + len + j];
                        a[start + i + j] = u + v;
                        a[start + i + len + j] = u - v;
                    }
                } else {
                    for (int j = 0; j < len; j++) {
                        float u = a[start + i + j];
                        float v = a[start + i + len + j];
                        a[start + i + j] = u + v;
                        a[start + i + len + j] = u - v;
                    }
                }
            }
        }
        // Orthogonal normalization: scale by 1 / sqrt(length)
        float scale = (float) (1.0 / Math.sqrt(length));
        int idx = 0;
        int upper = SPECIES.loopBound(length);
        for (; idx < upper; idx += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, start + idx);
            va.mul(scale).intoArray(a, start + idx);
        }
        for (; idx < length; idx++) {
            a[start + idx] *= scale;
        }
    }

    /// Kronecker-factorized orthogonal rotation for non-power-of-two block widths:
    /// `R = H_u ⊗ Ω_v`
    /// where Ω_v ∈ ℝ^(v × v) is an orthogonal DCT-II basis matrix and H_u is a Walsh-Hadamard matrix of size u.
    private void kroneckerRotate(float[] a, int start, int u, int v) {

        // Construct deterministic orthogonal matrix Omega_v using DCT basis
        float[][] omega = new float[v][v];
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                if (i == 0) {
                    omega[i][j] = (float) (1.0 / Math.sqrt(v));
                } else {
                    omega[i][j] = (float) (Math.sqrt(2.0 / v) * Math.cos(Math.PI * i * (2 * j + 1) / (2.0 * v)));
                }
            }
        }

        // Apply Omega_v to each block of size v
        float[] temp = new float[u * v];
        for (int block = 0; block < u; block++) {
            int blockStart = start + block * v;
            for (int i = 0; i < v; i++) {
                float sum = 0.0f;
                for (int j = 0; j < v; j++) {
                    sum += omega[i][j] * a[blockStart + j];
                }
                temp[block * v + i] = sum;
            }
        }

        // Apply FWHT of size u across block coordinates
        float[] column = new float[u];
        for (int coord = 0; coord < v; coord++) {
            for (int block = 0; block < u; block++) {
                column[block] = temp[block * v + coord];
            }

            fwht(column, 0, u);

            for (int block = 0; block < u; block++) {
                a[start + block * v + coord] = column[block];
            }
        }
    }
}
