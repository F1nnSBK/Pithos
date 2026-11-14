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
/// 5. **Spectral Energy Profiling:** Computes the cumulative projection weight energy distribution Φ(k) across output dimensions
///    from row L2 norms of projection/LoRA weights, enabling dynamic Matryoshka tier budgeting.
///
/// All dense vector computations leverage the **Java Vector API (`jdk.incubator.vector`)** to compile
/// into native SIMD instructions (AVX2, AVX-512, or ARM NEON).
public final class TransformOperator {

    /// Preferred hardware SIMD vector shape for 32-bit floats.
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ThreadLocal<float[]> KRONECKER_SCRATCH = new ThreadLocal<>();

    private final int dimension;
    private final int[] tiers;
    private final float[] signs;
    private final float[][][] tierDctMatrices;

    /// Constructs a `TransformOperator` using a deterministic Rademacher seed (`42`).
    ///
    /// @param dimension the total vector dimensionality (D)
    /// @param tiers cumulative Matryoshka tier step boundaries
    public TransformOperator(int dimension, int[] tiers) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive: " + dimension);
        }
        int[] validatedTiers = validateTiers(dimension, tiers);
        this.dimension = dimension;
        this.tiers = validatedTiers;
        this.signs = createDefaultSigns(dimension);
        this.tierDctMatrices = buildTierDctMatrices(validatedTiers);
        super();
    }

    /// Constructs a `TransformOperator` with explicit custom preconditioning signs.
    ///
    /// @param dimension the total vector dimensionality (D)
    /// @param tiers cumulative Matryoshka tier step boundaries
    /// @param customSigns preconditioning sign array of length `dimension`
    public TransformOperator(int dimension, int[] tiers, float[] customSigns) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive: " + dimension);
        }
        if (customSigns == null || customSigns.length != dimension) {
            throw new IllegalArgumentException("Signs length must match dimension: expected " + dimension);
        }
        int[] validatedTiers = validateTiers(dimension, tiers);
        this.dimension = dimension;
        this.tiers = validatedTiers;
        this.signs = Arrays.copyOf(customSigns, customSigns.length);
        this.tierDctMatrices = buildTierDctMatrices(validatedTiers);
        super();
    }

    private static int[] validateTiers(int dimension, int[] tiers) {
        if (tiers == null || tiers.length == 0) {
            throw new IllegalArgumentException("Tiers must not be null or empty");
        }
        int prev = 0;
        for (int tier : tiers) {
            if (tier <= prev || tier > dimension) {
                throw new IllegalArgumentException("Invalid cumulative tier boundary " + tier + " for dimension " + dimension);
            }
            prev = tier;
        }
        return Arrays.copyOf(tiers, tiers.length);
    }

    private static float[] createDefaultSigns(int dimension) {
        float[] defaultSigns = new float[dimension];
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            defaultSigns[i] = rand.nextBoolean() ? 1.0f : -1.0f;
        }
        return defaultSigns;
    }

    private static float[][][] buildTierDctMatrices(int[] tiers) {
        float[][][] matrices = new float[tiers.length][][];
        int start = 0;
        for (int t = 0; t < tiers.length; t++) {
            int width = tiers[t] - start;
            if ((width & (width - 1)) != 0) {
                int u = 1;
                while ((width % (u << 1)) == 0) {
                    u <<= 1;
                }
                matrices[t] = buildDctMatrix(width / u);
            }
            start = tiers[t];
        }
        return matrices;
    }

    private static float[][] buildDctMatrix(int v) {
        float[][] omega = new float[v][v];
        float c0 = (float) (1.0 / Math.sqrt(v));
        double scale = Math.sqrt(2.0 / v);
        Arrays.fill(omega[0], c0);
        for (int i = 1; i < v; i++) {
            for (int j = 0; j < v; j++) {
                omega[i][j] = (float) (scale * Math.cos(Math.PI * i * (2 * j + 1) / (2.0 * v)));
            }
        }
        return omega;
    }

    /// Computes the cumulative prefix energy distribution Φ(k) from projection/LoRA weights W ∈ ℝ^(D × D₀)
    /// without destructive global eigenvalue sorting.
    ///
    /// ### Mathematical Foundation:
    /// Because Pithos applies a block-diagonal Walsh-Hadamard Transform (H_BD) that rotates each
    /// Matryoshka tier block orthogonally in-place without permuting coordinates across tiers,
    /// Parseval's theorem guarantees that the spectral energy within prefix [0..k] equals the
    /// unsorted cumulative squared Euclidean row norms of W:
    /// 1. Compute coordinate row energy in O(D · D₀) using SIMD FMA lanes:
    ///    `eᵢ = ||W_{i,:}||₂² = ∑_{j=0}^{D₀-1} W_{i,j}²`
    /// 2. Accumulate prefix energy ratio in native coordinate order (preserving spatial geometry):
    ///    `Φ(k) = (∑_{i=0}^k eᵢ) / (∑_{i=0}^{D-1} eᵢ)`
    ///
    /// @param flatW row-major flattened weight matrix of size D × D₀
    /// @param D output dimension (rows of W)
    /// @param D0 bottleneck/LoRA rank dimension (columns of W)
    /// @return cumulative prefix energy array Φ of length D, where Φ[D-1] = 1.0
    public static float[] computeCumulativeEnergy(float[] flatW, int D, int D0) {
        float[] rowEnergies = new float[D];
        float totalEnergy = 0.0f;
        int kBound = SPECIES.loopBound(D0);

        for (int i = 0; i < D; i++) {
            int rowOffsetI = i * D0;
            int k = 0;
            FloatVector vSum = FloatVector.zero(SPECIES);
            for (; k < kBound; k += SPECIES.length()) {
                FloatVector va = FloatVector.fromArray(SPECIES, flatW, rowOffsetI + k);
                vSum = va.fma(va, vSum);
            }
            float sum = vSum.reduceLanes(VectorOperators.ADD);
            for (; k < D0; k++) {
                float val = flatW[rowOffsetI + k];
                sum += val * val;
            }
            rowEnergies[i] = sum;
            totalEnergy += sum;
        }

        float[] phi = new float[D];
        float runningSum = 0.0f;
        for (int i = 0; i < D; i++) {
            runningSum += rowEnergies[i];
            phi[i] = totalEnergy > 0.0f ? (runningSum / totalEnergy) : ((float) (i + 1) / D);
        }
        return phi;
    }

    /// Returns a defensive copy of the tier step boundaries.
    public int[] getTiers() {
        return Arrays.copyOf(tiers, tiers.length);
    }

    /// Preconditions an input vector with Rademacher signs, rotates it via block-diagonal Hadamard,
    /// and binarizes it into packed 64-bit words (1 bit per dimension).
    ///
    /// @param x raw input float vector of length `dimension`
    /// @return packed 64-bit `long[]` array of length ⌈D / 64⌉
    public long[] transformAndQuantize(float[] x) {
        return quantize1Bit(preconditionAndRotate(x));
    }

    /// Binarizes an already preconditioned and rotated vector z into packed 64-bit words (1 bit per dimension),
    /// avoiding redundant Rademacher and Walsh-Hadamard transforms.
    ///
    /// @param z rotated float vector of length `dimension`
    /// @return packed 64-bit `long[]` array of length ⌈D / 64⌉
    public long[] quantize1Bit(float[] z) {
        if (z.length != dimension) {
            throw new IllegalArgumentException("Rotated vector size " + z.length + " must match dimension " + dimension);
        }
        int numLongs = (dimension + 63) / 64;
        long[] packed = new long[numLongs];
        for (int j = 0; j < dimension; j++) {
            if (z[j] >= 0.0f) {
                packed[j >>> 6] |= (1L << (j & 63));
            }
        }
        return packed;
    }

    /// Extracts a 64-bit word for a specific tier `[startDim, startDim + width)` from a contiguous bit-packed array.
    /// Supports arbitrary tier boundaries and widths (including sub-64-bit and non-64-aligned tiers).
    public static long extractTierWord(long[] packed, int startDim, int width, int wordIdx) {
        int bitPos = startDim + (wordIdx << 6);
        int remBits = width - (wordIdx << 6);
        if (remBits <= 0) {
            return 0L;
        }
        int srcIdx = bitPos >>> 6;
        int shift = bitPos & 63;
        long word = (srcIdx < packed.length) ? (packed[srcIdx] >>> shift) : 0L;
        if (shift != 0 && (srcIdx + 1) < packed.length) {
            word |= (packed[srcIdx + 1] << (64 - shift));
        }
        if (remBits < 64) {
            word &= (1L << remBits) - 1L;
        }
        return word;
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
        for (int t = 0; t < tiers.length; t++) {
            int tier = tiers[t];
            rotateBlock(z, start, tier - start, t, false);
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

        // 1. Inverse rotate by block Hadamard / Kronecker transpose
        int start = 0;
        for (int t = 0; t < tiers.length; t++) {
            int tier = tiers[t];
            rotateBlock(x, start, tier - start, t, true);
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

    private void rotateBlock(float[] z, int start, int width, int tierIdx, boolean inverse) {
        float[][] omega = tierDctMatrices[tierIdx];
        if (omega == null) {
            fwht(z, start, width);
        } else {
            int v = omega.length;
            kroneckerRotate(z, start, width / v, v, omega, inverse);
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
    /// `R = H_u ⊗ Ω_v` (and inverse `R^T = H_u ⊗ Ω_v^T`)
    /// where Ω_v ∈ ℝ^(v × v) is an orthogonal DCT-II basis matrix and H_u is a Walsh-Hadamard matrix of size u.
    private void kroneckerRotate(float[] a, int start, int u, int v, float[][] omega, boolean inverse) {
        int needed = v + u;
        float[] scratch = KRONECKER_SCRATCH.get();
        if (scratch == null || scratch.length < needed) {
            scratch = new float[needed];
            KRONECKER_SCRATCH.set(scratch);
        }

        // Apply Omega_v (or Omega_v^T when inverse == true) to each block of size v in-place via scratch[0..v)
        for (int block = 0; block < u; block++) {
            int blockStart = start + block * v;
            for (int i = 0; i < v; i++) {
                float sum = 0.0f;
                if (inverse) {
                    for (int j = 0; j < v; j++) {
                        sum += omega[j][i] * a[blockStart + j];
                    }
                } else {
                    float[] omegaRow = omega[i];
                    for (int j = 0; j < v; j++) {
                        sum += omegaRow[j] * a[blockStart + j];
                    }
                }
                scratch[i] = sum;
            }
            System.arraycopy(scratch, 0, a, blockStart, v);
        }

        // Apply FWHT of size u across block coordinates using scratch[v..v+u)
        for (int coord = 0; coord < v; coord++) {
            for (int block = 0; block < u; block++) {
                scratch[v + block] = a[start + block * v + coord];
            }

            fwht(scratch, v, u);

            for (int block = 0; block < u; block++) {
                a[start + block * v + coord] = scratch[v + block];
            }
        }
    }
}
