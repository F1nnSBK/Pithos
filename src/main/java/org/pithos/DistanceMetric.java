package org.pithos;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// # DistanceMetric
///
/// Dimension-agnostic Hamming distance metric calculator for binary vectors.
///
/// ### Mathematical Definition:
/// The Hamming distance between two binary vectors a, b ∈ {0, 1}^D packed into 64-bit words is:
/// `d_H(a, b) = ∑_{w=0}^{W-1} popcount(a_w ⊕ b_w)`
/// where W = ⌈D / 64⌉, ⊕ is bitwise XOR, and popcount counts the number of set bits (1s).
public enum DistanceMetric {

    /// Hamming distance calculation over packed 64-bit integer words.
    HAMMING {
        @Override
        public int calculate(long[] a, long[] b) {
            int sum = 0;
            int len = Math.min(a.length, b.length);
            for (int i = 0; i < len; i++) {
                sum += Long.bitCount(a[i] ^ b[i]);
            }
            return sum;
        }
    };

    /// Computes the distance between two on-heap packed vector arrays.
    ///
    /// @param a first packed vector word array
    /// @param b second packed vector word array
    /// @return computed distance
    public abstract int calculate(long[] a, long[] b);

    /// Calculates the Hamming distance directly between an on-heap query vector and an off-heap `MemorySegment` byte offset.
    ///
    /// @param query on-heap query word array
    /// @param segment off-heap memory-mapped segment
    /// @param byteOffset absolute byte offset into `segment`
    /// @param numLongs number of 64-bit words (W) to scan
    /// @return total Hamming distance

    public static int calculateSegment(long[] query, MemorySegment segment, long byteOffset, int numLongs) {
        int sum = 0;
        for (int i = 0; i < numLongs; i++) {
            sum += Long.bitCount(query[i] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset + (i * 8L)));
        }
        return sum;
    }
}
