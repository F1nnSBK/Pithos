package org.pithos;

import java.lang.foreign.MemorySegment;
import java.util.List;

/// # Index
///
/// Core abstraction representing a multi-tier binary vector index in Pithos.
///
/// Supports:
/// - Exact and approximate Nearest Neighbor ($k$-NN) batch queries.
/// - Multi-family resonant voting for geospatial and planetary grid searches.
/// - Hardware-accelerated GPU offload fallbacks.
public interface Index extends AutoCloseable {

    /// Inserts a vector record into the index.
    ///
    /// @param record the vector record to insert
    /// @throws UnsupportedOperationException if the underlying index implementation is read-only
    void insert(VectorRecord record);

    /// Searches for the top $k$ nearest neighbors for a single raw continuous float query vector $\mathbf{q} \in \mathbb{R}^D$.
    ///
    /// @param query float query vector
    /// @param k number of closest neighbors to retrieve
    /// @return list of search results sorted ascending by distance
    List<SearchResult> search(float[] query, int k);

    /// Performs a batch $k$-NN search for multiple query vectors simultaneously across worker threads.
    ///
    /// @param queries 2D array of query vectors of shape $[\text{numQueries}][D]$
    /// @param k number of closest neighbors per query
    /// @return array of result lists, one list per query vector
    List<SearchResult>[] batchSearch(float[][] queries, int k);

    /// Performs a parallel scan over the entire index and evaluates multi-family resonant voting:
    ///
    /// For each record $i$, evaluates queries $q \in \{0, \dots, Q-1\}$ with threshold $\theta_q$:
    /// $$\text{mask}_i = \bigvee_{q : d_H(q, i) \le \theta_q} 2^{\text{family}(q)}$$
    ///
    /// @param queries query vectors of shape $[\text{numQueries}][D]$
    /// @param families semantic family index $[0, 7]$ for each query
    /// @param thresholds maximum Hamming distance cutoff for each query
    /// @param votingMask pre-allocated off-heap memory segment of size $N$ bytes to accumulate bitmasks
    /// @return count of highly resonant candidate records
    long queryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask);

    // =========================================================================
    // CUDA Acceleration Fallback Methods
    // =========================================================================

    /// Performs a CUDA-accelerated batch search. Defaults to CPU multi-threaded `batchSearch` if CUDA is unavailable.
    ///
    /// @param queries query vectors
    /// @param k nearest neighbor count
    /// @return search results
    default List<SearchResult>[] cudaBatchSearch(float[][] queries, int k) {
        return batchSearch(queries, k);
    }

    /// Performs a CUDA-accelerated planetary grid voting search. Defaults to CPU `queryPlanetaryGrid` if CUDA is unavailable.
    ///
    /// @param queries query vectors
    /// @param families semantic family array
    /// @param thresholds Hamming cutoff thresholds
    /// @param votingMask output off-heap mask segment
    /// @return count of resonant candidate records
    default long cudaQueryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        return queryPlanetaryGrid(queries, families, thresholds, votingMask);
    }

    /// Returns the vector dimensionality ($D$).
    int getDimension();

    /// Returns the total record count ($N$) in the index.
    long size();

    /// Returns the planetary target identifier code.
    byte getPlanetId();

    /// Returns the equatorial radius of the target planet in meters.
    long getPlanetRadius();

    /// Returns the number of cumulative Matryoshka tiers.
    int getTierCount();

    /// Represents a single search result match containing the resolved record ID and distance score.
    ///
    /// @param id the unique record identifier
    /// @param score metric distance (scaled by $1{,}000{,}000$ for float precision)
    record SearchResult(long id, int score) {}

    @Override
    default void close() throws Exception {}
}
