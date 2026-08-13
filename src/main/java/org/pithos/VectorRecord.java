package org.pithos;

import java.util.Objects;

/// # VectorRecord
///
/// Immutable representation of a single vector record in the Pithos database.
///
/// @param id unique 64-bit identifier for the record
/// @param vector raw continuous float vector representation $\mathbf{x} \in \mathbb{R}^D$
/// @param metadata 64-bit attribute flags and tombstone bitfield (bit 0 = tombstone, bit 1..63 = attribute mask)
public record VectorRecord(long id, float[] vector, long metadata) {

    public VectorRecord {
        Objects.requireNonNull(vector, "Vector cannot be null");
    }

    /// Convenience constructor initializing metadata to default active state (`0L`).
    ///
    /// @param id unique identifier
    /// @param vector raw float vector
    public VectorRecord(long id, float[] vector) {
        this(id, vector, 0L);
    }
}
