import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark import PithosMIDB, generate_hypersphere_vectors

def main():
    print("[Python Verify WAL] Starting WAL verification...")
    
    # 1. Initialize engine
    engine = PithosMIDB()
    
    dimension = 64
    tiers = np.array([32, 64], dtype=np.int32)
    db_path = "temp/db_wal_py"
    for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_fp16.bin", "_wal.bin"]:
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)
            
    # Compile base index
    vectors = generate_hypersphere_vectors(10, dimension)

    ids = np.arange(10, dtype=np.int64)
    status = engine.compile_index_file(db_path, 1, 1000, dimension, tiers, ids, vectors)
    if status != 0:
        raise RuntimeError(f"Base index compilation failed: {status}")
        
    # Load index
    status = engine.load_index("wal_test_idx", db_path)
    if status != 0:
        raise RuntimeError(f"Index load failed: {status}")
        
    # Create delta buffer
    status = engine.create_delta_buffer("wal_test_idx", 100)
    if status != 0:
        raise RuntimeError(f"Failed to create delta buffer: {status}")
        
    # Insert vectors to delta buffer
    ins0 = generate_hypersphere_vectors(1, dimension)[0]
    ins1 = generate_hypersphere_vectors(1, dimension)[0]
    ins2 = generate_hypersphere_vectors(1, dimension)[0]
    
    engine.insert("wal_test_idx", 1000, ins0)
    engine.insert("wal_test_idx", 2000, ins1)
    engine.insert("wal_test_idx", 3000, ins2)
    
    # Delete one
    engine.delete_from_delta("wal_test_idx", 2000)
    
    # Check sizes
    d_size = engine.delta_size("wal_test_idx")
    print(f"[Python Verify WAL] Initial delta size: {d_size}")
    assert d_size == 2, f"Expected delta size 2, got {d_size}"
    
    # Check WAL file exists
    wal_file = f"{db_path}_wal.bin"
    assert os.path.exists(wal_file), "WAL file should exist"
    assert os.path.getsize(wal_file) > 0, "WAL file should not be empty"
    
    # Close library (resets isolate thread)
    engine.close()
    
    # 2. Re-open engine and verify recovery
    print("[Python Verify WAL] Re-initializing engine context...")
    engine2 = PithosMIDB()
    
    status = engine2.load_index("wal_test_idx", db_path)
    if status != 0:
        raise RuntimeError(f"Index reload failed: {status}")
        
    # Re-create delta buffer (triggers WAL replay)
    status = engine2.create_delta_buffer("wal_test_idx", 100)
    if status != 0:
        raise RuntimeError(f"Failed to create delta buffer on recovery: {status}")
        
    # Check recovered size
    rec_size = engine2.delta_size("wal_test_idx")
    print(f"[Python Verify WAL] Recovered delta size: {rec_size}")
    assert rec_size == 2, f"Expected recovered size 2, got {rec_size}"
    
    # Verify search includes recovered inserts (ID 1000 or 3000 should be queryable)
    ids_out, dists_out = engine2.search_merged("wal_test_idx", ins0, 5)
    print(f"[Python Verify WAL] Merged Search Results (top-5): IDs={ids_out}, Dists={dists_out}")
    assert 1000 in ids_out, "ID 1000 should be in search results"
    assert 2000 not in ids_out, "Tombstoned ID 2000 should not be in search results"
    
    # Clean up files
    engine2.close()
    for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_fp16.bin", "_wal.bin"]:
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)
            
    print("[Python Verify WAL] SUCCESS!")

if __name__ == "__main__":
    main()
