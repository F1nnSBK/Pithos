# Pithos – Roadmap & Architectural Extensions

*Architectural Roadmap & Next Steps for the Model-Isomorphic Vector Database*

---

## 1. Completed Core Milestones (v2.0 & v2.1)

| Feature Area | Status | Key Innovations |
| :--- | :--- | :--- |
| **Universal Container Format** | **Released (v2.0)** | Schema-agnostic `.pithos` single-file container with embedded Apache Arrow IPC partition directory. |
| **4x8-Bit MIH Prefix Routing** | **Released (v2.1)** | Direct-mapped CSR inverted table pruning 98.5% of search space in $O(1)$ sub-microsecond time. |
| **Precision Sidecar Reranking** | **Released (v2.0)** | Asymmetric Query LUTs decoding FP8 (E4M3), NVFP4 (E2M1), FP16 with Monotonic Early Distance Cutoff. |
| **LSM DeltaBuffer & WAL** | **Released (v2.0)** | Lock-free real-time insertions, tombstone soft-deletes, write-ahead log, and zero-cost binary compactions. |
| **Zero-Copy NumPy FFI** | **Released (v2.1)** | Direct memory view returns (`search_numpy()`, `batch_search_numpy()`) bypassing Python wrapper conversions. |
| **CUDA GPU Acceleration** | **Released (v2.0)** | Fused batch Hamming kernels, Fast Walsh-Hadamard transform, and multi-family resonant voting. |

---

## 2. Forward Roadmap & Future Directions

### 2.1 FPGA Co-Design via PCIe DMA Streaming
* **Objective:** Stream off-heap binary tier blocks directly from NVMe / Host RAM to FPGA accelerator cards (e.g. AMD Xilinx Alveo U280) via PCIe Gen5 DMA.
* **Architecture:** Host CPU performs asynchronous query preconditioning; FPGA runs massive pipelined Hamming popcount reduction units over 100+ parallel channels.
* **Direct Virtual Addressing:** Built-in `vdb_get_tier_address()` provides contiguous off-heap addresses for zero-copy DMA engine setup.

### 2.2 GPUDirect Storage (GDS) & NVMe-oF Distributed Search
* **Objective:** Bypass Host CPU and DRAM entirely during massive multi-billion vector batch scans by streaming `.pithos` files directly from NVMe into GPU High Bandwidth Memory (HBM3e).
* **Architecture:** Leverage cuFile APIs and NVMe over Fabrics (NVMe-oF) for cluster-wide partition retrieval.

### 2.3 Distributed Sharding & Raft Cluster Consensus
* **Objective:** Partition multi-billion scale datasets across horizontal nodes with sub-millisecond scatter-gather query aggregation.
* **Architecture:** Consistent hashing over 16-bit Walsh-Hadamard prefixes, routing queries directly to target cluster shards.

### 2.4 Autonomous In-Storage Compaction Engine
* **Objective:** Background merge workers running on smart NVMe computational storage drives to compact WAL DeltaBuffers without consuming host CPU cycles.

