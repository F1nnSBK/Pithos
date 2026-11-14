# Pithos – Roadmap & Architectural Extensions

*Architectural Roadmap & Next Steps for the Model-Isomorphic Vector Database*

---

## 1. Completed Core Milestones (v2.0 – v2.2.6)

| Feature Area | Status | Key Innovations |
| :--- | :--- | :--- |
| **Universal Container Format** | **Released (v2.0)** | Schema-agnostic `.pithos` single-file container with embedded Apache Arrow IPC partition directory. |
| **4x8-Bit MIH Prefix Routing** | **Released (v2.1)** | Direct-mapped CSR inverted table (`4 x 256` buckets) for fast Gate 0 candidate pruning. |
| **Precision Sidecar Reranking** | **Released (v2.0)** | Asymmetric Query LUTs decoding FP8 (E4M3), NVFP4 (E2M1), and FP16 with Monotonic Early Distance Cutoff. |
| **LSM DeltaBuffer & WAL** | **Released (v2.0)** | Lock-free real-time insertions, persistent base/delta tombstones, write-ahead log, and zero-cost compactions. |
| **Zero-Copy NumPy FFI** | **Released (v2.1)** | Direct memory view returns (`search_numpy()`, `batch_search_numpy()`) bypassing Python wrapper conversions. |
| **CUDA GPU Acceleration** | **Released (v2.0)** | Fused batch Hamming kernels, Fast Walsh-Hadamard transform, FP8 reranking, and multi-family resonant voting. |
| **Matryoshka Recall Verification**| **Released (v2.2)** | Verified `92.20%` Recall@10 on `D=384` Matryoshka embeddings (`FP8 E4M3` + MIH, `1.94 ms` mean CPU latency). |
| **Disruptor 4.0.0 & VarHandles**| **Released (v2.2)** | Replaced `sun.misc.Unsafe` with `java.lang.invoke.VarHandle` and FFM `MemorySegment`, zero warnings on Java 25+. |
| **Zero-Allocation BitSets** | **Released (v2.2)** | `ThreadLocal<long[]>` scratch buffers minimizing heap allocations during search queries. |
| **Zero-Overhead Security** | **Released (v2.2)** | Defensive C-API pointer guards, Superblock/TOC bounds verification, and path traversal sanitization. |
| **Java 25, Vector API & WHT Parity**| **Released (v2.2.6)** | Finalized Java 25 syntax with GraalVM `-H:+VectorAPISupport`, row-norm energy pruning replacing Jacobi SVD, unified WHT-space FP8/NVFP4 alignment across all layouts, and full native macOS ARM64 execution. |

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

