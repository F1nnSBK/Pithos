# ⚱ Pithos Examples & Developer SDK

This directory contains complete, up-to-date code examples demonstrating how to use Pithos across **Python**, **C/C++**, and **Java** for application development, real-time ingestion, and FPGA / DMA hardware co-design.

---

## Directory Structure

```text
examples/
├── python/
│   ├── quickstart.py          # Complete Python quickstart (compilation, LSM tree, merged search)
│   └── fpga_offload_demo.py   # Python FPGA/DMA hardware descriptors & zero-copy direct buffers
├── cpp/
│   ├── demo.c                 # C99 API demo (lifecycle, compilation, search)
│   └── fpga_dma_demo.c        # C/C++ FPGA DMA offloading & query binarization
└── java/
    ├── PithosApiDemo.java     # Pure Java database coordinator & multi-tier search demo
    └── ZeroCostDemo.java      # Java 25 Foreign Function & Memory (FFM) API off-heap demo
```

---

## 1. Python Examples

Install the official package from PyPI or link your local workspace:

```bash
pip install pithosdb numpy
```

Run the quickstart:
```bash
python examples/python/quickstart.py
```

Run the FPGA / Hardware Co-Design example:
```bash
python examples/python/fpga_offload_demo.py
```

---

## 2. C / C++ Examples

To compile and link the C examples against `libpithos`:

```bash
# Compile and run general C API demo
clang -Iinclude -Ltarget -lpithos -Wl,-rpath,@loader_path/target examples/cpp/demo.c -o /tmp/pithos_demo
/tmp/pithos_demo

# Compile and run FPGA & DMA Co-Design demo
clang -Iinclude -Ltarget -lpithos -Wl,-rpath,@loader_path/target examples/cpp/fpga_dma_demo.c -o /tmp/pithos_fpga_demo
/tmp/pithos_fpga_demo
```

---

## 3. Java Examples

Run directly using Maven:

```bash
mvn test-compile exec:java -Dexec.mainClass="examples.java.PithosApiDemo"
mvn test-compile exec:java -Dexec.mainClass="examples.java.ZeroCostDemo"
```
