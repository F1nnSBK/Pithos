# Copyright (c) 2026 Pithos Authors and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
================================================================================
Pithos Native C-FFI Bindings & GraalVM Native Image Isolate Management
================================================================================

This module encapsulates the low-level Foreign Function Interface (FFI) bindings
to the native GraalVM shared library (`libpithos.so` / `libpithos.dylib`).
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Optional
from .loader import find_or_fetch_native_library

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

class PithosNativeError(RuntimeError):
    """Raised when a Pithos C-API call returns a negative error code."""
    ERROR_MESSAGES = {
        -1: "Pithos database coordinator not initialized (vdb_init not called).",
        -2: "Index not found or not mapped in database coordinator.",
        -3: "Invalid operation or unsupported parameter for this index layout.",
        -4: "Internal Java/GraalVM runtime exception occurred.",
        -5: "File I/O error reading or writing index files on disk.",
        -6: "Unsupported index structure or layout mismatch.",
    }
    
    def __init__(self, code: int, message: Optional[str] = None):
        self.code = code
        detail = message or self.ERROR_MESSAGES.get(code, f"Unknown native error code: {code}")
        super().__init__(f"[Pithos Native Error {code}] {detail}")

def _preload_cuda_runtime():
    """Attempts to locate and pre-load libcudart.so into the process with RTLD_GLOBAL."""
    search_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/local/lib/ollama/cuda_v11",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
    ]
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            search_dirs.insert(0, torch_lib)
    except Exception:
        pass

    try:
        import nvidia.cuda_runtime.lib as nvcuda
        nvcuda_dir = os.path.dirname(nvcuda.__file__)
        if os.path.exists(nvcuda_dir):
            search_dirs.insert(0, nvcuda_dir)
    except Exception:
        pass

    for s_dir in search_dirs:
        if os.path.isdir(s_dir):
            for candidate in ["libcudart.so.12", "libcudart.so.11", "libcudart.so"]:
                cand_path = os.path.join(s_dir, candidate)
                if os.path.exists(cand_path):
                    try:
                        ctypes.CDLL(cand_path, mode=ctypes.RTLD_GLOBAL)
                        return
                    except Exception:
                        pass

class NativeBindings:
    """Thread-safe singleton managing the GraalVM Native Image isolate lifecycle and C-API bindings."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, lib_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                instance = super(NativeBindings, cls).__new__(cls)
                _preload_cuda_runtime()
                resolved_path = find_or_fetch_native_library(lib_path)
                instance._init_library(resolved_path)
                cls._instance = instance
        return cls._instance

    def _init_library(self, lib_path: str):
        self.lib_path = lib_path
        self.lib = ctypes.CDLL(lib_path)
        self.isolate = ctypes.POINTER(GraalIsolate)()
        self.thread = ctypes.POINTER(GraalIsolateThread)()
        
        # GraalVM Isolate Management
        self.lib.graal_create_isolate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(GraalIsolate)),
            ctypes.POINTER(ctypes.POINTER(GraalIsolateThread))
        ]
        self.lib.graal_create_isolate.restype = ctypes.c_int
        
        self.lib.graal_tear_down_isolate.argtypes = [ctypes.c_void_p]
        self.lib.graal_tear_down_isolate.restype = ctypes.c_int
        
        # Database Coordinator Lifecycle
        self.lib.vdb_init.argtypes = [ctypes.c_void_p]
        self.lib.vdb_init.restype = ctypes.c_int
        
        self.lib.vdb_close.argtypes = [ctypes.c_void_p]
        self.lib.vdb_close.restype = ctypes.c_int

        # Index Management
        self.lib.vdb_load_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.vdb_load_index.restype = ctypes.c_int

        self.lib.vdb_load_index_with_weights.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int
        ]
        self.lib.vdb_load_index_with_weights.restype = ctypes.c_int

        self.lib.vdb_drop_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_drop_index.restype = ctypes.c_int

        self.lib.vdb_get_info.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_get_info.restype = ctypes.c_int

        self.lib.vdb_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_size.restype = ctypes.c_longlong

        self.lib.vdb_set_chunk_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong]
        self.lib.vdb_set_chunk_size.restype = ctypes.c_int

        self.lib.vdb_set_energy_budget.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
        self.lib.vdb_set_energy_budget.restype = ctypes.c_int

        if hasattr(self.lib, "vdb_get_sidecar_mode"):
            self.lib.vdb_get_sidecar_mode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.lib.vdb_get_sidecar_mode.restype = ctypes.c_int

        # Search & Resonant Voting
        self.lib.vdb_batch_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_batch_search.restype = ctypes.c_int

        self.lib.vdb_query_planetary_grid.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p
        ]
        self.lib.vdb_query_planetary_grid.restype = ctypes.c_longlong

        # Index Compilation & Compaction
        self.lib.vdb_compile_index_file.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_byte, ctypes.c_longlong,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        self.lib.vdb_compile_index_file.restype = ctypes.c_int

        self.lib.vdb_compile_index_file_ext.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_byte, ctypes.c_longlong,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.vdb_compile_index_file_ext.restype = ctypes.c_int

        if hasattr(self.lib, "vdb_compile_container"):
            self.lib.vdb_compile_container.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p
            ]
            self.lib.vdb_compile_container.restype = ctypes.c_int

        if hasattr(self.lib, "vdb_get_user_metadata"):
            self.lib.vdb_get_user_metadata.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
            ]
            self.lib.vdb_get_user_metadata.restype = ctypes.c_int

        self.lib.vdb_compact_indexes.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.vdb_compact_indexes.restype = ctypes.c_int

        # Direct DMA / Memory Address Access
        self.lib.vdb_get_tier_address.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_get_tier_address.restype = ctypes.c_int

        self.lib.vdb_get_metadata_address.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_get_metadata_address.restype = ctypes.c_int

        self.lib.vdb_get_ids_address.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_get_ids_address.restype = ctypes.c_int

        self.lib.vdb_transform_and_quantize.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_transform_and_quantize.restype = ctypes.c_int

        # LSM Delta Buffer Operations
        self.lib.vdb_create_delta_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.vdb_create_delta_buffer.restype = ctypes.c_int

        self.lib.vdb_insert.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong, ctypes.c_void_p]
        self.lib.vdb_insert.restype = ctypes.c_int

        self.lib.vdb_delete_from_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong]
        self.lib.vdb_delete_from_delta.restype = ctypes.c_int

        self.lib.vdb_delta_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_delta_size.restype = ctypes.c_longlong

        self.lib.vdb_needs_flush.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_needs_flush.restype = ctypes.c_int

        self.lib.vdb_search_merged.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_search_merged.restype = ctypes.c_int

        self.lib.vdb_backup_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.vdb_backup_delta.restype = ctypes.c_int

        self.lib.vdb_restore_delta.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
        ]
        self.lib.vdb_restore_delta.restype = ctypes.c_int

        # CUDA Acceleration Bindings (Optional)
        self._has_cuda = hasattr(self.lib, "vdb_cuda_init")
        if self._has_cuda:
            self.lib.vdb_cuda_init.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.vdb_cuda_init.restype = ctypes.c_int

            self.lib.vdb_cuda_shutdown.argtypes = [ctypes.c_void_p]
            self.lib.vdb_cuda_shutdown.restype = ctypes.c_int

            self.lib.vdb_cuda_is_available.argtypes = [ctypes.c_void_p]
            self.lib.vdb_cuda_is_available.restype = ctypes.c_int

            self.lib.vdb_cuda_batch_search.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p
            ]
            self.lib.vdb_cuda_batch_search.restype = ctypes.c_int

            self.lib.vdb_cuda_query_planetary_grid.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_void_p
            ]
            self.lib.vdb_cuda_query_planetary_grid.restype = ctypes.c_longlong

        if hasattr(self.lib, "vdb_shrink_to_fit"):
            self.lib.vdb_shrink_to_fit.argtypes = [ctypes.c_void_p]
            self.lib.vdb_shrink_to_fit.restype = ctypes.c_int

        # Start GraalVM Isolate (Suppress stderr temporarily to hide sun.misc.Unsafe deprecation warnings)
        import os, sys
        try:
            null_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stderr_fd = os.dup(2)
            os.dup2(null_fd, 2)
        except Exception:
            null_fd = None
            saved_stderr_fd = None

        try:
            status = self.lib.graal_create_isolate(None, ctypes.byref(self.isolate), ctypes.byref(self.thread))
            if status != 0:
                raise RuntimeError(f"Failed to create GraalVM Native Image isolate (status={status})")
                
            status = self.lib.vdb_init(self.thread)
            if status != 0:
                raise PithosNativeError(status, "Failed to initialize Pithos database coordinator.")
        finally:
            if saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            if null_fd is not None:
                os.close(null_fd)

    def check_status(self, status: int, action: str = "operation"):
        if status != 0:
            raise PithosNativeError(status, f"Failed to execute {action}.")

    def create_isolate(self):
        """Creates an ephemeral GraalVM isolate and returns (isolate, thread)."""
        iso = ctypes.POINTER(GraalIsolate)()
        thr = ctypes.POINTER(GraalIsolateThread)()
        
        import os, sys
        try:
            null_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stderr_fd = os.dup(2)
            os.dup2(null_fd, 2)
        except Exception:
            null_fd = None
            saved_stderr_fd = None

        try:
            status = self.lib.graal_create_isolate(None, ctypes.byref(iso), ctypes.byref(thr))
        finally:
            if saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            if null_fd is not None:
                os.close(null_fd)
                
        if status != 0:
            raise RuntimeError(f"Failed to create GraalVM Native Image isolate (status={status})")
        return iso, thr

    def tear_down_isolate(self, thread) -> None:
        """Tears down a GraalVM isolate thread, returning all allocated heap pages to the OS."""
        if thread:
            self.lib.graal_tear_down_isolate(thread)

    from contextlib import contextmanager

    @contextmanager
    def isolated_context(self):
        """
        Context manager that yields an ephemeral GraalVM isolate thread and
        guarantees full teardown upon exit.
        """
        iso, thr = self.create_isolate()
        try:
            yield thr
        finally:
            self.tear_down_isolate(thr)

    def reset_isolate(self) -> None:
        """
        Tears down the active GraalVM isolate and initializes a fresh coordinator.
        Use this to reclaim all native memory during long-running pipelines.
        """
        with self._lock:
            if self.thread:
                try:
                    self.lib.vdb_close(self.thread)
                except Exception:
                    pass
                try:
                    self.lib.graal_tear_down_isolate(self.thread)
                except Exception:
                    pass
                self.isolate = None
                self.thread = None

            self.isolate = ctypes.POINTER(GraalIsolate)()
            self.thread = ctypes.POINTER(GraalIsolateThread)()
            status = self.lib.graal_create_isolate(None, ctypes.byref(self.isolate), ctypes.byref(self.thread))
            if status != 0:
                raise RuntimeError(f"Failed to create GraalVM Native Image isolate (status={status})")
            status = self.lib.vdb_init(self.thread)
            if status != 0:
                raise PithosNativeError(status, "Failed to initialize Pithos database coordinator.")

    def shrink_to_fit(self) -> None:
        """
        Triggers explicit GraalVM garbage collection, system malloc_trim (on Linux),
        and Python GC to release unreferenced memory to the operating system.
        """
        if self.thread and hasattr(self.lib, "vdb_shrink_to_fit"):
            try:
                self.lib.vdb_shrink_to_fit(self.thread)
            except Exception:
                pass

        try:
            import sys
            if sys.platform.startswith("linux"):
                try:
                    libc = ctypes.CDLL(None)
                    if hasattr(libc, "malloc_trim"):
                        libc.malloc_trim(0)
                except Exception:
                    pass
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    if hasattr(libc, "malloc_trim"):
                        libc.malloc_trim(0)
                except Exception:
                    pass
        except Exception:
            pass

        import gc
        gc.collect()

    def close(self) -> None:
        """Closes the coordinator and destroys the native isolate."""
        with self._lock:
            if self.thread:
                try:
                    self.lib.vdb_close(self.thread)
                except Exception:
                    pass
                try:
                    self.lib.graal_tear_down_isolate(self.thread)
                except Exception:
                    pass
                self.isolate = None
                self.thread = None


def reset_isolate(lib_path: Optional[str] = None) -> None:
    """Tears down the active GraalVM isolate and initializes a fresh coordinator."""
    NativeBindings(lib_path).reset_isolate()


def shrink_to_fit(lib_path: Optional[str] = None) -> None:
    """Explicitly reclaims unused memory across GraalVM Native Image, glibc, and Python runtime."""
    NativeBindings(lib_path).shrink_to_fit()

