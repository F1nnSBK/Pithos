import os
import platform
import sys
import urllib.request
import shutil
from pathlib import Path
from typing import Optional, List

LIB_NAME_BASE = "pithos"
GITHUB_REPO = "F1nnSBK/lcvk"
DEFAULT_CACHE_DIR = Path.home() / ".pithos" / "lib"

def get_platform_info() -> tuple[str, str, str]:
    """Returns (system_normalized, arch_normalized, extension)."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    
    if sys_name == "darwin":
        ext = "dylib"
        sys_tag = "macos"
        arch_tag = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    elif sys_name == "linux":
        ext = "so"
        sys_tag = "linux"
        arch_tag = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    elif sys_name == "windows":
        ext = "dll"
        sys_tag = "windows"
        arch_tag = "x86_64"
    else:
        raise OSError(f"Unsupported operating system: {platform.system()}")
        
    return sys_tag, arch_tag, ext

def get_release_asset_name(cuda: bool = False) -> str:
    """Computes the expected asset filename in GitHub Releases."""
    sys_tag, arch_tag, ext = get_platform_info()
    if cuda and sys_tag == "linux":
        return f"libpithos-linux-cuda-{arch_tag}.{ext}"
    return f"libpithos-{sys_tag}-{arch_tag}.{ext}"

def get_candidate_library_paths(custom_path: Optional[str] = None) -> List[Path]:
    """Generates candidate search paths for libpithos in priority order."""
    candidates: List[Path] = []
    
    if custom_path:
        candidates.append(Path(custom_path).expanduser().resolve())
        
    env_path = os.environ.get("PITHOS_LIB_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    sys_tag, arch_tag, ext = get_platform_info()
    
    # 1. Bundled package library directory: src/pithos/lib/ or site-packages/pithos/lib/
    package_dir = Path(__file__).parent.resolve()
    lib_dir = package_dir / "lib"
    candidates.append(lib_dir / f"libpithos.{ext}")
    candidates.append(lib_dir / f"pithos.{ext}")
    candidates.append(lib_dir / f"libpithos-{sys_tag}-{arch_tag}.{ext}")
    
    # 2. Local repository build directories (target/, build-binaries/)
    repo_root = package_dir.parent.parent
    candidates.append(repo_root / "target" / f"pithos.{ext}")
    candidates.append(repo_root / "target" / f"libpithos.{ext}")
    candidates.append(repo_root / "build-binaries" / f"libpithos-{sys_tag}-{arch_tag}.{ext}")
    candidates.append(repo_root / "build-binaries" / f"pithos.{ext}")
    
    # 3. User cache directory: ~/.pithos/lib/
    candidates.append(DEFAULT_CACHE_DIR / f"libpithos-{sys_tag}-{arch_tag}.{ext}")
    candidates.append(DEFAULT_CACHE_DIR / f"libpithos.{ext}")
    
    return candidates

def download_native_library(target_path: Path, cuda: bool = False, version: str = "latest") -> Path:
    """Downloads the pre-built native library asset from GitHub Releases."""
    asset_name = get_release_asset_name(cuda=cuda)
    
    if version == "latest":
        url = f"https://github.com/{GITHUB_REPO}/releases/latest/download/{asset_name}"
    else:
        tag = version if version.startswith("v") else f"v{version}"
        url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{asset_name}"
        
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp")
    
    sys.stderr.write(f"[Pithos] Downloading native binary from {url} to {target_path}...\n")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Pithos-Python-Client"}
        )
        with urllib.request.urlopen(req) as response, open(temp_target, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
            
        temp_target.chmod(0o755)
        temp_target.rename(target_path)
        sys.stderr.write(f"[Pithos] Download complete: {target_path}\n")
        return target_path
    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        raise RuntimeError(
            f"Failed to auto-download Pithos native library '{asset_name}' from GitHub Releases: {e}\n"
            f"You can manually compile it via 'mvn package' or set PITHOS_LIB_PATH."
        ) from e

def find_or_fetch_native_library(custom_path: Optional[str] = None, auto_download: bool = True) -> str:
    """Finds the local native library or downloads it automatically from GitHub Releases."""
    for candidate in get_candidate_library_paths(custom_path):
        if candidate.exists() and candidate.is_file():
            return str(candidate)
            
    if auto_download:
        sys_tag, arch_tag, ext = get_platform_info()
        dest_path = DEFAULT_CACHE_DIR / f"libpithos-{sys_tag}-{arch_tag}.{ext}"
        downloaded = download_native_library(dest_path)
        return str(downloaded)
        
    candidates_str = "\n  - ".join(str(p) for p in get_candidate_library_paths(custom_path))
    raise FileNotFoundError(
        f"Could not locate Pithos native library on this system ({platform.system()} {platform.machine()}).\n"
        f"Checked paths:\n  - {candidates_str}\n"
        f"Set the PITHOS_LIB_PATH environment variable or compile via 'mvn package'."
    )
