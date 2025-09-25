# Copyright (c) 2025, XXX.
#
# Thanks to Tri Dao for the original FlashAttention repository structure.
#
# Sorry, in this version, to prevent theft, we're only open-sourcing the forward portion.
# Once our paper is accepted, we'll make it fully open-source.
# Please understand our difficulties, as we've experienced the pain of theft in the past.
# However, for understanding and learning, the forward portion is completely sufficient.

import sys
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# --- Configuration ---

# This directory is the root of the project
this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "rbt_attn"
# By default, we build for Ampere (80), Hopper (90). Add others if needed.
# For example: "80;86;90"
DEFAULT_CUDA_ARCHS = "80;90"


# --- Helper Functions ---

def get_package_version():
    """Reads the version from the package's __init__.py file."""
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    if version_match:
        return ast.literal_eval(version_match.group(1))
    raise RuntimeError("Version string not found.")


def get_cuda_bare_metal_version(cuda_dir):
    """Gets the CUDA version from the nvcc binary."""
    if not cuda_dir:
        return None
    try:
        raw_output = subprocess.check_output([os.path.join(cuda_dir, "bin", "nvcc"), "-V"], universal_newlines=True)
        match = re.search(r"release (\d+\.\d+)", raw_output)
        if match:
            return parse(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None

def get_cuda_archs_flags():
    """Generate gencode flags for the target CUDA architectures."""
    archs_env = os.getenv("FLASH_ATTN_CUDA_ARCHS", DEFAULT_CUDA_ARCHS)
    archs = [arch.strip() for arch in archs_env.split(";") if arch.strip()]
    
    if not archs:
        return []

    flags = []
    for arch in archs:
        try:
            # Add both compute and code flags for each architecture
            flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
        except ValueError:
            print(f"Warning: Invalid CUDA architecture '{arch}' ignored.")
            
    return flags

# --- Main Setup Logic ---

ext_modules = []
IS_WINDOWS = sys.platform == "win32"

# Only attempt to build the CUDA extension if CUDA is available
if CUDA_HOME and not os.environ.get("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE":
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_version = get_cuda_bare_metal_version(CUDA_HOME)
    if cuda_version is None:
        print("Warning: Could not determine CUDA version from nvcc. Proceeding with build.")
    else:
        print(f"Detected CUDA version: {cuda_version}")
        if cuda_version < Version("11.7"):
            raise RuntimeError("FlashAttention requires CUDA 11.7 or later.")

    # List of all source files for the forward pass
    forward_sources = [
        # Main API file
        "csrc/flash_attn/flash_api.cpp",
        # Custom kernel for compression logic
        "csrc/flash_attn/src/compress_attention.cu",
        # Kernels for different head dimensions, data types, and causal/non-causal
        "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim160_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim160_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
        # Split-K kernels for large sequence lengths
        "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
    ]

    # Common compiler flags
    cc_flag = get_cuda_archs_flags()
    
    # Compiler settings for NVCC
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
    ] + cc_flag
    
    # Use a large number of threads for parallel compilation
    num_threads = os.getenv("NVCC_THREADS", str(max(1, os.cpu_count() or 1)))
    nvcc_flags.extend(["--threads", num_threads])

    ext_modules.append(
        CUDAExtension(
            name="rbt_attn_2_cuda",
            sources=forward_sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )

# A custom build extension to use ninja for faster compilation
class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # Set a reasonable default for MAX_JOBS if not already set.
        if not os.environ.get("MAX_JOBS"):
            import psutil
            # Heuristic: use half the CPU cores, but no more than what available memory can support.
            # Each compilation job can take a few GB of RAM.
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
            max_jobs_by_mem = int(free_memory_gb / 4) # Assume ~4GB per job
            max_jobs_by_cpu = max(1, (os.cpu_count() or 2) // 2)
            os.environ["MAX_JOBS"] = str(min(max_jobs_by_cpu, max_jobs_by_mem))
        
        super().__init__(*args, **kwargs)

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "*.egg-info"
        )
    ),
    author="Mingkuan Zhao",
    description="RBT Attention: A fast and memory-efficient attention mechanism (Forward Pass Only)",
    long_description="This package provides the forward pass implementation of RBT Attention. The full source code, including the backward pass, will be released upon paper acceptance.",
    long_description_content_type="text/markdown",
    url="https://github.com/your_repo/rbt_attn",  # TODO: Update this URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License", # Assuming BSD License
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension} if ext_modules else {},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)