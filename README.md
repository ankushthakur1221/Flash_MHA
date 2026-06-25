# High-Performance CUDA Attention Kernels with TileLang

This repository contains high-performance GPU self-attention kernels (including forward and backward passes) implemented in **TileLang** and integrated as a fully autograd-capable PyTorch module.

TileLang is a tiled domain-specific language (DSL) for high-performance GPU programming. It is built on top of Apache TVM and compiles high-level Pythonic kernel representations into highly optimized Triton and CUDA code.

---

## 🚀 Features

*   **Forward Attention JIT Kernel (`kernel.py`)**: A tiled forward pass self-attention implementation featuring causal masking, online softmax reductions for numerical stability, and JIT compilation.
*   **Fully Autograd-Capable Module (`Mla.py` & `MLAkernels.py`)**: Seamless integration with PyTorch's Autograd system. Both forward and backward passes of FlashAttention are custom-routed to JIT-compiled TileLang CUDA kernels, enabling seamless backpropagation in larger neural networks.
*   **Hardware Autotuning**: Uses TileLang's built-in autotuning decorator (`@tilelang.autotune`) to search block sizes (`block_M`, `block_N`), stage counts, and warp thread configurations to maximize GPU occupancy and memory bandwidth.
*   **Testing & Verification (`test.py`)**: Out-of-the-box validation script to verify forward activations and backward gradients against PyTorch's native `F.scaled_dot_product_attention`.

---

## 📂 Project Structure

```bash
├── MLAkernels.py     # Custom forward/backward JIT kernels & PyTorch _attention Autograd wrapper
├── Mla.py            # PyTorch MultiHeadAttention nn.Module integrated with TileLang backends
├── kernel.py         # Forward-only autotuned self-attention JIT kernel
├── test.py           # Verification script for correctness testing (GPU only, safe exit on CPU)
├── requirements.txt  # Project Python dependencies
└── .gitignore        # Standard python & JIT build artifact ignores
```

---

## 🛠️ Installation

### Prerequisites
*   **NVIDIA GPU** with CUDA support.
*   **Python 3.8+**

### Steps
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/MultiHeadAttention-TileLang.git
    cd MultiHeadAttention-TileLang
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: To install the latest nightly builds of TileLang (highly recommended for performance improvements), use:*
    ```bash
    pip install tilelang -f https://tile-ai.github.io/whl/nightly
    ```

---

## 🧪 Testing and Verification

To run correctness tests against the PyTorch reference implementation, run:

```bash
python test.py
```

### Script Behavior
*   **On CUDA-capable machines**: The script will JIT compile the TileLang kernels, run the forward and backward passes on a CUDA device, and check the values against PyTorch's native scaled dot product attention with strict tolerances.
*   **On CPU-only machines (e.g., Apple Silicon Macs)**: Since CUDA is a compilation requirement for JIT compiling these kernels, the script will gracefully catch the absence of a GPU and print a informative warning before exiting.

---

## 📄 References
*   [TileLang Official GitHub](https://github.com/tile-ai/tilelang)
*   FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al.)
