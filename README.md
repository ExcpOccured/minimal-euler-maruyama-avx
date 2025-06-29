# Minimal Euler-Maruyama SDE Solver (AVX-512)

This repository provides a minimal, self-contained C++ implementation of the Euler-Maruyama method for solving Stochastic Differential Equations (SDEs). 
The code is optimized with AVX-512 intrinsics for high-performance parallel computation of multiple trajectories.


## About the Code

The core of this repository is a single C++ function, `euler_maruyama_avx`, which demonstrates:
* **High Performance:** Uses AVX-512 to solve 8 SDE trajectories simultaneously.
* **Adaptive Step Size:** Dynamically adjusts the time step `dt` based on the gradient of the drift function to ensure stability and accuracy.
* **Self-Contained:** Has no external dependencies beyond a standard C++ compiler.

## Prerequisites

* A C++17 compliant compiler (e.g., `g++`, `clang++`).
* A CPU with support for the **AVX-512** instruction set.