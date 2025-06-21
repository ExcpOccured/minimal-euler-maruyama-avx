#include <immintrin.h>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

// Drift (mu) and diffusion (sigma) coefficients
// Example: dp(t) = 0.1*p*dt + 0.2*|p|*dW(t)
inline float mu(float p, float t) { 
    return 0.1f * p; 
}

inline float sigma(float p, float t) { 
    return 0.2f * std::abs(p); 
}

/**
 * @brief Solves a batch of SDEs using the Euler-Maruyama method with AVX-512.
 * * @param P Pointer to an array of initial values (size N). Results are written back to this array.
 * @param t0 Start time.
 * @param T End time.
 * @param initial_dt The initial time step.
 */
void euler_maruyama_avx(float* P, float t0, float T, float initial_dt) {
    const int N = 8; // Vector dimension for AVX-512 float (8 * 32-bit)
    const float dt_min = 1e-4f;
    const float dt_max = 0.05f;
    const float grad_threshold = 0.2f;

    std::mt19937 gen(42);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    float t = t0;
    float dt = initial_dt;
    
    // Store last gradients for adaptive step calculation
    float last_grad[N];
    for(int i = 0; i < N; ++i) {
        last_grad[i] = mu(P[i], t);
    }

    while (t < T) {
        // Load current values into an AVX register
        __m256 p_vec = _mm256_loadu_ps(P);

        // Prepare coefficients for the batch
        float mu_arr[N], sigma_arr[N], dW_arr[N];
        for(int i = 0; i < N; ++i) {
            mu_arr[i] = mu(P[i], t);
            sigma_arr[i] = sigma(P[i], t);
            dW_arr[i] = norm(gen) * std::sqrt(dt);
        }

        __m256 mu_vec = _mm256_loadu_ps(mu_arr);
        __m256 sigma_vec = _mm256_loadu_ps(sigma_arr);
        __m256 dW_vec = _mm256_loadu_ps(dW_arr);
        __m256 dt_vec = _mm256_set1_ps(dt);

        // Main Euler-Maruyama step, performed in parallel for N values
        __m256 dP = _mm256_add_ps(
            _mm256_mul_ps(mu_vec, dt_vec),
            _mm256_mul_ps(sigma_vec, dW_vec)
        );
        p_vec = _mm256_add_ps(p_vec, dP);
        _mm256_storeu_ps(P, p_vec);

        // --- Adaptive step size logic ---
        float grad_arr[N];
        for (int i = 0; i < N; ++i) {
            grad_arr[i] = mu(P[i], t + dt);
        }

        float max_grad_diff = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_grad_diff = std::max(max_grad_diff, std::abs(grad_arr[i] - last_grad[i]));
            last_grad[i] = grad_arr[i];
        }
        
        // Adjust step size based on gradient change
        if (max_grad_diff > grad_threshold) {
            dt = std::max(dt / 2.0f, dt_min);
        } else if (max_grad_diff < grad_threshold / 2.0f) {
            dt = std::min(dt * 2.0f, dt_max);
        }

        t += dt;
    }
}