#pragma once

/**
 * @brief Solves a batch of SDEs using the Euler-Maruyama method with AVX-512.
 * * This function implements the numerical solution for a system of 8 parallel
 * Stochastic Differential Equations. It uses an adaptive time step to ensure
 * stability and accuracy. The SDE is defined by the internal mu and sigma functions.
 * * @param P Pointer to an array of 8 float values. These are the initial conditions.
 * The final results will be written back into this same array.
 * @param t0 The starting time for the simulation.
 * @param T The end time for the simulation.
 * @param initial_dt The initial time step for the adaptive solver.
 */
void euler_maruyama_avx(float* P, float t0, float T, float initial_dt);