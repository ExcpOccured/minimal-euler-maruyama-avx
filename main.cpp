#include <iostream>
#include <vector>
#include <iomanip> 

#include "src/euler_maruyama_avx.h"

int main() {
    const float startTime = 0.0f;
    const float endTime = 1.0f;
    const float initial_dt = 0.01f;
    const int num_trajectories = 8; 

    alignas(32) float P[num_trajectories];

    for (int i = 0; i < num_trajectories; ++i) {
        P[i] = 0.5f;
    }

    std::cout << "New state of trajectories:" << std::endl;
    for (int i = 0; i < num_trajectories; ++i) {
        std::cout << "  P[" << i << "] = " << std::fixed << std::setprecision(6) << P[i] << std::endl;
    }

    std::cout << "\nRunning SDE solver from t=" << startTime << " to t=" << endTime << "..." << std::endl;

    euler_maruyama_avx(P, startTime, endTime, initial_dt);

    std::cout << "\nFinal state of trajectories:" << std::endl;
    for (int i = 0; i < num_trajectories; ++i) {
        std::cout << "  P[" << i << "] = " << std::fixed << std::setprecision(6) << P[i] << std::endl;
    }

    return 0;
}