
// Experiment 5: Neural Network Layers in CUDA
// Matrix multiplication with bias, ReLU, and Sigmoid
// CPU and GPU forward-pass comparison

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int BATCH = 256;
constexpr int INPUT_SIZE = 128;
constexpr int HIDDEN_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;
constexpr int REPEATS = 20;

// ---------------- CUDA KERNELS ----------------

__global__ void matrixMulBiasKernel(
    const float* A, const float* B, const float* bias,
    float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum + bias[col];
    }
}

__global__ void reluKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}

__global__ void sigmoidKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        values[idx] = 1.0f / (1.0f + expf(-values[idx]));
    }
}

// ---------------- CPU FUNCTIONS ----------------

void cpuMatMulBias(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& bias,
    std::vector<float>& C,
    int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum + bias[col];
        }
    }
}

void cpuReLU(std::vector<float>& values)
{
    for (float& x : values) {
        x = std::max(0.0f, x);
    }
}

void cpuSigmoid(std::vector<float>& values)
{
    for (float& x : values) {
        x = 1.0f / (1.0f + std::exp(-x));
    }
}

void cpuForwardPass(
    const std::vector<float>& X,
    const std::vector<float>& W1,
    const std::vector<float>& b1,
    const std::vector<float>& W2,
    const std::vector<float>& b2,
    std::vector<float>& output)
{
    std::vector<float> z1(BATCH * HIDDEN_SIZE);

    cpuMatMulBias(
        X, W1, b1, z1,
        BATCH, HIDDEN_SIZE, INPUT_SIZE
    );

    cpuReLU(z1);

    std::vector<float> z2(BATCH * OUTPUT_SIZE);

    cpuMatMulBias(
        z1, W2, b2, z2,
        BATCH, OUTPUT_SIZE, HIDDEN_SIZE
    );

    cpuSigmoid(z2);

    output = z2;
}

// ---------------- GPU FORWARD PASS ----------------

void launchGpuForwardPass(
    const float* dX,
    const float* dW1,
    const float* db1,
    float* dZ1,
    const float* dW2,
    const float* db2,
    float* dZ2)
{
    dim3 block(16, 16);

    dim3 grid1(
        (HIDDEN_SIZE + block.x - 1) / block.x,
        (BATCH + block.y - 1) / block.y
    );

    matrixMulBiasKernel<<<grid1, block>>>(
        dX, dW1, db1, dZ1,
        BATCH, HIDDEN_SIZE, INPUT_SIZE
    );

    int threads = 256;
    int hiddenCount = BATCH * HIDDEN_SIZE;
    int blocks = (hiddenCount + threads - 1) / threads;

    reluKernel<<<blocks, threads>>>(dZ1, hiddenCount);

    dim3 grid2(
        (OUTPUT_SIZE + block.x - 1) / block.x,
        (BATCH + block.y - 1) / block.y
    );

    matrixMulBiasKernel<<<grid2, block>>>(
        dZ1, dW2, db2, dZ2,
        BATCH, OUTPUT_SIZE, HIDDEN_SIZE
    );

    int outputCount = BATCH * OUTPUT_SIZE;
    int outputBlocks = (outputCount + threads - 1) / threads;

    sigmoidKernel<<<outputBlocks, threads>>>(dZ2, outputCount);

    CUDA_CHECK(cudaGetLastError());
}

// ---------------- KERNEL TESTS ----------------

bool testMatrixMultiplication()
{
    float hA[] = {1, 2, 3, 4, 5, 6};
    float hB[] = {1, 2, 3, 4, 5, 6};
    float hBias[] = {1, -1};
    float hC[4] = {};

    float *dA, *dB, *dBias, *dC;

    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dBias, sizeof(hBias)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hBias, sizeof(hBias), cudaMemcpyHostToDevice));

    matrixMulBiasKernel<<<dim3(1, 1), dim3(16, 16)>>>(
        dA, dB, dBias, dC, 2, 2, 3
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));

    bool passed =
        std::fabs(hC[0] - 23.0f) < 1e-5f &&
        std::fabs(hC[1] - 27.0f) < 1e-5f &&
        std::fabs(hC[2] - 50.0f) < 1e-5f &&
        std::fabs(hC[3] - 63.0f) < 1e-5f;

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dBias));
    CUDA_CHECK(cudaFree(dC));

    return passed;
}

bool testReLU()
{
    float hValues[] = {-2, 0, 3, -4, 5};
    float expected[] = {0, 0, 3, 0, 5};

    float* dValues;

    CUDA_CHECK(cudaMalloc(&dValues, sizeof(hValues)));
    CUDA_CHECK(cudaMemcpy(
        dValues, hValues, sizeof(hValues), cudaMemcpyHostToDevice
    ));

    reluKernel<<<1, 256>>>(dValues, 5);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hValues, dValues, sizeof(hValues), cudaMemcpyDeviceToHost
    ));

    bool passed = true;

    for (int i = 0; i < 5; ++i) {
        if (std::fabs(hValues[i] - expected[i]) > 1e-5f) {
            passed = false;
        }
    }

    CUDA_CHECK(cudaFree(dValues));

    return passed;
}

bool testSigmoid()
{
    float hValues[] = {-2, 0, 2};

    float* dValues;

    CUDA_CHECK(cudaMalloc(&dValues, sizeof(hValues)));
    CUDA_CHECK(cudaMemcpy(
        dValues, hValues, sizeof(hValues), cudaMemcpyHostToDevice
    ));

    sigmoidKernel<<<1, 256>>>(dValues, 3);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hValues, dValues, sizeof(hValues), cudaMemcpyDeviceToHost
    ));

    bool passed =
        std::fabs(hValues[0] - (1.0f / (1.0f + std::exp(2.0f)))) < 1e-5f &&
        std::fabs(hValues[1] - 0.5f) < 1e-5f &&
        std::fabs(hValues[2] - (1.0f / (1.0f + std::exp(-2.0f)))) < 1e-5f;

    CUDA_CHECK(cudaFree(dValues));

    return passed;
}

// ---------------- MAIN PROGRAM ----------------

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found." << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\n============================================\n";
    std::cout << " EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA\n";
    std::cout << "============================================\n";

    std::cout << "GPU detected: " << prop.name << "\n\n";

    std::cout << "Testing CUDA kernels...\n";

    bool matrixPassed = testMatrixMultiplication();
    bool reluPassed = testReLU();
    bool sigmoidPassed = testSigmoid();

    std::cout << "Matrix multiplication + bias: "
              << (matrixPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "ReLU: "
              << (reluPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "Sigmoid: "
              << (sigmoidPassed ? "PASSED" : "FAILED") << "\n";

    if (!matrixPassed || !reluPassed || !sigmoidPassed) {
        std::cerr << "\nKernel test failed. Stopping experiment.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll kernel tests passed!\n";

    // Initialize deterministic input data

    std::vector<float> X(BATCH * INPUT_SIZE);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> b1(HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> b2(OUTPUT_SIZE);

    for (size_t i = 0; i < X.size(); ++i) {
        X[i] = 0.5f * std::sin(static_cast<float>(i) * 0.013f);
    }

    for (size_t i = 0; i < W1.size(); ++i) {
        W1[i] = 0.08f * std::cos(static_cast<float>(i) * 0.017f);
    }

    for (size_t i = 0; i < b1.size(); ++i) {
        b1[i] = 0.01f * static_cast<float>(i % 7);
    }

    for (size_t i = 0; i < W2.size(); ++i) {
        W2[i] = 0.10f * std::sin(static_cast<float>(i) * 0.019f);
    }

    for (size_t i = 0; i < b2.size(); ++i) {
        b2[i] = 0.02f * static_cast<float>(i % 5);
    }

    std::vector<float> cpuOutput(BATCH * OUTPUT_SIZE);
    std::vector<float> gpuOutput(BATCH * OUTPUT_SIZE);

    // CPU timing

    auto cpuStart = std::chrono::steady_clock::now();

    for (int i = 0; i < REPEATS; ++i) {
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);
    }

    auto cpuEnd = std::chrono::steady_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart
        ).count() / REPEATS;

    // Allocate GPU memory

    float *dX, *dW1, *db1, *dZ1, *dW2, *db2, *dZ2;

    CUDA_CHECK(cudaMalloc(&dX, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ1, BATCH * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ2, BATCH * OUTPUT_SIZE * sizeof(float)));

    // Copy data to GPU for kernel-only timing

    CUDA_CHECK(cudaMemcpy(
        dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice
    ));

    // Warm-up

    launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU kernel-only timing

    cudaEvent_t startEvent, stopEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int i = 0; i < REPEATS; ++i) {
        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float totalGpuKernelTime = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalGpuKernelTime, startEvent, stopEvent
    ));

    double gpuKernelTime = totalGpuKernelTime / REPEATS;

    // GPU end-to-end timing, including transfers

    auto gpuStart = std::chrono::steady_clock::now();

    for (int i = 0; i < REPEATS; ++i) {
        CUDA_CHECK(cudaMemcpy(
            dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice
        ));

        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            gpuOutput.data(), dZ2,
            gpuOutput.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }

    auto gpuEnd = std::chrono::steady_clock::now();

    double gpuEndToEndTime =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart
        ).count() / REPEATS;

    // Compare CPU and GPU results

    float maxDifference = 0.0f;

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        maxDifference = std::max(
            maxDifference,
            std::fabs(cpuOutput[i] - gpuOutput[i])
        );
    }

    // Display results

    std::cout << "\n============================================\n";
    std::cout << " PERFORMANCE RESULTS\n";
    std::cout << "============================================\n";

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Network dimensions:\n";
    std::cout << "Input:  " << BATCH << " x " << INPUT_SIZE << "\n";
    std::cout << "Hidden: " << BATCH << " x " << HIDDEN_SIZE << "\n";
    std::cout << "Output: " << BATCH << " x " << OUTPUT_SIZE << "\n\n";

    std::cout << "Average CPU execution time: "
              << cpuTime << " ms\n";

    std::cout << "Average GPU kernel-only time: "
              << gpuKernelTime << " ms\n";

    std::cout << "Average GPU end-to-end time: "
              << gpuEndToEndTime << " ms\n";

    std::cout << "Kernel-only speedup: "
              << cpuTime / gpuKernelTime << "x\n";

    std::cout << "End-to-end speedup: "
              << cpuTime / gpuEndToEndTime << "x\n";

    std::cout << "\nMaximum absolute output difference: "
              << maxDifference << "\n";

    std::cout << "CPU and GPU outputs: "
              << (maxDifference < 1e-4f
                      ? "MATCH (within tolerance)"
                      : "DO NOT MATCH")
              << "\n";

    std::cout << "\nFirst 10 GPU output values:\n";

    for (int i = 0; i < 10; ++i) {
        std::cout << gpuOutput[i] << " ";
    }

    std::cout << "\n\nExperiment completed.\n";

    // Cleanup

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dW1));
    CUDA_CHECK(cudaFree(db1));
    CUDA_CHECK(cudaFree(dZ1));
    CUDA_CHECK(cudaFree(dW2));
    CUDA_CHECK(cudaFree(db2));
    CUDA_CHECK(cudaFree(dZ2));

    return (maxDifference < 1e-4f)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
