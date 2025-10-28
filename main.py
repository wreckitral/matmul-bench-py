import random
import numpy as np
import torch
import time


def naive_matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    if n != len(B):
        raise ValueError(f"Cannot multiply: A is {m}x{n}, B is {len(B)}x{p}")

    C = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C


def calculate_gflops(M, N, K, time_seconds):
    flops = 2 * M * N * K
    gflops = (flops / time_seconds) / 1e9

    return gflops


def benchmark(func, A, B, warmup=5, iterations=100, sync_gpu=False):
    # warmup
    for _ in range(warmup):
        _ = func(A, B)
        if sync_gpu:
            torch.cuda.synchronize()

    # benchmark
    times = []
    for _ in range(iterations):
        if sync_gpu:
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = func(A, B)

        if sync_gpu:
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)


def populate_array(size=1024):
    return [random.random() for _ in range(size)]


def populate_matrix(rows=1024, cols=1024):
    return [[random.random() for _ in range(cols)] for _ in range(rows)]


def main():
    sizes = [64, 128, 1024, 2048]

    device_gpu = 'cuda' if torch.cuda.is_available() else None

    for size in sizes:
        M = N = K = size
        print(f"Matrix size: {M}x{K} @ {K}x{N} = {M}x{N}")
        print("-" * 80)

        # create test matrices
        tensor1_cpu = torch.randn(M, K, dtype=torch.float32)
        tensor2_cpu = torch.randn(K, N, dtype=torch.float32)

        # numpy matrices
        np_A = tensor1_cpu.numpy()
        np_B = tensor2_cpu.numpy()

        results = []

        # 1. numpy (CPU)
        avg_time = benchmark(np.dot, np_A, np_B, warmup=3, iterations=50)
        gflops = calculate_gflops(M, N, K, avg_time)
        results.append(("NumPy (CPU)", avg_time, gflops))

        # 2. pytorch CPU
        avg_time = benchmark(torch.matmul, tensor1_cpu, tensor2_cpu, warmup=3,
                             iterations=50)
        gflops = calculate_gflops(M, N, K, avg_time)
        results.append(("PyTorch (CPU)", avg_time, gflops))

        # 3. pytorch GPU
        if device_gpu:
            tensor_gpu = tensor1_cpu.to(device_gpu)
            tensor2_gpu = tensor2_cpu.to(device_gpu)
            avg_time = benchmark(torch.matmul, tensor_gpu, tensor2_gpu,
                                 warmup=10, iterations=100, sync_gpu=True)
            gflops = calculate_gflops(M, N, K, avg_time)
            results.append(("PyTorch (GPU)", avg_time, gflops))

        # 4. naive matmul (only for small sizes)
        if size <= 512:
            avg_time = benchmark(naive_matmul, tensor1_cpu, tensor2_cpu,
                                 warmup=1, iterations=3)
            gflops = calculate_gflops(M, N, K, avg_time)
            results.append(("Naive (CPU)", avg_time, gflops))

        # print results
        print(f"{'Method':<20} {'Time (ms)':<15} {'GFLOPS':<15} {'Speedup':<10}")
        baseline_time = results[0][1]  # use numpy as baseline

        for method, avg_time, gflops in results:
            speedup = baseline_time / avg_time
            print(f"{method:<20} {avg_time*1000:>12.4f} {gflops:>12.2f} {speedup:>8.2f}x")

        print("\n" + "="*80 + "\n")

    # tensor1 = torch.randn(8, 8)
    # tensor2 = torch.randn(8, 8)
    # tensor3 = torch.matmul(tensor1, tensor2)

    # tensor_naive = naive_matmul(tensor1, tensor2)
    # tensor_np = np.dot(tensor1, tensor2)

    # print(tensor3)
    # print(tensor_naive)
    # print(tensor_np)

    # matrix1 = populate_matrix(8, 8)
    # matrix2 = populate_matrix(8, 8)

    # matrix3 = naive_matmul(matrix1, matrix2)
    # matrix_np = np.dot(matrix1, matrix2)

    # print(matrix3)
    # print(matrix_np)


if __name__ == "__main__":
    main()
