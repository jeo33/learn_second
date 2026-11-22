// Lesson 9: Full L2 SIMD vs PQ LUT/lookup and memory locality.
// Compile: g++ -std=c++17 -O3 -march=native -Wall -Wextra lesson9_pq_cpu_bench.cpp -o lesson9
// Run:     ./lesson9 --mode all --n 131072 --pq 32
// This is a synthetic CPU/RAM microbenchmark, NOT a DiskANN end-to-end benchmark.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#define LESSON_X86 1
#include <immintrin.h>
#else
#define LESSON_X86 0
#endif

constexpr std::size_t D = 128;
constexpr std::size_t K = 256;
static volatile double g_sink = 0.0; // Make every measured result observable.

struct Options {
    std::size_t n = 131072;
    std::size_t chunks = 32;
    std::size_t reps = 0; // Zero: choose mode-specific defaults.
    std::string mode = "all";
};

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "Usage: ./lesson9 [--mode MODE] [--n N] [--pq 16|32|64] [--reps R]\n"
                      << "MODE: all, lut, pq_seq, pq_rand, gather_pq, full_seq, full_rand\n"
                      << "Defaults: N=131072, PQ32; each mode has its own repeat count.\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
        const std::string v = argv[++i];
        if (a == "--mode") opt.mode = v;
        else if (a == "--n") opt.n = std::stoull(v);
        else if (a == "--pq") opt.chunks = std::stoull(v);
        else if (a == "--reps") opt.reps = std::stoull(v);
        else throw std::runtime_error("Unknown argument: " + a);
    }
    if (opt.n == 0 || opt.chunks == 0 || D % opt.chunks != 0)
        throw std::runtime_error("N must be positive, and PQ chunks must divide 128");
    if (opt.chunks != 16 && opt.chunks != 32 && opt.chunks != 64)
        throw std::runtime_error("Use --pq 16, 32, or 64 for these lessons");
    const std::vector<std::string> allowed = {
        "all", "lut", "pq_seq", "pq_rand", "gather_pq", "full_seq", "full_rand"
    };
    if (std::find(allowed.begin(), allowed.end(), opt.mode) == allowed.end())
        throw std::runtime_error("Unknown mode: " + opt.mode);
    return opt;
}

float full_l2_scalar(const float* q, const float* x) {
    float sum = 0.0f;
    for (std::size_t d = 0; d < D; ++d) {
        const float diff = q[d] - x[d];
        sum += diff * diff;
    }
    return sum;
}

#if LESSON_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
float full_l2_avx2(const float* q, const float* x) {
    __m256 acc = _mm256_setzero_ps();
    for (std::size_t d = 0; d < D; d += 8) {
        const __m256 qv = _mm256_loadu_ps(q + d);
        const __m256 xv = _mm256_loadu_ps(x + d);
        const __m256 diff = _mm256_sub_ps(qv, xv);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float sum = 0.0f;
    for (float v : lanes) sum += v;
    return sum;
}
#endif

// Offline codebook layout: [dimension][centroid], K floats per dimension.
// LUT layout: [PQ chunk][centroid], K floats per chunk.
void build_lut(const float* query,
               const std::vector<float>& codebook,
               std::vector<float>& lut,
               std::size_t chunks) {
    std::fill(lut.begin(), lut.end(), 0.0f);
    const std::size_t dims_per_chunk = D / chunks;
    for (std::size_t c = 0; c < chunks; ++c) {
        float* row = lut.data() + c * K;
        for (std::size_t d = c * dims_per_chunk; d < (c + 1) * dims_per_chunk; ++d) {
            const float qd = query[d];
            const float* center_values = codebook.data() + d * K;
            for (std::size_t k = 0; k < K; ++k) {
                const float diff = qd - center_values[k];
                row[k] += diff * diff;
            }
        }
    }
}

// Query LUT is already built. No subtraction or multiplication per candidate.
float pq_lookup(const float* lut, const std::uint8_t* code, std::size_t chunks) {
    float sum = 0.0f;
    for (std::size_t c = 0; c < chunks; ++c)
        sum += lut[c * K + code[c]];
    return sum;
}

struct Data {
    std::vector<float> full;               // N * D FP32 values
    std::vector<std::uint8_t> pq;          // N * C byte codes
    std::vector<float> codebook;           // D * K FP32 values (synthetic)
    std::vector<float> query;              // D FP32 values
    std::vector<float> lut;                // C * K FP32 values
    std::vector<std::uint32_t> shuffled;   // Random permutation of node IDs
};

Data make_data(std::size_t n, std::size_t chunks) {
    Data data;
    data.full.resize(n * D);
    data.pq.resize(n * chunks);
    data.codebook.resize(D * K);
    data.query.resize(D);
    data.lut.resize(chunks * K);
    data.shuffled.resize(n);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> cdist(0, 255);

    for (float& f : data.codebook) f = fdist(rng);
    for (float& f : data.query) f = fdist(rng);

    // Synthetic but internally consistent representation:
    // each chunk's PQ code points to a subvector centroid; full vector = centroid + noise.
    const std::size_t subdim = D / chunks;
    for (std::size_t id = 0; id < n; ++id) {
        for (std::size_t c = 0; c < chunks; ++c) {
            const auto code = static_cast<std::uint8_t>(cdist(rng));
            data.pq[id * chunks + c] = code;
            for (std::size_t j = 0; j < subdim; ++j) {
                const std::size_t d = c * subdim + j;
                data.full[id * D + d] = data.codebook[d * K + code] + 0.01f * fdist(rng);
            }
        }
    }
    std::iota(data.shuffled.begin(), data.shuffled.end(), 0u);
    std::shuffle(data.shuffled.begin(), data.shuffled.end(), rng);
    build_lut(data.query.data(), data.codebook, data.lut, chunks);
    return data;
}

struct Result {
    double milliseconds;
    double ns_per_operation;
};

template <typename F>
Result measure(const std::string& name, std::size_t operations_per_round,
               std::size_t reps, F&& one_round) {
    // Warmup; not included in chrono timing.
    for (int i = 0; i < 2; ++i) g_sink += one_round(static_cast<std::size_t>(i));
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < reps; ++r) {
        g_sink += one_round(r);
    }
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double ns = ms * 1e6 / (static_cast<double>(reps) * operations_per_round);
    std::cout << std::left << std::setw(11) << name << "  "
              << std::right << std::setw(8) << reps << " rounds  "
              << std::setw(11) << std::fixed << std::setprecision(2) << ms << " ms  "
              << std::setw(10) << std::setprecision(2) << ns
              << (name == "lut" ? " ns/LUT" : " ns/candidate") << '\n';
    return {ms, ns};
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        const std::size_t N = opt.n, C = opt.chunks;
        const Data data = make_data(N, C);

        bool use_avx2 = false;
#if LESSON_X86 && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        use_avx2 = __builtin_cpu_supports("avx2");
#endif
        std::cout << "D=" << D << " N=" << N << " PQ=" << C
                  << " LUT=" << C * K * sizeof(float) / 1024 << " KiB"
                  << " full=" << N * D * sizeof(float) / (1024 * 1024) << " MiB"
                  << " PQ_codes=" << N * C / (1024 * 1024) << " MiB"
                  << " Full_L2=" << (use_avx2 ? "AVX2 intrinsics" : "scalar fallback")
                  << '\n';

        auto chosen_reps = [&](std::size_t default_reps) {
            return opt.reps ? opt.reps : default_reps;
        };
        auto wants = [&](const char* name) {
            return opt.mode == "all" || opt.mode == name;
        };

        if (wants("lut")) {
            // Change query slightly every time so LUT build cannot be hoisted out of loop.
            std::vector<float> query = data.query;
            std::vector<float> lut(C * K);
            measure("lut", 1, chosen_reps(2000), [&](std::size_t r) {
                query[0] += 1e-5f;
                build_lut(query.data(), data.codebook, lut, C);
                return static_cast<double>(lut[r % lut.size()]);
            });
        }
        if (wants("pq_seq")) {
            measure("pq_seq", N, chosen_reps(200), [&](std::size_t) {
                double sum = 0.0;
                for (std::size_t id = 0; id < N; ++id)
                    sum += pq_lookup(data.lut.data(), data.pq.data() + id * C, C);
                return sum;
            });
        }
        if (wants("pq_rand")) {
            measure("pq_rand", N, chosen_reps(200), [&](std::size_t) {
                double sum = 0.0;
                for (std::size_t i = 0; i < N; ++i) {
                    const std::size_t id = data.shuffled[i];
                    sum += pq_lookup(data.lut.data(), data.pq.data() + id * C, C);
                }
                return sum;
            });
        }
        if (wants("gather_pq")) {
            // DiskANN-style: gather codes of a batch of (random) node IDs to scratch,
            // then evaluate distances with a contiguous scratch buffer.
            constexpr std::size_t BATCH = 64;
            std::vector<std::uint8_t> scratch(BATCH * C);
            measure("gather_pq", N, chosen_reps(200), [&](std::size_t) {
                double sum = 0.0;
                for (std::size_t first = 0; first < N; first += BATCH) {
                    const std::size_t count = std::min(BATCH, N - first);
                    for (std::size_t j = 0; j < count; ++j) {
                        const std::size_t id = data.shuffled[first + j];
                        std::memcpy(scratch.data() + j * C, data.pq.data() + id * C, C);
                    }
                    for (std::size_t j = 0; j < count; ++j)
                        sum += pq_lookup(data.lut.data(), scratch.data() + j * C, C);
                }
                return sum;
            });
        }
        if (wants("full_seq") || wants("full_rand")) {
            auto full_distance = [&](std::size_t id) -> float {
                const float* x = data.full.data() + id * D;
#if LESSON_X86 && (defined(__GNUC__) || defined(__clang__))
                if (use_avx2) return full_l2_avx2(data.query.data(), x);
#endif
                return full_l2_scalar(data.query.data(), x);
            };
            if (wants("full_seq")) {
                measure("full_seq", N, chosen_reps(30), [&](std::size_t) {
                    double sum = 0.0;
                    for (std::size_t id = 0; id < N; ++id) sum += full_distance(id);
                    return sum;
                });
            }
            if (wants("full_rand")) {
                measure("full_rand", N, chosen_reps(30), [&](std::size_t) {
                    double sum = 0.0;
                    for (std::size_t i = 0; i < N; ++i)
                        sum += full_distance(data.shuffled[i]);
                    return sum;
                });
            }
        }
        std::cout << "Checksum (avoid optimizer removal): " << std::fixed
                  << std::setprecision(2) << g_sink << '\n';
        std::cout << "Note: CPU/RAM-only synthetic data; no SSD I/O and no ANN recall results.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
