// Lesson 12: Graph-neighbor PQ trace replay (C++17, CPU/RAM-only).
// Modes: global PQ, trace-trained reorder, neighbor-packed PQ.
// This version explicitly 64-byte aligns all PQ payloads for a fair cache-line comparison.
// Default uses a SYNTHETIC clustered graph and held-out expansion traces.
// For REAL DiskANN traces, use --train-file and --test-file. One line:
//   parent_id neighbor_id_0 neighbor_id_1 ...
// Parent adjacency must be identical across occurrences. Separate training/test query sets.
// This is an isolated PQ-neighbor replay, NOT an end-to-end ANN or SSD benchmark.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

struct Options {
    size_t n = 65536, degree = 32, chunks = 32;
    size_t train_exp = 18000, test_exp = 12000, reps = 15, trials = 3;
    uint32_t seed = 42;
    std::string train_file, test_file, pq_bytes_file;
};
struct Expansion {
    uint32_t parent = 0;
    std::vector<uint32_t> neighbors;
};
using Trace = std::vector<Expansion>;

static size_t read_integer(const char* val, const char* name) {
    std::string s(val);
    size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos);
    if (pos != s.size() || v == 0 || v > std::numeric_limits<size_t>::max())
        throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    return static_cast<size_t>(v);
}
static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--help") {
            std::cout << "Usage: ./lesson12 [--n 65536] [--degree 32] [--pq 32] "
                         "[--train-exp 18000] [--test-exp 12000] [--reps 15] [--trials 3] "
                         "[--seed 42] [--train-file train.txt --test-file test.txt] "
                         "[--pq-bytes raw_headerless_codes.bin]\n"
                         "Trace line: parent_id neighbor_id_0 neighbor_id_1 ... (separate query sets).\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
        const char* v = argv[++i];
        if (a == "--n") o.n = read_integer(v, "--n");
        else if (a == "--degree") o.degree = read_integer(v, "--degree");
        else if (a == "--pq") o.chunks = read_integer(v, "--pq");
        else if (a == "--train-exp") o.train_exp = read_integer(v, "--train-exp");
        else if (a == "--test-exp") o.test_exp = read_integer(v, "--test-exp");
        else if (a == "--reps") o.reps = read_integer(v, "--reps");
        else if (a == "--trials") o.trials = read_integer(v, "--trials");
        else if (a == "--seed") o.seed = static_cast<uint32_t>(read_integer(v, "--seed"));
        else if (a == "--train-file") o.train_file = v;
        else if (a == "--test-file") o.test_file = v;
        else if (a == "--pq-bytes") o.pq_bytes_file = v;
        else throw std::runtime_error("Unknown option: " + a);
    }
    if (o.train_file.empty() != o.test_file.empty())
        throw std::runtime_error("Specify BOTH --train-file and --test-file");
    if (o.n > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("--n too large");
    if (o.degree >= o.n) throw std::runtime_error("--degree must be less than --n");
    if (o.chunks > 1024) throw std::runtime_error("--pq too large");
    return o;
}

// IDs have intentionally scrambled cluster membership, so locality isn't baked into IDs.
static std::vector<std::vector<uint32_t>> synthetic_graph(const Options& o) {
    std::mt19937 rng(o.seed);
    const size_t cluster_size = 64;
    std::vector<uint32_t> perm(o.n);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<std::vector<uint32_t>> graph(o.n);
    std::uniform_int_distribution<size_t> all(0, o.n - 1);
    for (size_t logical = 0; logical < o.n; ++logical) {
        auto& neighbors = graph[perm[logical]];
        neighbors.reserve(o.degree);
        const size_t lo = (logical / cluster_size) * cluster_size;
        const size_t hi = std::min(lo + cluster_size, o.n);
        std::uniform_int_distribution<size_t> local(lo, hi - 1);
        size_t attempts = 0;
        while (neighbors.size() < o.degree) {
            if (++attempts > o.degree * 10000)
                throw std::runtime_error("Could not generate enough distinct neighbors");
            bool in_cluster = (neighbors.size() % 8 != 7);
            uint32_t id = perm[in_cluster ? local(rng) : all(rng)];
            if (id == perm[logical]) continue;
            if (std::find(neighbors.begin(), neighbors.end(), id) != neighbors.end()) continue;
            neighbors.push_back(id);
        }
    }
    return graph;
}
static Trace sample_trace(const std::vector<std::vector<uint32_t>>& graph,
                          size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> pick(0, uint32_t(graph.size() - 1));
    Trace t;
    t.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        uint32_t p = pick(rng);
        t.push_back(Expansion{p, graph[p]});
    }
    return t;
}
static Trace load_trace(const std::string& path, size_t n) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open trace: " + path);
    Trace t;
    std::string line;
    size_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        for (char& ch : line) if (ch == ',') ch = ' ';
        size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream in(line);
        uint64_t parent;
        if (!(in >> parent) || parent >= n)
            throw std::runtime_error("Invalid parent ID in " + path + " line " + std::to_string(lineno));
        Expansion e;
        e.parent = uint32_t(parent);
        uint64_t x;
        while (in >> x) {
            if (x >= n) throw std::runtime_error("Invalid neighbor ID in " + path + " line " + std::to_string(lineno));
            e.neighbors.push_back(uint32_t(x));
        }
        if (e.neighbors.empty())
            throw std::runtime_error("Missing neighbors in " + path + " line " + std::to_string(lineno));
        t.push_back(std::move(e));
    }
    if (t.empty()) throw std::runtime_error("Empty trace: " + path);
    return t;
}
static void merge_graph(std::vector<std::vector<uint32_t>>& graph, const Trace& t) {
    for (const auto& e : t) {
        auto& g = graph[e.parent];
        if (g.empty()) g = e.neighbors;
        else if (g != e.neighbors)
            throw std::runtime_error("Inconsistent neighbor list for parent " + std::to_string(e.parent));
    }
}

// Keep all PQ payload arrays on 64-byte boundaries. std::vector<uint8_t>
// by itself does not guarantee 32/64-byte alignment. The underlying storage
// owns extra bytes, but .data() and .size() expose only the logical payload.
class AlignedBytes {
    std::vector<uint8_t> storage_;
    size_t count_ = 0;
public:
    void resize(size_t count) {
        count_ = count;
        if (count > std::numeric_limits<size_t>::max() - 63)
            throw std::runtime_error("AlignedBytes allocation overflow");
        storage_.resize(count + 63);
    }
    uint8_t* data() {
        auto p = reinterpret_cast<std::uintptr_t>(storage_.data());
        return reinterpret_cast<uint8_t*>((p + 63) & ~std::uintptr_t(63));
    }
    const uint8_t* data() const {
        auto p = reinterpret_cast<std::uintptr_t>(storage_.data());
        return reinterpret_cast<const uint8_t*>((p + 63) & ~std::uintptr_t(63));
    }
    size_t size() const { return count_; }
    uint8_t* begin() { return data(); }
    uint8_t* end() { return data() + count_; }
};

struct Layouts {
    AlignedBytes global, reorder, packed;
    std::vector<uint32_t> id_to_slot;
    std::vector<size_t> packed_offsets; // byte offsets for node's neighbor codes
    std::vector<float> lut;
    size_t n = 0, c = 0, total_edges = 0, observed_parents = 0;
};
static Layouts make_layouts(const Options& o,
                             const std::vector<std::vector<uint32_t>>& graph,
                             const Trace& training) {
    Layouts l;
    l.n = o.n; l.c = o.chunks;
    std::mt19937 rng(o.seed + 100);
    std::uniform_int_distribution<int> code_dist(0, 255);
    std::uniform_real_distribution<float> lut_dist(0.0f, 1.0f);
    l.global.resize(o.n * o.chunks);
    l.lut.resize(o.chunks * 256);
    if (o.pq_bytes_file.empty()) {
        for (auto& c : l.global) c = uint8_t(code_dist(rng));
    } else {
        std::ifstream f(o.pq_bytes_file, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("Cannot open PQ bytes: " + o.pq_bytes_file);
        const auto size = f.tellg();
        if (size < 0 || static_cast<uint64_t>(size) != l.global.size())
            throw std::runtime_error("PQ bytes file must be exactly N * PQ bytes (no header)");
        f.seekg(0);
        f.read(reinterpret_cast<char*>(l.global.data()),
               static_cast<std::streamsize>(l.global.size()));
        if (!f) throw std::runtime_error("Failed to read PQ bytes file");
    }
    for (auto& v : l.lut) v = lut_dist(rng);

    // First occurrence in TRAINING neighbor stream. Held-out TEST does not train ordering.
    std::vector<uint32_t> order;
    order.reserve(o.n);
    std::vector<uint8_t> seen(o.n, 0);
    for (const auto& e : training) {
        for (uint32_t id : e.neighbors) {
            if (!seen[id]) { seen[id] = 1; order.push_back(id); }
        }
    }
    for (uint32_t id = 0; id < o.n; ++id)
        if (!seen[id]) order.push_back(id);
    l.id_to_slot.resize(o.n);
    l.reorder.resize(l.global.size());
    for (size_t slot = 0; slot < order.size(); ++slot) {
        uint32_t id = order[slot];
        l.id_to_slot[id] = uint32_t(slot);
        std::memcpy(l.reorder.data() + slot * o.chunks,
                    l.global.data() + size_t(id) * o.chunks, o.chunks);
    }

    // Materialize per-parent copies of all neighbor PQ codes.
    l.packed_offsets.resize(o.n + 1, 0);
    for (size_t id = 0; id < o.n; ++id) {
        l.packed_offsets[id] = l.total_edges * o.chunks;
        l.total_edges += graph[id].size();
        if (!graph[id].empty()) ++l.observed_parents;
    }
    l.packed_offsets[o.n] = l.total_edges * o.chunks;
    l.packed.resize(l.total_edges * o.chunks);
    for (size_t id = 0; id < o.n; ++id) {
        uint8_t* dst = l.packed.data() + l.packed_offsets[id];
        for (uint32_t nbr : graph[id]) {
            std::memcpy(dst, l.global.data() + size_t(nbr) * o.chunks, o.chunks);
            dst += o.chunks;
        }
    }
    return l;
}
static void check_equivalence(const Layouts& l, const Trace& test) {
    for (const auto& e : test) {
        size_t packed_begin = l.packed_offsets[e.parent];
        for (size_t i = 0; i < e.neighbors.size(); ++i) {
            size_t id = e.neighbors[i];
            const uint8_t* expected = l.global.data() + id * l.c;
            const uint8_t* reordered = l.reorder.data() + size_t(l.id_to_slot[id]) * l.c;
            const uint8_t* packed = l.packed.data() + packed_begin + i * l.c;
            if (std::memcmp(expected, reordered, l.c) || std::memcmp(expected, packed, l.c))
                throw std::runtime_error("PQ code mismatch in layout validation");
        }
    }
}

enum class Mode { Global, Reordered, Packed };
static const char* mode_name(Mode m) {
    return m == Mode::Global ? "global_gather" :
           m == Mode::Reordered ? "reorder_gather" : "neighbor_packed";
}
static inline void lookup_chunk_major(const float* lut, const uint8_t* codes,
                                       size_t n, size_t c, float* dists) {
    std::fill(dists, dists + n, 0.0f);
    for (size_t chunk = 0; chunk < c; ++chunk) {
        const float* subtable = lut + 256 * chunk;
        for (size_t i = 0; i < n; ++i)
            dists[i] += subtable[codes[i * c + chunk]];
    }
}
static double replay(const Layouts& l, const Trace& t, Mode mode,
                     size_t reps, size_t max_degree) {
    std::vector<uint8_t> scratch(max_degree * l.c);
    std::vector<float> dists(max_degree);
    double checksum = 0;
    for (size_t rep = 0; rep < reps; ++rep) {
        for (const auto& e : t) {
            const size_t n = e.neighbors.size();
            const uint8_t* ptr = nullptr;
            if (mode == Mode::Packed) {
                ptr = l.packed.data() + l.packed_offsets[e.parent];
            } else {
                const uint8_t* data = mode == Mode::Global ? l.global.data() : l.reorder.data();
                for (size_t i = 0; i < n; ++i) {
                    uint32_t id = e.neighbors[i];
                    size_t slot = mode == Mode::Global ? id : l.id_to_slot[id];
                    std::memcpy(scratch.data() + i * l.c,
                                data + slot * l.c, l.c);
                }
                ptr = scratch.data();
            }
            lookup_chunk_major(l.lut.data(), ptr, n, l.c, dists.data());
            for (size_t i = 0; i < n; ++i) checksum += dists[i];
        }
    }
    return checksum;
}
static size_t count_lines(const uint8_t* base, size_t c,
                          const std::vector<uint32_t>& neighbors,
                          const std::vector<uint32_t>* map) {
    std::vector<uintptr_t> lines;
    lines.reserve(neighbors.size() * (c / 64 + 2));
    for (auto id : neighbors) {
        size_t slot = map ? (*map)[id] : id;
        uintptr_t begin = reinterpret_cast<uintptr_t>(base + slot * c);
        uintptr_t end = begin + c - 1;
        for (auto x = begin / 64; x <= end / 64; ++x) lines.push_back(x);
    }
    std::sort(lines.begin(), lines.end());
    lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
    return lines.size();
}
static size_t count_contiguous_lines(const uint8_t* start, size_t bytes) {
    if (!bytes) return 0;
    uintptr_t first = reinterpret_cast<uintptr_t>(start) / 64;
    uintptr_t last = (reinterpret_cast<uintptr_t>(start) + bytes - 1) / 64;
    return size_t(last - first + 1);
}
static void report_lines(const Layouts& l, const Trace& t) {
    double sum[3] = {0, 0, 0};
    double count = double(t.size());
    for (const auto& e : t) {
        sum[0] += count_lines(l.global.data(), l.c, e.neighbors, nullptr);
        sum[1] += count_lines(l.reorder.data(), l.c, e.neighbors, &l.id_to_slot);
        sum[2] += count_contiguous_lines(
            l.packed.data() + l.packed_offsets[e.parent], e.neighbors.size() * l.c);
    }
    std::cout << "PQ-code cache-line footprint (64B lines, test traces; NOT cache misses):\n";
    for (int i = 0; i < 3; ++i)
        std::cout << "  " << std::left << std::setw(17)
                  << mode_name(static_cast<Mode>(i)) << std::right
                  << std::fixed << std::setprecision(2) << (sum[i] / count)
                  << " unique lines/expansion\n";
    std::cout << "  NOTE: reordered id_to_slot mapping reads are NOT in footprint metric,"
                 " but ARE in timing.\n";
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double mib(size_t bytes) { return double(bytes) / (1024.0 * 1024); }

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        std::vector<std::vector<uint32_t>> graph;
        Trace train, test;
        bool real = !o.train_file.empty();
        if (real) {
            graph.resize(o.n);
            train = load_trace(o.train_file, o.n);
            test = load_trace(o.test_file, o.n);
            merge_graph(graph, train);
            merge_graph(graph, test);
        } else {
            graph = synthetic_graph(o);
            train = sample_trace(graph, o.train_exp, o.seed + 1);
            test = sample_trace(graph, o.test_exp, o.seed + 2);
        }
        size_t eval_neighbors = 0, max_degree = 0;
        for (const auto& e : test) {
            eval_neighbors += e.neighbors.size();
            max_degree = std::max(max_degree, e.neighbors.size());
        }
        if (eval_neighbors == 0) throw std::runtime_error("Test trace has no neighbors");
        const Layouts l = make_layouts(o, graph, train);
        std::cout << "PQ payload alignment (address mod 64): global="
                  << (reinterpret_cast<std::uintptr_t>(l.global.data()) % 64)
                  << " reorder=" << (reinterpret_cast<std::uintptr_t>(l.reorder.data()) % 64)
                  << " packed=" << (reinterpret_cast<std::uintptr_t>(l.packed.data()) % 64)
                  << "\n";
        check_equivalence(l, test);
        std::cout << "Source=" << (real ? "REAL trace replay" : "SYNTHETIC clustered graph")
                  << " N=" << o.n << " PQ=" << o.chunks << " train_exp=" << train.size()
                  << " test_exp=" << test.size() << " test_neighbors=" << eval_neighbors << "\n";
        std::cout << std::fixed << std::setprecision(2)
                  << "Memory payloads (MiB): global=" << mib(l.global.size())
                  << " reorder=" << mib(l.reorder.size())
                  << " +mapping=" << mib(l.id_to_slot.size() * sizeof(uint32_t))
                  << " packed=" << mib(l.packed.size())
                  << " +offsets=" << mib(l.packed_offsets.size() * sizeof(size_t))
                  << " common_LUT=" << mib(l.lut.size() * sizeof(float)) << "\n";
        std::cout << "Neighbor-packed edges=" << l.total_edges
                  << " parents_with_records=" << l.observed_parents << "/" << o.n;
        if (real && l.observed_parents < o.n)
            std::cout << " (PARTIAL observed-parent index: packed memory is a LOWER BOUND)";
        std::cout << "\n";
        report_lines(l, test);
        std::cout << "Validation: global/reordered/packed PQ bytes identical for ALL test neighbors.\n";
        const Mode modes[] = {Mode::Global, Mode::Reordered, Mode::Packed};
        double reference_checksum = replay(l, test, Mode::Global, 1, max_degree);
        for (Mode mode : modes) {
            double ch = replay(l, test, mode, 1, max_degree);
            if (std::abs(ch - reference_checksum) > 1e-6 * std::max(1.0, std::abs(reference_checksum)))
                throw std::runtime_error("checksum mismatch across modes");
        }
        std::cout << "Warmup checksums match; starting benchmark.\n";
        std::vector<double> measurements[3];
        for (size_t trial = 0; trial < o.trials; ++trial) {
            for (size_t j = 0; j < 3; ++j) {
                size_t mi = (j + trial) % 3;
                auto start = std::chrono::steady_clock::now();
                double checksum = replay(l, test, modes[mi], o.reps, max_degree);
                auto stop = std::chrono::steady_clock::now();
                // Print checksum to prevent compiler elimination of calculations.
                if (trial == 0) std::cout << "checksum[" << mode_name(modes[mi]) << "]="
                                          << std::setprecision(4) << checksum << "\n";
                double ns = std::chrono::duration<double, std::nano>(stop - start).count();
                measurements[mi].push_back(ns / (double(eval_neighbors) * o.reps));
            }
        }
        std::cout << "Results (median of " << o.trials << " trials, " << o.reps
                  << " replay repetitions/trial; lower is better):\n";
        for (size_t mi = 0; mi < 3; ++mi) {
            std::cout << "  " << std::left << std::setw(17) << mode_name(modes[mi]) << std::right
                      << std::fixed << std::setprecision(2)
                      << median(measurements[mi]) << " ns/neighbor"
                      << "    (" << median(measurements[mi]) * double(eval_neighbors) / test.size()
                      << " ns/expansion)\n";
        }
        std::cout << "IMPORTANT: replay measures PQ gather + lookup only. No real SSD I/O, "
                     "retset, visited, candidate selection, or ANN recall.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
