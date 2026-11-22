// Lesson 7: a tiny DiskANN-like search, not a copy of Microsoft DiskANN.
// Compile (portable synchronous reads):
//   g++ -std=c++17 -O2 -Wall -Wextra lesson7_mini_diskann.cpp -o lesson7
// Compile (Linux io_uring, as in Lesson 3):
//   g++ -std=c++17 -O2 -Wall -Wextra -DUSE_URING lesson7_mini_diskann.cpp -o lesson7 -luring
// Run: ./lesson7

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#ifdef USE_URING
#include <liburing.h>
#endif

constexpr int N = 8;
constexpr int D = 2;
constexpr int K = 4;               // Toy codebook: 4 centroids per dimension
constexpr int BLOCK = 4096;        // One padded record per 4-KiB block
constexpr int MAX_DEG = 4;
constexpr int BEAM = 2;             // Up to 2 node reads per round
constexpr int L = 6;                // At most 6 PQ-ranked candidates
constexpr float CENTERS[K] = {0.f, 2.f, 4.f, 6.f};

struct NodeRecord {                 // Stored on disk
    float vector[D];
    uint32_t degree;
    uint32_t neighbors[MAX_DEG];
};
static_assert(sizeof(NodeRecord) <= BLOCK, "record must fit in a block");

using Vec = std::array<float, D>;
const std::array<Vec, N> points = {{{0.f, 0.f}, {1.f, 0.5f},
    {2.f, 2.f}, {3.f, 3.f}, {4.f, 4.f}, {5.f, 4.4f},
    {6.f, 6.f}, {2.5f, 5.8f}}};
const std::array<std::vector<uint32_t>, N> adjacency = {{
    {1, 2}, {0, 3}, {0, 3, 7}, {1, 2, 4, 5},
    {3, 5, 7}, {3, 4, 6}, {5}, {2, 4}
}};

void checked_write(int fd, const char *data, size_t n) {
    while (n != 0) {
        ssize_t m = ::write(fd, data, n);
        if (m < 0 && errno == EINTR) continue;
        if (m <= 0) throw std::runtime_error("write failed");
        data += m;
        n -= static_cast<size_t>(m);
    }
}

void make_disk_index(const std::string &path) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw std::runtime_error("cannot create index file");
    for (int id = 0; id < N; ++id) {
        NodeRecord r{};
        for (int d = 0; d < D; ++d) r.vector[d] = points[id][d];
        r.degree = static_cast<uint32_t>(adjacency[id].size());
        for (size_t j = 0; j < adjacency[id].size(); ++j)
            r.neighbors[j] = adjacency[id][j];
        std::array<char, BLOCK> page{}; // zero-filled padding
        std::memcpy(page.data(), &r, sizeof(r));
        checked_write(fd, page.data(), page.size());
    }
    if (::fsync(fd) != 0) { ::close(fd); throw std::runtime_error("fsync failed"); }
    ::close(fd);
}

// PQ codes are in RAM, not on SSD. One byte per dimension (a toy PQ2).
using Code = std::array<uint8_t, D>;
std::array<Code, N> build_pq_codes() {
    std::array<Code, N> codes{};
    for (int id = 0; id < N; ++id) {
        for (int d = 0; d < D; ++d) {
            int best = 0;
            for (int k = 1; k < K; ++k)
                if (std::abs(points[id][d] - CENTERS[k]) <
                    std::abs(points[id][d] - CENTERS[best])) best = k;
            codes[id][d] = static_cast<uint8_t>(best);
        }
    }
    return codes;
}

using LUT = std::array<std::array<float, K>, D>;
LUT build_query_lut(const Vec &q) {
    LUT lut{};
    for (int d = 0; d < D; ++d)
        for (int k = 0; k < K; ++k) {
            float diff = q[d] - CENTERS[k];
            lut[d][k] = diff * diff;
        }
    return lut;
}
float pq_distance(const LUT &lut, const Code &c) {
    float sum = 0.f;
    for (int d = 0; d < D; ++d) sum += lut[d][c[d]];
    return sum;
}
float exact_distance(const Vec &q, const NodeRecord &node) {
    float sum = 0.f;
    for (int d = 0; d < D; ++d) {
        float diff = q[d] - node.vector[d];
        sum += diff * diff;
    }
    return sum;
}

struct Neighbor {
    uint32_t id;
    float distance;                // PQ squared L2, NOT exact distance
    bool expanded = false;
};

class Retset {
    std::vector<Neighbor> sorted_;
    size_t cur_ = 0;
    void reset_cur() {
        cur_ = 0;
        while (cur_ < sorted_.size() && sorted_[cur_].expanded) ++cur_;
    }
public:
    bool insert(uint32_t id, float dist) {
        for (const auto &x : sorted_)
            if (x.id == id) return false;
        Neighbor n{id, dist, false};
        auto less = [](const Neighbor &a, const Neighbor &b) {
            return a.distance < b.distance ||
                   (a.distance == b.distance && a.id < b.id);
        };
        auto pos = std::lower_bound(sorted_.begin(), sorted_.end(), n, less);
        if (sorted_.size() == L && pos == sorted_.end()) return false;
        sorted_.insert(pos, n);
        if (sorted_.size() > L) sorted_.pop_back();
        reset_cur();                 // Simpler than DiskANN's optimized _cur update
        return true;
    }
    bool take_next(uint32_t &id) {
        reset_cur();
        if (cur_ == sorted_.size()) return false;
        id = sorted_[cur_].id;
        sorted_[cur_].expanded = true;  // Means SELECTED, not I/O completed
        reset_cur();
        return true;
    }
    void print() const {
        std::printf("  retset:");
        for (const auto &n : sorted_)
            std::printf(" [%u PQ=%.2f %s]", n.id, n.distance,
                        n.expanded ? "expanded" : "pending");
        std::printf("\n");
    }
};

struct FinishedRead {
    uint32_t id;
    NodeRecord record;
};

// Read up to BEAM node records. Exactly one aligned 4-KiB SSD block per node.
std::vector<FinishedRead> read_beam(int fd, const std::vector<uint32_t> &ids
#ifdef USE_URING
    , io_uring &ring
#endif
) {
    alignas(BLOCK) std::array<std::array<char, BLOCK>, BEAM> buffers{};
    std::vector<FinishedRead> out;
    out.reserve(ids.size());
#ifdef USE_URING
    struct Request { uint32_t id; int slot; } requests[BEAM]{};
    for (size_t i = 0; i < ids.size(); ++i) {
        requests[i] = {ids[i], static_cast<int>(i)};
        io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (!sqe) throw std::runtime_error("io_uring SQ is full");
        io_uring_prep_read(sqe, fd, buffers[i].data(), BLOCK,
                           static_cast<off_t>(ids[i]) * BLOCK);
        io_uring_sqe_set_data(sqe, &requests[i]);
    }
    int sent = io_uring_submit(&ring);
    if (sent != static_cast<int>(ids.size()))
        throw std::runtime_error("io_uring_submit did not submit entire beam");
    for (size_t done = 0; done < ids.size(); ++done) {

        io_uring_cqe *cqe = nullptr;
        int err = io_uring_wait_cqe(&ring, &cqe);
        if (err < 0) throw std::runtime_error(std::string("wait_cqe: ") + std::strerror(-err));
        auto *req = static_cast<Request *>(io_uring_cqe_get_data(cqe));
        const int result = cqe->res;
        const uint32_t id = req->id;
        const int slot = req->slot;
        io_uring_cqe_seen(&ring, cqe);
        if (result != BLOCK)
            throw std::runtime_error("short/failed io_uring read (result=" + std::to_string(result) + ")");
        NodeRecord node{};
        std::memcpy(&node, buffers[slot].data(), sizeof(node));
        out.push_back({id, node});
    }
#else
    for (size_t i = 0; i < ids.size(); ++i) {
        ssize_t n = ::pread(fd, buffers[i].data(), BLOCK,
                            static_cast<off_t>(ids[i]) * BLOCK);
        if (n != BLOCK) throw std::runtime_error("pread failed/short read");
        NodeRecord node{};
        std::memcpy(&node, buffers[i].data(), sizeof(node));
        out.push_back({ids[i], node});
    }
#endif
    return out;
}

int main() {
    try {
        const std::string path = "mini_index.bin"; // NOT lesson1's index.bin
        make_disk_index(path);
        int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) { ::perror("open O_DIRECT"); return 1; }
#ifdef USE_URING
        io_uring ring{};
        int err = io_uring_queue_init(8, &ring, 0);
        if (err < 0) {
            std::fprintf(stderr, "io_uring_queue_init: %s\n", std::strerror(-err));
            ::close(fd); return 1;
        }
#endif
        Vec query{5.2f, 4.5f};
        auto codes = build_pq_codes();
        auto lut = build_query_lut(query);
        Retset retset;
        std::unordered_set<uint32_t> visited;
        std::vector<std::pair<uint32_t, float>> full_retset;
        visited.insert(0);
        retset.insert(0, pq_distance(lut, codes[0]));
        std::printf("Query=(%.1f, %.1f), entry=0, L=%d, beam=%d\n",
                    query[0], query[1], L, BEAM);
        std::printf("PQ code for node 5 = [%u,%u]; PQ estimate=%.2f\n",
                    codes[5][0], codes[5][1], pq_distance(lut, codes[5]));
        int round = 0;
        for (;;) {
            std::vector<uint32_t> frontier;
            for (int i = 0; i < BEAM; ++i) {
                uint32_t id = 0;
                if (!retset.take_next(id)) break;
                frontier.push_back(id);
            }
            if (frontier.empty()) break;
            std::printf("\nRound %d: read", ++round);
            for (uint32_t id : frontier) std::printf(" node%u@%u", id, id * BLOCK);
            std::printf("\n");
            auto records = read_beam(fd, frontier
#ifdef USE_URING
                , ring
#endif
            );
            for (const auto &read : records) {
                const NodeRecord &node = read.record;
                float exact = exact_distance(query, node);
                full_retset.push_back({read.id, exact});
                std::printf("  completed node%u: exact squared L2=%.2f, neighbors:", read.id, exact);
                for (uint32_t j = 0; j < node.degree; ++j) {
                    uint32_t nbr = node.neighbors[j];
                    if (nbr >= N) throw std::runtime_error("corrupt neighbor ID");
                    float approx = pq_distance(lut, codes[nbr]);
                    bool fresh = visited.insert(nbr).second;
                    bool kept = fresh && retset.insert(nbr, approx);
                    std::printf(" %u(PQ=%.2f,%s)", nbr, approx,
                                !fresh ? "visited" : (kept ? "insert" : "drop"));
                }
                std::printf("\n");
            }
            retset.print();
        }
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const auto &a, const auto &b) {return a.second < b.second;});
        std::printf("\nTop-2 among expanded nodes (exact squared L2):\n");
        for (size_t i = 0; i < std::min<size_t>(2, full_retset.size()); ++i)
            std::printf("  node%u = %.3f\n", full_retset[i].first, full_retset[i].second);
        std::printf("Visited IDs: %zu; SSD node reads: %zu\n",
                    visited.size(), full_retset.size());
#ifdef USE_URING
        io_uring_queue_exit(&ring);
#endif
        ::close(fd);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
