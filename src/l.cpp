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
