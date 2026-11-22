
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

constexpr int BLOCK = 4096;
constexpr int NUM_READS = 4;

struct Request {
    int node_id;
};

int main() {
    // 1. Open the SSD index
    int fd = open("index.bin", O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // 2. Initialize io_uring
    io_uring ring;
    int ret = io_uring_queue_init(8, &ring, 0);
    if (ret < 0) {
        printf("queue_init: %s\n", strerror(-ret));
        close(fd);
        return 1;
    }

    // 3. Prepare aligned buffers
    alignas(4096) char buffers[NUM_READS][BLOCK] = {};
    Request requests[NUM_READS] = {};

    // 4. Prepare four read requests
    for (int i = 0; i < NUM_READS; i++) {
        requests[i].node_id = i;

        io_uring_sqe* sqe = io_uring_get_sqe(&ring);

        if (!sqe) {
            fprintf(stderr, "No free SQE\n");
            return 1;
        }

        io_uring_prep_read(
            sqe,
            fd,
            buffers[i],
            BLOCK,
            static_cast<off_t>(i) * BLOCK
        );

        io_uring_sqe_set_data(sqe, &requests[i]);
    }

    // 5. Submit all four reads
    int submitted = io_uring_submit(&ring);
    printf("Submitted: %d requests\n", submitted);

    if (submitted != NUM_READS) {
        fprintf(stderr, "Submission failed\n");
        return 1;
    }

    // 6. Collect four completions
    for (int i = 0; i < NUM_READS; i++) {
        io_uring_cqe* cqe = nullptr;

        ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) {
            fprintf(stderr, "wait: %s\n", strerror(-ret));
            return 1;
        }

        Request* req = static_cast<Request*>(
            io_uring_cqe_get_data(cqe)
        );

        int node = req->node_id;

        if (cqe->res == BLOCK) {
            printf("Node %d completed: %s\n",
                   node, buffers[node]);
        } else {
            printf("Node %d read result: %d\n",
                   node, cqe->res);
        }

        io_uring_cqe_seen(&ring, cqe);
    }

    // 7. Release resources
    io_uring_queue_exit(&ring);
    close(fd);
    return 0;
}
