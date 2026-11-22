
#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>

int main() {
    constexpr size_t BLOCK = 4096;

    // 1. Open using Direct I/O
    int fd = open("index.bin", O_RDONLY | O_DIRECT);

    if (fd < 0) {
        perror("open");
        return 1;
    }

    // 2. Allocate aligned RAM buffer
    void* buffer = nullptr;

    if (posix_memalign(&buffer, BLOCK, BLOCK) != 0) {
        fprintf(stderr, "Allocation failed\n");
        close(fd);
        return 1;
    }

    // 3. Read Node 2
    off_t offset = 2 * BLOCK;

    ssize_t n = pread(fd, buffer, BLOCK, offset);

    if (n < 0) {
        perror("pread");
    } else {
        printf("Read %zd bytes\n", n);
        printf("Content: %s\n", static_cast<char*>(buffer));
    }

    // 4. Release resources
    free(buffer);
    close(fd);

    return n == BLOCK ? 0 : 1;
}
