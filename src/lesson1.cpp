
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

int main() {
    constexpr size_t BLOCK = 4096;

    // 1. 创建一个测试文件
    int fd = open("index.bin", O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // 2. 写入四个 node records
    for (int i = 0; i < 4; ++i) {
        char block[BLOCK] = {};
        snprintf(block, sizeof(block), "This is node %d", i);

        if (write(fd, block, BLOCK) != BLOCK) {
            perror("write");
            close(fd);
            return 1;
        }
    }

    // 3. 随机读取 Node 2
    char buffer[BLOCK] = {};
    off_t offset = 2 * BLOCK;

    ssize_t n = pread(fd, buffer, BLOCK, offset);

    if (n != BLOCK) {
        perror("pread");
        close(fd);
        return 1;
    }

    // 4. 查看读取到的数据
    printf("Read %zd bytes\n", n);
    printf("Content: %s\n", buffer);

    close(fd);
    return 0;
}
