#include "task5/uart.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <vector>

namespace task5 {

UartTransport::~UartTransport() { close(); }

bool UartTransport::open(const std::string& device, int baud) {
    close();
    fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) return false;
    termios options{};
    if (tcgetattr(fd_, &options) != 0) {
        close();
        return false;
    }
    cfmakeraw(&options);
    speed_t speed = baud == 115200 ? B115200 : B115200;
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);
    options.c_cflag |= CLOCAL | CREAD;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CRTSCTS;
    if (tcsetattr(fd_, TCSANOW, &options) != 0) {
        close();
        return false;
    }
    tcflush(fd_, TCIOFLUSH);
    return true;
}

void UartTransport::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool UartTransport::is_open() const { return fd_ >= 0; }

std::array<uint8_t, 11> UartTransport::encode_set_dds(
    const DdsCommand& command) {
    std::array<uint8_t, 11> packet{};
    packet[0] = 0xA5;
    packet[1] = 0x5A;
    packet[2] = 0x01;
    packet[3] = static_cast<uint8_t>(command.ftw & 0xff);
    packet[4] = static_cast<uint8_t>((command.ftw >> 8) & 0xff);
    packet[5] = static_cast<uint8_t>((command.ftw >> 16) & 0xff);
    packet[6] = static_cast<uint8_t>((command.ftw >> 24) & 0xff);
    packet[7] = command.phase;
    packet[8] = command.amplitude;
    packet[9] = command.flags;
    uint8_t checksum = 0;
    for (int i = 0; i < 10; ++i) checksum ^= packet[static_cast<size_t>(i)];
    packet[10] = checksum;
    return packet;
}

bool UartTransport::send_dds(const DdsCommand& command) {
    return write_packet(encode_set_dds(command));
}

std::array<uint8_t, 11> UartTransport::encode_set_fine_dds(
    const FineDdsCommand& command) {
    std::array<uint8_t, 11> packet{};
    packet[0] = 0xA5;
    packet[1] = 0x5A;
    packet[2] = 0x05;
    const uint64_t ftw_q8 = command.ftw_q8 & 0xffffffffffULL;
    for (int index = 0; index < 5; ++index) {
        packet[static_cast<size_t>(3 + index)] = static_cast<uint8_t>(
            (ftw_q8 >> (8 * index)) & 0xffu);
    }
    packet[8] = command.phase;
    packet[9] = command.amplitude;
    uint8_t checksum = 0;
    for (int i = 0; i < 10; ++i) checksum ^= packet[static_cast<size_t>(i)];
    packet[10] = checksum;
    return packet;
}

bool UartTransport::send_fine_dds(const FineDdsCommand& command) {
    return write_packet(encode_set_fine_dds(command));
}

std::array<uint8_t, 11> UartTransport::encode_set_locked(bool locked) {
    std::array<uint8_t, 11> packet{};
    packet[0] = 0xA5;
    packet[1] = 0x5A;
    packet[2] = 0x06;
    packet[3] = locked ? 1 : 0;
    uint8_t checksum = 0;
    for (int i = 0; i < 10; ++i) checksum ^= packet[static_cast<size_t>(i)];
    packet[10] = checksum;
    return packet;
}

bool UartTransport::set_locked(bool locked) {
    return write_packet(encode_set_locked(locked));
}

std::array<uint8_t, 11> UartTransport::encode_request(uint8_t command) {
    std::array<uint8_t, 11> packet{};
    packet[0] = 0xA5;
    packet[1] = 0x5A;
    packet[2] = command;
    uint8_t checksum = 0;
    for (int i = 0; i < 10; ++i) checksum ^= packet[static_cast<size_t>(i)];
    packet[10] = checksum;
    return packet;
}

bool UartTransport::write_packet(const std::array<uint8_t, 11>& packet) {
    if (fd_ < 0) return false;
    size_t offset = 0;
    while (offset < packet.size()) {
        const ssize_t written = ::write(fd_, packet.data() + offset,
                                        packet.size() - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
        } else if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            usleep(1000);
        } else {
            return false;
        }
    }
    return true;
}

bool UartTransport::request(uint8_t command, FpgaResponse* response,
                            int timeout_ms) {
    if (!write_packet(encode_request(command))) return false;
    return read_response(response, timeout_ms, command);
}

bool UartTransport::read_response(FpgaResponse* response, int timeout_ms,
                                  int expected_command) {
    if (fd_ < 0 || !response || timeout_ms <= 0) return false;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    std::vector<uint8_t> buffer;
    buffer.reserve(64);
    while (std::chrono::steady_clock::now() < deadline) {
        uint8_t chunk[64];
        const ssize_t count = ::read(fd_, chunk, sizeof(chunk));
        if (count > 0) {
            buffer.insert(buffer.end(), chunk, chunk + count);
        } else if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            return false;
        }
        while (buffer.size() >= 12) {
            size_t start = 0;
            while (start + 1 < buffer.size() &&
                   !(buffer[start] == 0x5A && buffer[start + 1] == 0xA5)) {
                ++start;
            }
            if (start > 0) buffer.erase(buffer.begin(), buffer.begin() + start);
            if (buffer.size() < 12) break;
            uint8_t checksum = 0;
            for (int i = 0; i < 11; ++i) checksum ^= buffer[static_cast<size_t>(i)];
            const uint8_t command = buffer[10];
            if (checksum == buffer[11] &&
                (expected_command < 0 || command == expected_command)) {
                response->mode = buffer[2];
                response->flags = buffer[3];
                response->payload = static_cast<uint32_t>(buffer[4]) |
                    (static_cast<uint32_t>(buffer[5]) << 8) |
                    (static_cast<uint32_t>(buffer[6]) << 16) |
                    (static_cast<uint32_t>(buffer[7]) << 24);
                response->phase = buffer[8];
                response->amplitude = buffer[9];
                response->command = command;
                response->checksum_valid = true;
                return true;
            }
            buffer.erase(buffer.begin(), buffer.begin() + 12);
        }
        usleep(1000);
    }
    return false;
}

}  // namespace task5
