#pragma once

#include "task5/types.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace task5 {

class UartTransport {
public:
    UartTransport() = default;
    ~UartTransport();
    bool open(const std::string& device, int baud = 115200);
    void close();
    bool is_open() const;
    bool send_dds(const DdsCommand& command);
    bool send_fine_dds(const FineDdsCommand& command);
    bool set_locked(bool locked);
    bool request(uint8_t command, FpgaResponse* response,
                 int timeout_ms = 300);
    bool read_response(FpgaResponse* response, int timeout_ms = 300,
                       int expected_command = -1);

    static std::array<uint8_t, 11> encode_set_dds(const DdsCommand& command);
    static std::array<uint8_t, 11> encode_set_fine_dds(
        const FineDdsCommand& command);
    static std::array<uint8_t, 11> encode_set_locked(bool locked);
    static std::array<uint8_t, 11> encode_request(uint8_t command);

private:
    bool write_packet(const std::array<uint8_t, 11>& packet);
    int fd_ = -1;
};

}  // namespace task5
