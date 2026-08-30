#pragma once

#include "task5/types.hpp"
#include "task5/uart.hpp"

#include <chrono>
#include <cstdint>
#include <optional>

namespace task5 {

struct CommandTicket {
    uint64_t generation = 0;
    Clock::time_point acknowledged_at{};
    Clock::time_point visible_after{};
    uint64_t first_frame_sequence = 0;
    FineDdsCommand command{};
    double requested_frequency_hz = 0.0;
    double realized_frequency_hz = 0.0;
};

class DdsCommandScheduler {
public:
    explicit DdsCommandScheduler(UartTransport* uart = nullptr);
    void attach(UartTransport* uart);
    bool set_clock_hz(double clock_hz);
    double clock_hz() const;

    static uint64_t frequency_to_ftw_q8(double frequency_hz,
                                        double clock_hz);
    static double ftw_q8_to_frequency(uint64_t ftw_q8, double clock_hz);

    std::optional<CommandTicket> issue_sine(
        double frequency_hz, int phase, int amplitude,
        std::chrono::milliseconds display_settle,
        uint64_t last_frame_sequence);
    bool set_locked(bool locked);
    bool frame_is_fresh(const CommandTicket& ticket, const Frame& frame) const;

private:
    UartTransport* uart_ = nullptr;
    double clock_hz_ = 0.0;
    uint64_t generation_ = 0;
};

}  // namespace task5
