#include "task5/command_scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace task5 {

DdsCommandScheduler::DdsCommandScheduler(UartTransport* uart) : uart_(uart) {}

void DdsCommandScheduler::attach(UartTransport* uart) { uart_ = uart; }

bool DdsCommandScheduler::set_clock_hz(double clock_hz) {
    if (!std::isfinite(clock_hz) || clock_hz <= 0.0) return false;
    clock_hz_ = clock_hz;
    return true;
}

double DdsCommandScheduler::clock_hz() const { return clock_hz_; }

uint64_t DdsCommandScheduler::frequency_to_ftw_q8(double frequency_hz,
                                                   double clock_hz) {
    if (!std::isfinite(frequency_hz) || !std::isfinite(clock_hz) ||
        frequency_hz < 0.0 || clock_hz <= 0.0)
        return 0;
    constexpr long double kScale = 1099511627776.0L;  // 2^40
    const long double raw = static_cast<long double>(frequency_hz) * kScale /
                            static_cast<long double>(clock_hz);
    return static_cast<uint64_t>(std::llround(raw)) & 0xffffffffffULL;
}

double DdsCommandScheduler::ftw_q8_to_frequency(uint64_t ftw_q8,
                                                 double clock_hz) {
    constexpr long double kScale = 1099511627776.0L;
    return static_cast<double>(
        static_cast<long double>(ftw_q8 & 0xffffffffffULL) *
        static_cast<long double>(clock_hz) / kScale);
}

std::optional<CommandTicket> DdsCommandScheduler::issue_sine(
    double frequency_hz, int phase, int amplitude,
    std::chrono::milliseconds display_settle, uint64_t last_frame_sequence) {
    if (!uart_ || clock_hz_ <= 0.0 || display_settle.count() < 0)
        return std::nullopt;
    FineDdsCommand command;
    command.ftw_q8 = frequency_to_ftw_q8(frequency_hz, clock_hz_);
    command.phase = static_cast<uint8_t>(phase & 0xff);
    command.amplitude = static_cast<uint8_t>(std::clamp(amplitude, 0, 127));
    if (!uart_->send_fine_dds(command)) return std::nullopt;
    FpgaResponse acknowledgement;
    if (!uart_->read_response(&acknowledgement, 400, 0x05))
        return std::nullopt;

    CommandTicket ticket;
    ticket.generation = ++generation_;
    ticket.acknowledged_at = Clock::now();
    ticket.visible_after = ticket.acknowledged_at + display_settle;
    ticket.first_frame_sequence = last_frame_sequence + 1;
    ticket.command = command;
    ticket.requested_frequency_hz = frequency_hz;
    ticket.realized_frequency_hz = ftw_q8_to_frequency(
        command.ftw_q8, clock_hz_);
    return ticket;
}

bool DdsCommandScheduler::set_locked(bool locked) {
    if (!uart_) return false;
    if (!uart_->set_locked(locked)) return false;
    FpgaResponse acknowledgement;
    if (!uart_->read_response(&acknowledgement, 400, 0x06)) return false;
    return true;
}

bool DdsCommandScheduler::frame_is_fresh(const CommandTicket& ticket,
                                         const Frame& frame) const {
    return ticket.generation != 0 &&
           frame.sequence >= ticket.first_frame_sequence &&
           frame.timestamp >= ticket.visible_after;
}

}  // namespace task5
