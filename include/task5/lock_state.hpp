#pragma once

#include "task5/types.hpp"

#include <chrono>
#include <string>

namespace task5 {

enum class VisualLockState {
    Qualifying,
    Stable,
    Lost,
};

struct VisualLockConfig {
    std::chrono::milliseconds stable_hold{5'000};
    int bad_windows_to_lose = 3;
    std::chrono::milliseconds minimum_lock_hold{5'000};
};

struct VisualLockUpdate {
    VisualLockState state = VisualLockState::Qualifying;
    bool stable_5s = false;
    bool lost = false;
    std::string event;
};

class VisualLockStateMachine {
public:
    explicit VisualLockStateMachine(VisualLockConfig config = {});
    void reset(Clock::time_point now);
    VisualLockUpdate update(bool shape_good, Clock::time_point now);
    VisualLockState state() const;

private:
    VisualLockConfig config_;
    VisualLockState state_ = VisualLockState::Qualifying;
    Clock::time_point good_since_{};
    Clock::time_point locked_since_{};
    int consecutive_bad_ = 0;
};

}  // namespace task5
