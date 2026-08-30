#include "task5/lock_state.hpp"

#include <algorithm>

namespace task5 {

VisualLockStateMachine::VisualLockStateMachine(VisualLockConfig config)
    : config_(config) {
    config_.bad_windows_to_lose = std::max(1, config_.bad_windows_to_lose);
    config_.stable_hold = std::max(
        std::chrono::milliseconds::zero(), config_.stable_hold);
    config_.minimum_lock_hold = std::max(
        std::chrono::milliseconds::zero(), config_.minimum_lock_hold);
}

void VisualLockStateMachine::reset(Clock::time_point now) {
    state_ = VisualLockState::Qualifying;
    good_since_ = {};
    locked_since_ = now;
    consecutive_bad_ = 0;
}

VisualLockUpdate VisualLockStateMachine::update(bool shape_good,
                                                 Clock::time_point now) {
    VisualLockUpdate result;
    result.state = state_;
    if (state_ == VisualLockState::Lost) {
        result.lost = true;
        return result;
    }

    if (!shape_good) {
        good_since_ = {};
        if (state_ == VisualLockState::Stable) {
            state_ = VisualLockState::Qualifying;
            result.event = "STABLE_RESET";
        }
        if (now - locked_since_ < config_.minimum_lock_hold) {
            consecutive_bad_ = 0;
            result.state = state_;
            return result;
        }
        ++consecutive_bad_;
        if (consecutive_bad_ >= config_.bad_windows_to_lose) {
            state_ = VisualLockState::Lost;
            result.event = "LOCK_LOST";
            result.lost = true;
        }
        result.state = state_;
        return result;
    }

    consecutive_bad_ = 0;
    if (good_since_ == Clock::time_point{}) {
        good_since_ = now;
        result.event = "STABLE_START";
    }
    if (state_ == VisualLockState::Qualifying &&
        now - good_since_ >= config_.stable_hold) {
        state_ = VisualLockState::Stable;
        result.event = "STABLE_5S";
        result.stable_5s = true;
    }
    result.state = state_;
    return result;
}

VisualLockState VisualLockStateMachine::state() const { return state_; }

}  // namespace task5
