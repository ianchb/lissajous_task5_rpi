#pragma once

#include "task5/types.hpp"

#include <string>

namespace task5 {

enum class ControllerState {
    ModeEnter,
    GridReady,
    FrequencyCoarse,
    FrequencyCode,
    CandidateConfirm,
    ShapeAcquire,
    PhaseTrack,
    StableHold,
    Failed,
};

struct ControllerObservation {
    bool grid_ready = false;
    bool coarse_range_ready = false;
    bool frequency_code_ready = false;
    double frequency_margin = 0.0;
    bool shape_good = false;
    bool phase_good = false;
    bool source_changed = false;
};

struct ControllerConfig {
    std::chrono::milliseconds total_budget{10'000};
    std::chrono::milliseconds grid_budget{900};
    std::chrono::milliseconds coarse_budget{2'000};
    std::chrono::milliseconds code_budget{3'000};
    std::chrono::milliseconds shape_budget{2'500};
    std::chrono::milliseconds stable_hold{5'000};
    int candidate_confirm_frames = 3;
    int shape_confirm_frames = 4;
    int phase_confirm_frames = 8;
};

struct ControllerOutput {
    ControllerState state = ControllerState::ModeEnter;
    bool request_frequency_probe = false;
    bool allow_phase_tracking = false;
    bool stable = false;
    bool failed = false;
    std::string event;
};

class ControllerStateMachine {
public:
    explicit ControllerStateMachine(ControllerConfig config = {});
    void reset(Clock::time_point now);
    ControllerOutput update(const ControllerObservation& observation,
                            Clock::time_point now);
    ControllerState state() const;

private:
    void enter(ControllerState state, Clock::time_point now,
               const char* event);

    ControllerConfig config_;
    ControllerState state_ = ControllerState::ModeEnter;
    Clock::time_point started_{};
    Clock::time_point deadline_{};
    Clock::time_point stable_started_{};
    int consecutive_good_ = 0;
    std::string pending_event_;
};

}  // namespace task5
