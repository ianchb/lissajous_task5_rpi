#include "task5/controller_state.hpp"

namespace task5 {

ControllerStateMachine::ControllerStateMachine(ControllerConfig config)
    : config_(config) {
    reset(Clock::now());
}

void ControllerStateMachine::reset(Clock::time_point now) {
    state_ = ControllerState::ModeEnter;
    started_ = now;
    deadline_ = now + config_.grid_budget;
    stable_started_ = {};
    consecutive_good_ = 0;
    pending_event_ = "MODE_ENTER";
}

void ControllerStateMachine::enter(ControllerState state,
                                   Clock::time_point now,
                                   const char* event) {
    state_ = state;
    consecutive_good_ = 0;
    pending_event_ = event;
    switch (state_) {
        case ControllerState::GridReady:
            deadline_ = now + config_.coarse_budget;
            break;
        case ControllerState::FrequencyCoarse:
            deadline_ = now + config_.coarse_budget;
            break;
        case ControllerState::FrequencyCode:
            deadline_ = now + config_.code_budget;
            break;
        case ControllerState::CandidateConfirm:
            deadline_ = now + config_.code_budget;
            break;
        case ControllerState::ShapeAcquire:
            deadline_ = now + config_.shape_budget;
            break;
        case ControllerState::PhaseTrack:
            deadline_ = now + config_.shape_budget;
            break;
        case ControllerState::StableHold:
            stable_started_ = now;
            deadline_ = now + config_.stable_hold;
            break;
        default:
            break;
    }
}

ControllerOutput ControllerStateMachine::update(
    const ControllerObservation& observation, Clock::time_point now) {
    ControllerOutput output;
    output.state = state_;
    output.event = pending_event_;
    pending_event_.clear();

    if (observation.source_changed && state_ != ControllerState::ModeEnter) {
        enter(ControllerState::FrequencyCoarse, now, "REACQUIRE_SOURCE_CHANGE");
        output.state = state_;
        output.event = "REACQUIRE_SOURCE_CHANGE";
        output.request_frequency_probe = true;
        return output;
    }
    if (now - started_ > config_.total_budget &&
        state_ != ControllerState::StableHold) {
        enter(ControllerState::Failed, now, "TIMEOUT_TOTAL");
        output.state = state_;
        output.failed = true;
        output.event = "TIMEOUT_TOTAL";
        return output;
    }
    if (state_ != ControllerState::ModeEnter && state_ != ControllerState::Failed &&
        state_ != ControllerState::StableHold && now > deadline_) {
        enter(ControllerState::Failed, now, "TIMEOUT_STAGE");
        output.state = state_;
        output.failed = true;
        output.event = "TIMEOUT_STAGE";
        return output;
    }

    switch (state_) {
        case ControllerState::ModeEnter:
            if (observation.grid_ready) enter(ControllerState::GridReady, now,
                                               "GRID_READY");
            break;
        case ControllerState::GridReady:
            enter(ControllerState::FrequencyCoarse, now, "FREQUENCY_COARSE");
            break;
        case ControllerState::FrequencyCoarse:
            output.request_frequency_probe = true;
            if (observation.coarse_range_ready)
                enter(ControllerState::FrequencyCode, now, "FREQUENCY_CODE");
            break;
        case ControllerState::FrequencyCode:
            output.request_frequency_probe = true;
            if (observation.frequency_code_ready &&
                observation.frequency_margin > 0.01) {
                ++consecutive_good_;
                if (consecutive_good_ >= config_.candidate_confirm_frames)
                    enter(ControllerState::CandidateConfirm, now,
                          "CANDIDATE_CONFIRM");
            } else {
                consecutive_good_ = 0;
            }
            break;
        case ControllerState::CandidateConfirm:
            if (observation.frequency_code_ready &&
                observation.frequency_margin > 0.01) {
                ++consecutive_good_;
                if (consecutive_good_ >= config_.candidate_confirm_frames)
                    enter(ControllerState::ShapeAcquire, now, "SHAPE_ACQUIRE");
            } else {
                enter(ControllerState::FrequencyCode, now, "CANDIDATE_REJECTED");
            }
            break;
        case ControllerState::ShapeAcquire:
            if (observation.shape_good) {
                ++consecutive_good_;
                if (consecutive_good_ >= config_.shape_confirm_frames)
                    enter(ControllerState::PhaseTrack, now, "PHASE_TRACK");
            } else {
                consecutive_good_ = 0;
            }
            break;
        case ControllerState::PhaseTrack:
            output.allow_phase_tracking = true;
            if (observation.phase_good) {
                ++consecutive_good_;
                if (consecutive_good_ >= config_.phase_confirm_frames)
                    enter(ControllerState::StableHold, now, "STABLE_START");
            } else {
                consecutive_good_ = 0;
            }
            break;
        case ControllerState::StableHold:
            output.allow_phase_tracking = true;
            if (!observation.shape_good || !observation.phase_good) {
                enter(ControllerState::PhaseTrack, now, "STABLE_RESET");
            } else if (now - stable_started_ >= config_.stable_hold) {
                output.stable = true;
                output.event = "STABLE_PASS";
            }
            break;
        case ControllerState::Failed:
            output.failed = true;
            break;
    }
    output.state = state_;
    if (state_ == ControllerState::Failed) output.failed = true;
    return output;
}

ControllerState ControllerStateMachine::state() const { return state_; }

}  // namespace task5
