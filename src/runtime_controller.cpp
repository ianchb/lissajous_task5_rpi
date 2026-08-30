#include "task5/runtime_controller.hpp"

#include "task5/phase_code_matcher.hpp"
#include "task5/visual_frequency_servo.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>

namespace task5 {
namespace {

constexpr double kTwoPi = 6.28318530717958647692;
constexpr int kStartupGridSearchMs = 1'500;
constexpr int kExposureGridSearchMs = 900;
constexpr int kExposureSettleFrames = 6;
constexpr int kGridSearchSliceMs = 250;
constexpr int kGridAnalysisIntervalMs = 100;
constexpr int kModePollIntervalMs = 500;

uint32_t ftw_for_frequency(double frequency_hz, double clock_hz) {
    return static_cast<uint32_t>(std::llround(
        frequency_hz * 4294967296.0 / clock_hz));
}

bool is_auto_mode(uint8_t mode) { return mode >= 3 && mode <= 5; }

const char* mode_name(uint8_t mode) {
    switch (mode) {
        case 0: return "TASK1";
        case 1: return "TASK2";
        case 2: return "TASK3";
        case 3: return "AUTO_LINE";
        case 4: return "AUTO_CIRCLE";
        case 5: return "AUTO_FIGURE8";
        default: return "UNKNOWN";
    }
}

double startup_frequency_prior(AutoMode mode) {
    switch (mode) {
    case AutoMode::Line: return 1'100.0;
    case AutoMode::Circle: return 90'900.0;
    case AutoMode::FigureEight: return 49'900.0;
    }
    return 0.0;
}

constexpr bool use_startup_frequency_prior(int attempt) {
    return attempt >= 2 && attempt <= 5;
}

static_assert(!use_startup_frequency_prior(1));
static_assert(use_startup_frequency_prior(2));
static_assert(use_startup_frequency_prior(5));
static_assert(!use_startup_frequency_prior(6));

template <typename Accessor>
double median_value(const std::vector<ShapeMetrics>& values,
                    Accessor accessor) {
    std::vector<double> samples;
    samples.reserve(values.size());
    for (const ShapeMetrics& value : values)
        samples.push_back(accessor(value));
    auto middle = samples.begin() + samples.size() / 2;
    std::nth_element(samples.begin(), middle, samples.end());
    return *middle;
}

}  // namespace

RuntimeController::RuntimeController(RuntimeConfig config)
    : config_(std::move(config)) {}

std::optional<RuntimeController::ShapeChoice>
RuntimeController::refine_phase_locally(
    AutoMode mode, int starting_phase, const ShapeMetrics& starting_metrics,
    int maximum_rounds, bool require_two_fresh_windows,
    bool* filled_rectangle, int maximum_phase_step,
    int candidate_settle_ms, int candidate_measure_ms) {
    if (filled_rectangle) *filled_rectangle = false;
    int center_phase = starting_phase & 0xff;
    ShapeMetrics center_metrics = starting_metrics;
    double center_score = shape_observer_.search_score(mode, center_metrics);

    const auto initial_step = [&]() {
        if (mode == AutoMode::Line && center_metrics.valid) {
            const double error = std::acos(std::clamp(
                -center_metrics.correlation, -1.0, 1.0));
            return std::clamp(static_cast<int>(std::lround(
                error * 256.0 / kTwoPi)), 2, maximum_phase_step);
        }
        if (mode == AutoMode::Circle && center_metrics.valid) {
            const double error = std::asin(std::clamp(
                std::abs(center_metrics.correlation), 0.0, 1.0));
            return std::clamp(static_cast<int>(std::lround(
                error * 256.0 / kTwoPi)), 2, maximum_phase_step);
        }
        return std::max(2, maximum_phase_step);
    };
    int step = initial_step();

    const auto window_good = [&](const ShapeMetrics& aggregate,
                                 const ShapeWindowStats& stats) {
        if (shape_observer_.shape_ok(mode, aggregate)) return true;
        if (!stats.has_best_good) return false;
        if (mode == AutoMode::Circle)
            return stats.good_frames >= 3 &&
                   stats.good_frames * 3 >= std::max(1, stats.valid_frames);
        if (require_two_fresh_windows) return true;
        return stats.good_frames >= 2 || stats.valid_frames <= 2;
    };
    const auto confirm_at_phase = [&](int candidate_phase,
                                      const ShapeMetrics& candidate,
                                      bool candidate_good)
        -> std::optional<ShapeChoice> {
        ShapeWindowStats confirmation_stats;
        const auto confirmation = measure_shape(
            mode, candidate_phase, 350, 350, true, nullptr, {},
            &confirmation_stats, true);
        if (!confirmation) return std::nullopt;
        if (shape_observer_.frequency_mismatch_fill(*confirmation)) {
            if (filled_rectangle) *filled_rectangle = true;
            return std::nullopt;
        }
        const bool confirmation_good = window_good(
            *confirmation, confirmation_stats);
        std::cout << "PHASE_LOCAL_CONFIRM phase=" << candidate_phase
                  << " first_good=" << candidate_good
                  << " second_good=" << confirmation_good
                  << " first_score=" << std::fixed << std::setprecision(4)
                  << shape_observer_.search_score(mode, candidate)
                  << " second_score="
                  << shape_observer_.search_score(mode, *confirmation)
                  << "\n" << std::flush;
        if (confirmation_good &&
            (!require_two_fresh_windows || candidate_good))
            return ShapeChoice{candidate_phase, *confirmation, true};
        return std::nullopt;
    };

    for (int round = 0; round < maximum_rounds && expected_mode(mode);
         ++round) {
        struct Candidate {
            int phase = 0;
            ShapeMetrics metrics;
            ShapeWindowStats stats;
            double score = -1.0e9;
            bool good = false;
        };
        std::optional<Candidate> best;
        const std::array<int, 3> phases{
            center_phase,
            (center_phase + step) & 0xff,
            (center_phase - step) & 0xff};
        for (int candidate_phase : phases) {
            ShapeWindowStats stats;
            const auto metrics = measure_shape(
                mode, candidate_phase, candidate_settle_ms,
                candidate_measure_ms > 0 ? candidate_measure_ms :
                                           config_.shape_window_ms, true,
                nullptr, {}, &stats, true);
            if (!metrics || !expected_mode(mode)) return std::nullopt;
            if (shape_observer_.frequency_mismatch_fill(*metrics)) {
                if (filled_rectangle) *filled_rectangle = true;
                return std::nullopt;
            }
            Candidate candidate;
            candidate.phase = candidate_phase;
            candidate.metrics = *metrics;
            candidate.stats = stats;
            candidate.score = shape_observer_.search_score(mode, *metrics);
            candidate.good = window_good(*metrics, stats);
            std::cout << "PHASE_LOCAL_CANDIDATE round=" << round + 1
                      << " phase=" << candidate_phase
                      << " step=" << step
                      << " score=" << std::fixed << std::setprecision(4)
                      << candidate.score << " good=" << candidate.good
                      << " good_hits=" << stats.good_frames << "\n"
                      << std::flush;
            // Lock maintenance favors control bandwidth. The candidate has
            // already been observed in several fresh camera frames, so leave
            // the DDS there immediately instead of spending another window
            // confirming while the source phase continues to move.
            if (candidate.good && !require_two_fresh_windows)
                return ShapeChoice{candidate.phase, candidate.metrics, true};
            if (!best || candidate.score > best->score) best = candidate;
            // Do not perturb an already correct display. Confirm the centre
            // first; only explore either side when that fresh confirmation
            // fails.
            if (candidate_phase == center_phase && candidate.good) {
                if (const auto confirmed = confirm_at_phase(
                        candidate.phase, candidate.metrics, true))
                    return confirmed;
                if (filled_rectangle && *filled_rectangle)
                    return std::nullopt;
            }
        }
        if (!best) return std::nullopt;
        const bool improved = best->score > center_score + 0.02;
        if (improved || best->good) {
            center_phase = best->phase;
            center_metrics = best->metrics;
            center_score = best->score;
        } else {
            step = std::max(2, step / 2);
        }
        if (best->good) {
            if (const auto confirmed = confirm_at_phase(
                    best->phase, best->metrics, true))
                return confirmed;
            if (filled_rectangle && *filled_rectangle)
                return std::nullopt;
        }
        if (improved && step > 3) step = std::max(2, step / 2);
    }
    // Leave the DDS at the best point found so the next control iteration
    // continues from here instead of jumping back to the original phase.
    if (!send_output(active_output_hz_, center_phase, 102, 250))
        return std::nullopt;
    return ShapeChoice{center_phase, center_metrics, false};
}

std::optional<RuntimeController::PhaseServoChoice>
RuntimeController::probe_phase_abba(
    AutoMode mode, int center_phase, const ShapeMetrics& center_metrics,
    bool* filled_rectangle) {
    if (filled_rectangle) *filled_rectangle = false;
    constexpr int kProbeStep = 3;
    constexpr int kSettleMs = 80;
    constexpr int kMeasureMs = 120;
    constexpr double kMinimumScoreGap = 0.08;
    const std::array<int, 4> offsets{
        -kProbeStep, kProbeStep, kProbeStep, -kProbeStep};

    struct SideEvidence {
        std::vector<ShapeMetrics> metrics;
        double score_sum = 0.0;
        int score_count = 0;
        int good_windows = 0;
    };
    SideEvidence minus;
    SideEvidence plus;
    for (size_t index = 0; index < offsets.size(); ++index) {
        const int candidate_phase =
            (center_phase + offsets[index]) & 0xff;
        ShapeWindowStats stats;
        const auto metrics = measure_shape(
            mode, candidate_phase, kSettleMs, kMeasureMs, true,
            nullptr, {}, &stats, true);
        if (!metrics || !expected_mode(mode)) return std::nullopt;
        if (shape_observer_.frequency_mismatch_fill(*metrics)) {
            if (filled_rectangle) *filled_rectangle = true;
            return std::nullopt;
        }
        SideEvidence& side = offsets[index] < 0 ? minus : plus;
        side.metrics.push_back(*metrics);
        side.score_sum += shape_observer_.search_score(mode, *metrics);
        ++side.score_count;
        if (shape_observer_.shape_ok(mode, *metrics) ||
            (stats.has_best_good && stats.good_frames >= 2))
            ++side.good_windows;
    }
    if (minus.score_count != 2 || plus.score_count != 2)
        return std::nullopt;

    const double minus_score = minus.score_sum / minus.score_count;
    const double plus_score = plus.score_sum / plus.score_count;
    const double score_gap = plus_score - minus_score;
    const bool direction_confident =
        std::abs(score_gap) >= kMinimumScoreGap;
    const int chosen_offset = !direction_confident ? 0 :
                              (score_gap > 0.0 ? kProbeStep : -kProbeStep);
    const int chosen_phase = (center_phase + chosen_offset) & 0xff;
    const SideEvidence* chosen_side = chosen_offset > 0 ? &plus :
                                      chosen_offset < 0 ? &minus : nullptr;
    ShapeMetrics chosen_metrics = center_metrics;
    bool target_good = shape_observer_.shape_ok(mode, center_metrics);
    if (chosen_side) {
        chosen_metrics = median_metrics(chosen_side->metrics);
        target_good = shape_observer_.shape_ok(mode, chosen_metrics) ||
                      chosen_side->good_windows >= 1;
    }
    if (!send_output(active_output_hz_, chosen_phase, 102, 0))
        return std::nullopt;
    std::cout << "LOCK_PHASE_ABBA center=" << (center_phase & 0xff)
              << " minus_score=" << std::fixed << std::setprecision(4)
              << minus_score << " plus_score=" << plus_score
              << " score_gap=" << score_gap
              << " chosen=" << chosen_phase
              << " direction_confident=" << direction_confident
              << " target_good=" << target_good << "\n" << std::flush;
    return PhaseServoChoice{chosen_phase, chosen_metrics, target_good,
                            direction_confident};
}

std::optional<RuntimeController::PhaseServoChoice>
RuntimeController::probe_circle_centered(
    int center_phase, const ShapeMetrics& center_metrics,
    bool* filled_rectangle) {
    if (filled_rectangle) *filled_rectangle = false;
    constexpr int kProbeStep = 3;
    constexpr int kSettleMs = 70;
    constexpr int kMeasureMs = 120;
    constexpr double kMinimumScoreGain = 0.18;
    constexpr double kMinimumDirectionMargin = 0.06;
    const std::array<int, 5> offsets{
        0, -kProbeStep, 0, kProbeStep, 0};

    struct ProbeWindow {
        ShapeMetrics metrics;
        double score = -1.0e9;
        bool good = false;
    };
    std::array<ProbeWindow, 5> windows;
    for (size_t index = 0; index < offsets.size(); ++index) {
        const int candidate_phase =
            (center_phase + offsets[index]) & 0xff;
        ShapeWindowStats stats;
        const auto metrics = measure_shape(
            AutoMode::Circle, candidate_phase, kSettleMs, kMeasureMs, true,
            nullptr, {}, &stats, true);
        if (!metrics || !expected_mode(AutoMode::Circle))
            return std::nullopt;
        if (shape_observer_.frequency_mismatch_fill(*metrics)) {
            if (filled_rectangle) *filled_rectangle = true;
            return std::nullopt;
        }
        windows[index].metrics = *metrics;
        windows[index].score = shape_observer_.search_score(
            AutoMode::Circle, *metrics);
        windows[index].good = shape_observer_.shape_ok(
            AutoMode::Circle, *metrics) ||
            (stats.has_best_good && stats.good_frames >= 3 &&
             stats.good_frames * 3 >= std::max(1, stats.valid_frames));
    }

    // Each side is compared with the fresh centre windows immediately before
    // and after it. This removes the monotonic score drift that made the old
    // side-only ABBA walk through a circle and end at either diagonal line.
    const double minus_baseline =
        0.5 * (windows[0].score + windows[2].score);
    const double plus_baseline =
        0.5 * (windows[2].score + windows[4].score);
    const double minus_gain = windows[1].score - minus_baseline;
    const double plus_gain = windows[3].score - plus_baseline;

    int chosen_index = 4;
    int chosen_offset = 0;
    bool direction_confident = false;
    const std::array<int, 3> center_indices{0, 2, 4};
    for (int index : center_indices) {
        if (windows[index].good &&
            (!windows[chosen_index].good ||
             windows[index].score > windows[chosen_index].score))
            chosen_index = index;
    }
    if (!windows[chosen_index].good) {
        const bool minus_acceptable = windows[1].good ||
            (minus_gain >= kMinimumScoreGain &&
             minus_gain >= plus_gain + kMinimumDirectionMargin);
        const bool plus_acceptable = windows[3].good ||
            (plus_gain >= kMinimumScoreGain &&
             plus_gain >= minus_gain + kMinimumDirectionMargin);
        if (minus_acceptable &&
            (!plus_acceptable || minus_gain >= plus_gain)) {
            chosen_index = 1;
            chosen_offset = -kProbeStep;
            direction_confident = true;
        } else if (plus_acceptable) {
            chosen_index = 3;
            chosen_offset = kProbeStep;
            direction_confident = true;
        }
    }

    const int chosen_phase = (center_phase + chosen_offset) & 0xff;
    ShapeMetrics chosen_metrics = windows[chosen_index].metrics;
    if (!chosen_metrics.valid) chosen_metrics = center_metrics;
    const bool target_good = windows[chosen_index].good;
    if (!send_output(active_output_hz_, chosen_phase, 102, 0))
        return std::nullopt;
    std::cout << "LOCK_CIRCLE_CENTERED center=" << (center_phase & 0xff)
              << " center_score=" << std::fixed << std::setprecision(4)
              << windows[4].score
              << " minus_score=" << windows[1].score
              << " plus_score=" << windows[3].score
              << " minus_gain=" << minus_gain
              << " plus_gain=" << plus_gain
              << " chosen=" << chosen_phase
              << " direction_confident=" << direction_confident
              << " target_good=" << target_good << "\n" << std::flush;
    return PhaseServoChoice{chosen_phase, chosen_metrics, target_good,
                            direction_confident};
}

bool RuntimeController::stopped() const {
    return stop_requested_ && stop_requested_->load();
}

bool RuntimeController::poll_status(FpgaResponse* response) {
    FpgaResponse status;
    if (!uart_.request(0x02, &status, 400)) return false;
    current_mode_ = status.mode;
    last_status_poll_ = Clock::now();
    if (response) *response = status;
    return true;
}

bool RuntimeController::expected_mode(AutoMode mode) {
    if (Clock::now() - last_status_poll_ <
        std::chrono::milliseconds(kModePollIntervalMs))
        return current_mode_ == static_cast<uint8_t>(mode);
    FpgaResponse status;
    if (!poll_status(&status)) return true;
    return status.mode == static_cast<uint8_t>(mode);
}

bool RuntimeController::read_frame(Frame* frame) {
    if (!camera_.read(frame)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return false;
    }
    last_frame_sequence_ = frame->sequence;
    return true;
}

bool RuntimeController::send_output(double frequency_hz, int phase,
                                    int amplitude, int display_settle_ms) {
    if (!live_calibration_) return false;
    const auto ticket = scheduler_.issue_sine(
        frequency_hz, phase, amplitude,
        std::chrono::milliseconds(std::max(0, display_settle_ms)),
        last_frame_sequence_);
    if (!ticket) return false;
    active_ticket_ = ticket;
    active_output_hz_ = ticket->realized_frequency_hz;
    std::cout << "DDS_FINE generation=" << ticket->generation
              << " requested_hz=" << std::fixed << std::setprecision(6)
              << ticket->requested_frequency_hz << " realized_hz="
              << ticket->realized_frequency_hz << " error_hz="
              << ticket->realized_frequency_hz - ticket->requested_frequency_hz
              << " phase=" << (phase & 0xff) << "\n" << std::flush;
    return true;
}

bool RuntimeController::set_visual_lock(bool locked) {
    if (!scheduler_.set_locked(locked)) return false;
    std::cout << "LOCK_INDICATOR value=" << locked << "\n" << std::flush;
    return true;
}

bool RuntimeController::collect_grid(int budget_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(budget_ms);
    while (!stopped() && !calibrator_.locked() && Clock::now() < deadline) {
        Frame frame;
        if (!read_frame(&frame)) continue;
        const auto now = Clock::now();
        if (last_grid_analysis_ != Clock::time_point{} &&
            now - last_grid_analysis_ <
                std::chrono::milliseconds(kGridAnalysisIntervalMs))
            continue;
        calibrator_.auto_locate(frame.bgr);
        last_grid_analysis_ = Clock::now();
    }
    if (calibrator_.locked()) {
        const auto geometry = calibrator_.geometry();
        std::cout << "GRID_LOCKED roi=" << geometry->roi.x << ','
                  << geometry->roi.y << ',' << geometry->roi.width << ','
                  << geometry->roi.height << "\n" << std::flush;
    }
    return calibrator_.locked();
}

bool RuntimeController::collect_grid_with_exposure_sweep(
    int budget_per_exposure_ms) {
    for (int exposure : config_.grid_exposures) {
        calibrator_.reset();
        last_grid_analysis_ = Clock::time_point{};
        if (!camera_.set_manual_exposure(exposure)) {
            std::cout << "CAMERA_EXPOSURE unsupported=1\n" << std::flush;
            return collect_grid(std::max(budget_per_exposure_ms,
                                         kStartupGridSearchMs));
        }
        int settled_frames = 0;
        while (!stopped() && settled_frames < kExposureSettleFrames) {
            Frame discarded;
            if (read_frame(&discarded)) ++settled_frames;
        }
        std::cout << "CAMERA_EXPOSURE value=" << exposure
                  << " settled_frames=" << settled_frames << "\n"
                  << std::flush;
        if (collect_grid(budget_per_exposure_ms)) {
            std::cout << "CAMERA_EXPOSURE_SELECTED value=" << exposure
                      << "\n" << std::flush;
            // Grid visibility and phase-code contrast have different needs.
            // A brighter fallback may be necessary to fit the gray lattice,
            // but retaining it washes out the yellow marker texture under
            // ambient light. Once the homography is frozen, return to the
            // preferred operational exposure before frequency acquisition.
            const int operational_exposure = config_.grid_exposures.front();
            if (exposure != operational_exposure &&
                camera_.set_manual_exposure(operational_exposure)) {
                int operational_settled_frames = 0;
                while (!stopped() &&
                       operational_settled_frames < kExposureSettleFrames) {
                    Frame discarded;
                    if (read_frame(&discarded))
                        ++operational_settled_frames;
                }
                std::cout << "CAMERA_EXPOSURE_OPERATIONAL value="
                          << operational_exposure << " settled_frames="
                          << operational_settled_frames << "\n"
                          << std::flush;
            }
            return true;
        }
        std::cout << "CAMERA_EXPOSURE_REJECT value=" << exposure
                  << " reason=grid_not_confirmed\n" << std::flush;
    }
    // Do not leave the camera at the last (usually brightest) fallback when
    // the short sweep expires. Background grid search may lock a few frames
    // later, and frequency acquisition must then inherit the preferred
    // low-exposure setting rather than an overexposed phase-code image.
    const int preferred_exposure = config_.grid_exposures.front();
    if (camera_.set_manual_exposure(preferred_exposure)) {
        int settled_frames = 0;
        while (!stopped() && settled_frames < kExposureSettleFrames) {
            Frame discarded;
            if (read_frame(&discarded)) ++settled_frames;
        }
        std::cout << "CAMERA_EXPOSURE_RESTORED value="
                  << preferred_exposure << " settled_frames="
                  << settled_frames << "\n" << std::flush;
    }
    return false;
}

bool RuntimeController::wait_frames(int duration_ms, AutoMode mode) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(duration_ms);
    while (!stopped() && Clock::now() < deadline) {
        Frame discarded;
        read_frame(&discarded);
        if (!expected_mode(mode)) return false;
    }
    return !stopped() && expected_mode(mode);
}

std::optional<ClockCalibration> RuntimeController::read_live_calibration() {
    FpgaResponse response;
    if (!uart_.request(0x03, &response, 500)) return std::nullopt;
    const auto calibration = derive_clock_calibration(response);
    if (!calibration) return std::nullopt;
    live_calibration_ = calibration;
    scheduler_.set_clock_hz(calibration->effective_dds_clock_hz);
    std::cout << "CLOCK_READY sample_sum_q8=" << calibration->sample_sum_q8
              << " cycles=" << calibration->cycle_count
              << " source_grid_hz=" << std::fixed << std::setprecision(1)
              << calibration->source_grid_frequency_hz
              << " residual_hz=" << std::setprecision(6)
              << calibration->residual_hz << " dds_clock_hz="
              << std::setprecision(3) << calibration->effective_dds_clock_hz
              << "\n" << std::flush;
    return calibration;
}

std::optional<RuntimeController::FrequencyCandidates>
RuntimeController::acquire_frequency(AutoMode mode) {
    DdsCommand command;
    command.ftw = ftw_for_frequency(
        100.0, live_calibration_->effective_dds_clock_hz);
    command.amplitude = 102;
    command.flags = 0x80;
    if (!uart_.send_dds(command)) return std::nullopt;
    FpgaResponse acknowledgement;
    if (!uart_.read_response(&acknowledgement, 400, 0x01))
        return std::nullopt;
    active_ticket_.reset();
    active_output_hz_ = 100.0;
    if (!wait_frames(config_.phase_code_settle_ms, mode))
        return std::nullopt;

    std::vector<FrequencyEstimate> estimates;
    PhaseCodeSpec spec;
    const int maximum_attempts = config_.frequency_windows + 3;
    for (int attempt = 0;
         attempt < maximum_attempts &&
         static_cast<int>(estimates.size()) < config_.frequency_windows;
         ++attempt) {
        PhaseCodeMatcher matcher(spec);
        int captured_frames = 0;
        int read_failures = 0;
        int normalized_empty = 0;
        double capture_time_ms = 0.0;
        double processing_time_ms = 0.0;
        const auto deadline = Clock::now() +
            std::chrono::milliseconds(config_.frequency_window_ms);
        while (!stopped() && Clock::now() < deadline) {
            Frame frame;
            const auto capture_started = Clock::now();
            const bool captured = read_frame(&frame);
            capture_time_ms += std::chrono::duration<double, std::milli>(
                Clock::now() - capture_started).count();
            if (!captured) {
                ++read_failures;
                continue;
            }
            ++captured_frames;
            if (!expected_mode(mode)) return std::nullopt;
            const auto processing_started = Clock::now();
            const cv::Mat normalized = calibrator_.normalize(frame.bgr);
            if (normalized.empty()) {
                ++normalized_empty;
                processing_time_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - processing_started).count();
                continue;
            }
            const TraceObservation observation = segmenter_.process(
                normalized, frame.sequence, frame.timestamp);
            matcher.add(observation);
            processing_time_ms += std::chrono::duration<double, std::milli>(
                Clock::now() - processing_started).count();
        }
        const FrequencyEstimate estimate = matcher.estimate();
        std::cout << "FREQUENCY_WINDOW attempt=" << attempt + 1
                  << " valid_index=" << estimates.size() + 1
                  << " frames=" << matcher.frame_count() << " hz="
                  << std::fixed << std::setprecision(1)
                  << estimate.frequency_hz << " runner_hz="
                  << estimate.runner_up_hz << " score="
                  << std::setprecision(5) << estimate.score << " margin="
                  << estimate.margin << " valid=" << estimate.valid
                  << " strategy=" << estimate.strategy
                  << " visual_scale=" << std::setprecision(5)
                  << estimate.visual_scale << " visual_offset="
                  << estimate.visual_offset << " visual_rms_px="
                  << std::setprecision(3) << estimate.visual_residual_px
                  << " geometry_alt_hz=" << std::setprecision(1)
                  << estimate.geometry_alternate_hz
                  << " geometry_alt_kind="
                  << (estimate.geometry_alternate_hz <= 0.0 ? "none" :
                      estimate.geometry_alternate_scaled ? "scaled" :
                                                           "unscaled");
        if (estimate.strategy.find("low_centroid") != std::string::npos &&
            estimate.candidate_scores.size() >= 3) {
            std::cout << " centroid_hz=" << std::setprecision(1)
                      << estimate.candidate_scores[
                             estimate.candidate_scores.size() - 2] * 100.0
                      << " centroid_residual=" << std::setprecision(5)
                       << estimate.candidate_scores.back();
        }
        const int capture_calls = captured_frames + read_failures;
        std::cout << " captured=" << captured_frames
                  << " read_fail=" << read_failures
                  << " normalized_empty=" << normalized_empty
                  << " capture_ms=" << std::setprecision(2)
                  << capture_time_ms / std::max(1, capture_calls)
                  << " process_ms="
                  << processing_time_ms / std::max(1, captured_frames);
        std::cout << "\n" << std::flush;
        if (estimate.valid) {
            estimates.push_back(estimate);
        } else if (attempt + 1 < maximum_attempts) {
            std::cout << "FREQUENCY_WINDOW_REPLACE attempt=" << attempt + 1
                      << " reason=insufficient_or_invalid_frames\n"
                      << std::flush;
        }
    }
    if (estimates.size() < 2) return std::nullopt;

    // The matcher is intentionally conservative and returns a legal 100 Hz
    // bin, but a camera window can move the winning bin by two or three bins
    // while preserving the same physical frequency. Cluster primary bins in
    // a local neighbourhood before requiring temporal support. A distant
    // alias (for example 84.1 kHz in the 82.7 kHz log) remains a separate
    // cluster and cannot win on one noisy window.
    constexpr double kClusterToleranceHz = 300.0;
    struct CandidateVote {
        double hz = 0.0;
        int primary_votes = 0;
        int runner_votes = 0;
        double confidence = 0.0;
        double match_strength = 0.0;
    };
    std::vector<CandidateVote> votes;
    const auto add_vote = [&](double hz, bool primary, double confidence,
                              double match_strength) {
        if (!std::isfinite(hz) || hz < 1'000.0) return;
        const double legal_hz = std::clamp(
            std::round(hz / 100.0) * 100.0, 1'000.0, 100'000.0);
        auto item = std::find_if(votes.begin(), votes.end(),
            [&](const CandidateVote& value) {
                return std::abs(value.hz - legal_hz) < 1.0;
            });
        if (item == votes.end()) {
            votes.push_back({legal_hz, primary ? 1 : 0,
                             primary ? 0 : 1,
                             primary ? std::max(0.0, confidence) : 0.0,
                             primary ? match_strength : 0.0});
        } else {
            if (primary) ++item->primary_votes;
            else ++item->runner_votes;
            if (primary) item->confidence += std::max(0.0, confidence);
            if (primary)
                item->match_strength = std::max(item->match_strength,
                                                match_strength);
        }
    };
    for (const FrequencyEstimate& estimate : estimates) {
        add_vote(estimate.frequency_hz, true, estimate.margin,
                 -estimate.score);
        add_vote(estimate.runner_up_hz, false, 0.0, 0.0);
    }
    std::sort(votes.begin(), votes.end(),
        [](const CandidateVote& left, const CandidateVote& right) {
            return left.hz < right.hz;
        });

    struct CandidateCluster {
        std::vector<std::size_t> members;
        int primary_votes = 0;
        int runner_votes = 0;
        double confidence = 0.0;
        double median_hz = 0.0;
    };
    std::vector<CandidateCluster> clusters;
    for (std::size_t index = 0; index < votes.size();) {
        CandidateCluster cluster;
        cluster.members.push_back(index);
        std::size_t next = index + 1;
        while (next < votes.size() &&
               votes[next].hz - votes[index].hz <=
                   kClusterToleranceHz) {
            cluster.members.push_back(next++);
        }
        for (const std::size_t member : cluster.members) {
            cluster.primary_votes += votes[member].primary_votes;
            cluster.runner_votes += votes[member].runner_votes;
            cluster.confidence += votes[member].confidence;
        }
        const std::size_t middle = cluster.members.size() / 2;
        cluster.median_hz = votes[cluster.members[middle]].hz;
        clusters.push_back(std::move(cluster));
        index = next;
    }
    if (clusters.empty()) return std::nullopt;

    auto primary_cluster = std::max_element(clusters.begin(), clusters.end(),
        [](const CandidateCluster& left, const CandidateCluster& right) {
            if (left.primary_votes != right.primary_votes)
                return left.primary_votes < right.primary_votes;
            if (left.confidence != right.confidence)
                return left.confidence < right.confidence;
            return left.runner_votes < right.runner_votes;
        });
    const int support = primary_cluster->primary_votes;
    if (support < 2) return std::nullopt;

    // Prefer the legal bin with the strongest primary evidence inside the
    // winning cluster. This avoids rounding a 82.8/82.9 pair upward merely
    // because its arithmetic midpoint is 82.85 kHz.
    const auto best_member = [&](const CandidateCluster& cluster,
                                 bool exclude_primary) {
        std::size_t best = cluster.members.front();
        for (const std::size_t member : cluster.members) {
            const CandidateVote& candidate = votes[member];
            const CandidateVote& current = votes[best];
            const int candidate_weight = candidate.primary_votes * 4 +
                                          candidate.runner_votes;
            const int current_weight = current.primary_votes * 4 +
                                        current.runner_votes;
            if (exclude_primary && candidate.primary_votes > 0 &&
                current.primary_votes == 0) {
                best = member;
                continue;
            }
            if (candidate_weight > current_weight ||
                (candidate_weight == current_weight &&
                 candidate.confidence > current.confidence) ||
                (candidate_weight == current_weight &&
                 candidate.confidence == current.confidence &&
                 std::abs(candidate.hz - cluster.median_hz) <
                     std::abs(current.hz - cluster.median_hz)))
                best = member;
        }
        return best;
    };
    const std::size_t primary_member = best_member(*primary_cluster, false);
    const double consensus = votes[primary_member].hz;
    std::optional<double> alternate;
    // A nearby alternative needs independent evidence: either it won a
    // window, or it was runner-up twice. This prevents a weak runner from the
    // same wrong alias cluster hiding a strong primary from another window.
    std::optional<std::size_t> nearby_member;
    for (const std::size_t member : primary_cluster->members) {
        if (std::abs(votes[member].hz - consensus) < 1.0) continue;
        if (votes[member].primary_votes == 0 &&
            votes[member].runner_votes < 2)
            continue;
        if (!nearby_member ||
            votes[member].primary_votes * 4 + votes[member].runner_votes >
                votes[*nearby_member].primary_votes * 4 +
                    votes[*nearby_member].runner_votes ||
            (votes[member].primary_votes * 4 + votes[member].runner_votes ==
                 votes[*nearby_member].primary_votes * 4 +
                     votes[*nearby_member].runner_votes &&
             (std::abs(votes[member].hz - consensus) <
                  std::abs(votes[*nearby_member].hz - consensus) ||
              (std::abs(votes[member].hz - consensus) ==
                   std::abs(votes[*nearby_member].hz - consensus) &&
               votes[member].hz > votes[*nearby_member].hz))))
            nearby_member = member;
    }
    if (nearby_member) alternate = votes[*nearby_member].hz;
    if (!alternate) {
        std::vector<std::size_t> other_members;
        for (std::size_t member = 0; member < votes.size(); ++member) {
            if (member != primary_member) other_members.push_back(member);
        }
        if (!other_members.empty()) {
            const auto best_other = *std::max_element(
                other_members.begin(), other_members.end(),
                [&](std::size_t left, std::size_t right) {
                    const CandidateVote& a = votes[left];
                    const CandidateVote& b = votes[right];
                    const int wa = a.primary_votes * 4 + a.runner_votes;
                    const int wb = b.primary_votes * 4 + b.runner_votes;
                    if (wa != wb) return wa < wb;
                    if (a.match_strength != b.match_strength)
                        return a.match_strength < b.match_strength;
                    return a.confidence < b.confidence;
                });
            alternate = votes[best_other].hz;
        }
    }
    std::cout << "FREQUENCY_CLUSTER primary_hz=" << consensus
              << " support=" << support
              << " tolerance_hz=" << kClusterToleranceHz
              << " members=";
    for (const std::size_t member : primary_cluster->members)
        std::cout << votes[member].hz << ':' << votes[member].primary_votes
                  << '/' << votes[member].runner_votes << ',';
    std::cout << "\n" << std::flush;
    std::cout << "FREQUENCY_LOCK hz=" << std::fixed << std::setprecision(1)
              << consensus << " support=" << support << '/'
              << estimates.size() << " alternate_hz="
              << (alternate ? *alternate : 0.0) << "\n" << std::flush;
    std::vector<double> fallback_hz;
    const auto add_fallback = [&](double hz) {
        if (hz < 1'000.0 || hz > 100'000.0 ||
            std::abs(hz - consensus) < 1.0 ||
            std::any_of(fallback_hz.begin(), fallback_hz.end(),
                [&](double existing) {
                    return std::abs(existing - hz) < 1.0;
                }))
            return;
        fallback_hz.push_back(hz);
    };
    if (alternate) add_fallback(*alternate);
    std::vector<double> geometry_alternates;
    for (const FrequencyEstimate& estimate : estimates) {
        if (estimate.geometry_alternate_hz >= 1'000.0 &&
            estimate.geometry_alternate_hz <= 100'000.0) {
            geometry_alternates.push_back(
                std::round(estimate.geometry_alternate_hz / 100.0) * 100.0);
        }
    }
    if (!geometry_alternates.empty()) {
        std::sort(geometry_alternates.begin(), geometry_alternates.end());
        const double geometry_fallback = geometry_alternates[
            geometry_alternates.size() / 2];
        add_fallback(geometry_fallback);
        std::cout << "FREQUENCY_GEOMETRY_FALLBACK hz="
                  << geometry_fallback << " samples=";
        for (double hz : geometry_alternates) std::cout << hz << ',';
        std::cout << " order=after_runner\n" << std::flush;
    }
    // A systematic one-bin camera bias can make the primary and its runner
    // adjacent wrong bins. A filled rectangle is distinctive enough to
    // authorize checking both immediate neighbours before a full rescan.
    add_fallback(consensus - 100.0);
    add_fallback(consensus + 100.0);
    return FrequencyCandidates{consensus, alternate, fallback_hz, support,
                               static_cast<int>(estimates.size())};
}

ShapeMetrics RuntimeController::median_metrics(
    std::vector<ShapeMetrics> samples) {
    ShapeMetrics result;
    if (samples.empty()) return result;
    result = samples[samples.size() / 2];
    result.valid = true;
    result.pixels = static_cast<int>(std::lround(median_value(
        samples, [](const ShapeMetrics& value) { return value.pixels; })));
    result.center_div.x = median_value(
        samples, [](const ShapeMetrics& value) { return value.center_div.x; });
    result.center_div.y = median_value(
        samples, [](const ShapeMetrics& value) { return value.center_div.y; });
    result.span_div.x = median_value(
        samples, [](const ShapeMetrics& value) { return value.span_div.x; });
    result.span_div.y = median_value(
        samples, [](const ShapeMetrics& value) { return value.span_div.y; });
    result.coverage = median_value(
        samples, [](const ShapeMetrics& value) { return value.coverage; });
    result.correlation = median_value(
        samples, [](const ShapeMetrics& value) { return value.correlation; });
    result.thinness = median_value(
        samples, [](const ShapeMetrics& value) { return value.thinness; });
    result.minor_rms_div = median_value(
        samples, [](const ShapeMetrics& value) { return value.minor_rms_div; });
    result.radial_cv = median_value(
        samples, [](const ShapeMetrics& value) { return value.radial_cv; });
    result.symmetry_x = median_value(
        samples, [](const ShapeMetrics& value) { return value.symmetry_x; });
    result.symmetry_y = median_value(
        samples, [](const ShapeMetrics& value) { return value.symmetry_y; });
    result.crossing_fill = median_value(
        samples, [](const ShapeMetrics& value) { return value.crossing_fill; });
    result.crossing_offset_y_div = median_value(
        samples, [](const ShapeMetrics& value) {
            return value.crossing_offset_y_div;
        });
    result.phase_feature = median_value(
        samples, [](const ShapeMetrics& value) { return value.phase_feature; });
    return result;
}

std::optional<ShapeMetrics> RuntimeController::measure_shape(
    AutoMode mode, int phase, int settle_ms, int measure_ms,
    bool send_command, std::vector<PhaseTrackSample>* phase_track,
    Clock::time_point track_origin, ShapeWindowStats* window_stats,
    bool stop_on_frequency_mismatch) {
    if (window_stats) *window_stats = {};
    const double output_hz = active_output_hz_;
    if (send_command && !send_output(output_hz, phase, 102, settle_ms))
        return std::nullopt;
    if (!send_command && settle_ms > 0 && !wait_frames(settle_ms, mode))
        return std::nullopt;

    std::vector<ShapeMetrics> samples;
    std::optional<ShapeMetrics> early_frequency_mismatch;
    int consecutive_frequency_mismatch = 0;
    const auto measure_start = send_command && active_ticket_ ?
        active_ticket_->visible_after : Clock::now();
    const auto deadline = measure_start + std::chrono::milliseconds(measure_ms);
    while (!stopped() && Clock::now() < deadline) {
        Frame frame;
        if (!read_frame(&frame)) continue;
        if (!expected_mode(mode)) return std::nullopt;
        if (send_command && active_ticket_ &&
            !scheduler_.frame_is_fresh(*active_ticket_, frame))
            continue;
        const cv::Mat normalized = calibrator_.normalize(frame.bgr);
        const TraceObservation observation = segmenter_.process(
            normalized, frame.sequence, frame.timestamp);
        const ShapeMetrics metrics = shape_observer_.analyze(observation);
        if (metrics.valid) {
            samples.push_back(metrics);
            if (window_stats) {
                ++window_stats->valid_frames;
                if (shape_observer_.shape_ok(mode, metrics)) {
                    ++window_stats->good_frames;
                    const double score =
                        shape_observer_.search_score(mode, metrics);
                    if (!window_stats->has_best_good ||
                        score > window_stats->best_good_score) {
                        window_stats->has_best_good = true;
                        window_stats->best_good = metrics;
                        window_stats->best_good_score = score;
                    }
                }
            }
            if (phase_track && metrics.coverage < 0.35) {
                phase_track->push_back({
                    std::chrono::duration<double>(
                        frame.timestamp - track_origin).count(),
                    kTwoPi * (phase & 0xff) / 256.0,
                    mode == AutoMode::FigureEight ?
                        metrics.phase_feature : metrics.correlation});
            }
            if (stop_on_frequency_mismatch &&
                shape_observer_.frequency_mismatch_fill(metrics))
                ++consecutive_frequency_mismatch;
            else
                consecutive_frequency_mismatch = 0;
            if (stop_on_frequency_mismatch && samples.size() >= 3 &&
                consecutive_frequency_mismatch >= 2) {
                early_frequency_mismatch = metrics;
                std::cout << "SHAPE_EARLY_REJECT phase=" << (phase & 0xff)
                          << " reason=filled_rectangle frames="
                          << samples.size() << "\n" << std::flush;
                break;
            }
        }
    }
    const int valid_frame_count = static_cast<int>(samples.size());
    ShapeMetrics metrics = early_frequency_mismatch ?
        *early_frequency_mismatch : median_metrics(std::move(samples));
    std::cout << "SHAPE phase=" << (phase & 0xff)
              << " valid=" << metrics.valid << " score=" << std::fixed
              << std::setprecision(4)
              << shape_observer_.search_score(mode, metrics)
              << " corr=" << metrics.correlation
              << " thin=" << metrics.thinness
              << " minor_div=" << metrics.minor_rms_div
              << " span=" << metrics.span_div.x << ',' << metrics.span_div.y
              << " radial=" << metrics.radial_cv << " sym="
              << metrics.symmetry_x << ',' << metrics.symmetry_y
              << " cross=" << metrics.crossing_fill
              << " cross_y=" << metrics.crossing_offset_y_div
              << " feature="
              << metrics.phase_feature << " good="
              << shape_observer_.shape_ok(mode, metrics)
              << " frames=" << valid_frame_count
              << " good_hits="
              << (window_stats ? window_stats->good_frames : 0) << "\n"
              << std::flush;
    return metrics;
}

bool RuntimeController::run_abba_probe(
    AutoMode mode, double input_frequency_hz) {
    active_output_hz_ = input_frequency_hz *
        (mode == AutoMode::FigureEight ? 2.0 : 1.0);
    const int center = config_.abba_center_phase & 0xff;
    const int delta = config_.abba_delta_phase;
    const std::array<int, 4> pattern{
        (center - delta) & 0xff,
        (center + delta) & 0xff,
        (center + delta) & 0xff,
        (center - delta) & 0xff};
    std::vector<PhaseTrackSample> samples;
    const auto origin = Clock::now();
    std::cout << "ABBA_PROBE_START input_hz=" << std::fixed
              << std::setprecision(6) << input_frequency_hz
              << " output_hz=" << active_output_hz_
              << " center_phase=" << center
              << " delta_phase=" << delta
              << " settle_ms=" << config_.abba_settle_ms
              << " measure_ms=" << config_.abba_measure_ms
              << " rounds=" << config_.abba_rounds << "\n"
              << std::flush;
    for (int round = 0; round < config_.abba_rounds; ++round) {
        for (size_t index = 0; index < pattern.size(); ++index) {
            ShapeWindowStats stats;
            const auto metrics = measure_shape(
                mode, pattern[index], config_.abba_settle_ms,
                config_.abba_measure_ms, true, &samples, origin, &stats);
            if (!metrics || !expected_mode(mode)) return false;
            if (shape_observer_.frequency_mismatch_fill(*metrics)) {
                std::cout << "ABBA_PROBE_REJECT reason=filled_rectangle"
                          << " round=" << round
                          << " index=" << index << "\n" << std::flush;
                return false;
            }
        }
    }
    const double reference_seconds = std::chrono::duration<double>(
        Clock::now() - origin).count();
    for (const PhaseTrackSample& sample : samples) {
        std::cout << "ABBA_SAMPLE time_s=" << std::fixed
                  << std::setprecision(6) << sample.time_seconds
                  << " command_phase_rad=" << sample.command_phase_rad
                  << " response=" << sample.correlation << "\n";
    }
    const auto fit = fit_phase_track(
        samples, reference_seconds, 1.0, 0.0005);
    if (!fit) {
        std::cout << "ABBA_PROBE_RESULT valid=0 samples=" << samples.size()
                  << "\n" << std::flush;
        return false;
    }
    const auto solution = solve_quadrature_phase(
        mode, std::cos(fit->relative_phase_rad),
        -std::sin(fit->relative_phase_rad));
    std::cout << "ABBA_PROBE_RESULT valid=1 samples=" << samples.size()
              << " error_hz=" << std::fixed << std::setprecision(6)
              << fit->frequency_error_hz
              << " relative_phase_rad=" << fit->relative_phase_rad
              << " rms=" << fit->rms
              << " amplitude=" << fit->amplitude
              << " reference_s=" << reference_seconds
              << " predicted_phase="
              << (solution ? solution->phase : -1) << "\n"
              << std::flush;
    return solution.has_value();
}

std::optional<RuntimeController::ShapeChoice>
RuntimeController::acquire_shape(AutoMode mode, double input_frequency_hz,
                                 bool* frequency_mismatch,
                                 bool* frequency_confirmed,
                                 int* frequency_evidence_count,
                                 double* output_frequency_trim_hz) {
    const bool initial_line_circle =
        mode == AutoMode::Line || mode == AutoMode::Circle;
    if (frequency_mismatch) *frequency_mismatch = false;
    int coherent_observations = frequency_evidence_count ?
        std::clamp(*frequency_evidence_count, 0, 3) : 0;
    if (frequency_confirmed)
        *frequency_confirmed = coherent_observations >= 3;
    const auto enough_circle_frames = [&](const ShapeWindowStats& stats) {
        return stats.good_frames >= 3 &&
               stats.good_frames * 3 >= std::max(1, stats.valid_frames);
    };
    const auto good_from_window = [&](const ShapeMetrics& aggregate,
                                      const ShapeWindowStats& stats)
        -> std::optional<ShapeMetrics> {
        if (shape_observer_.shape_ok(mode, aggregate)) return aggregate;
        // The scheduler has already excluded buffered pre-command frames and
        // the display-settle delay has elapsed. On a low-frame-rate camera a
        // correctly refreshed scope image may occur only once in this window;
        // do not average that real target away. Line/circle still complete all
        // three calibration phases before this evidence can be accepted.
        if (stats.has_best_good &&
            (mode == AutoMode::Circle ? enough_circle_frames(stats) :
             (stats.good_frames >= 2 ||
              stats.valid_frames <= (initial_line_circle ? 2 : 4))))
            return stats.best_good;
        return std::nullopt;
    };
    const auto record_frequency_evidence = [&](int phase,
                                                const ShapeMetrics& metrics) {
        if (!shape_observer_.frequency_shape_coherent(mode, metrics)) return;
        const int previous = coherent_observations;
        coherent_observations = std::min(3, coherent_observations + 1);
        if (frequency_evidence_count)
            *frequency_evidence_count = coherent_observations;
        std::cout << "FREQUENCY_SHAPE_EVIDENCE hz=" << input_frequency_hz
                   << " phase=" << (phase & 0xff)
                   << " count=" << coherent_observations << "/3\n"
                   << std::flush;
        if (coherent_observations >= 3) {
            if (frequency_confirmed) *frequency_confirmed = true;
            if (previous < 3) {
                std::cout << "FREQUENCY_VISUAL_CONFIRMED hz="
                          << input_frequency_hz
                          << " reason=three_bounded_shapes\n"
                          << std::flush;
            }
        }
    };
    const auto record_window_frequency_evidence = [&]
        (int phase, const ShapeMetrics& aggregate,
         const ShapeWindowStats& stats) {
        if (shape_observer_.frequency_shape_coherent(mode, aggregate))
            record_frequency_evidence(phase, aggregate);
        else if (stats.has_best_good)
            record_frequency_evidence(phase, stats.best_good);
    };
    const auto reject_frequency_mismatch = [&](const ShapeMetrics& metrics) {
        const char* reason = nullptr;
        if (shape_observer_.frequency_mismatch_fill(metrics))
            reason = "filled_rectangle";
        else if (shape_observer_.frequency_mismatch_harmonic(mode, metrics))
            reason = "harmonic_shape";
        if (!reason) return false;
        if (frequency_mismatch) *frequency_mismatch = true;
        std::cout << "FREQUENCY_REJECT hz=" << input_frequency_hz
                  << " reason=" << reason << "\n" << std::flush;
        return true;
    };
    const double nominal_output_hz = input_frequency_hz *
        (mode == AutoMode::FigureEight ? 2.0 : 1.0);
    if (!phase_search_context_hz_ ||
        std::abs(*phase_search_context_hz_ - input_frequency_hz) >= 1.0 ||
        phase_search_context_mode_ != static_cast<uint8_t>(mode)) {
        phase_search_seed_.reset();
        phase_search_context_hz_ = input_frequency_hz;
        phase_search_context_mode_ = static_cast<uint8_t>(mode);
    }
    if (!visual_trim_context_hz_ ||
        std::abs(*visual_trim_context_hz_ - input_frequency_hz) >= 1.0 ||
        visual_trim_context_mode_ != static_cast<uint8_t>(mode)) {
        pending_visual_error_hz_.reset();
        visual_trim_context_hz_ = input_frequency_hz;
        visual_trim_context_mode_ = static_cast<uint8_t>(mode);
    }
    const double trim_limit_hz = initial_line_circle ? 0.25 : 1.0;
    const double existing_trim_hz = output_frequency_trim_hz ?
        std::clamp(*output_frequency_trim_hz,
                   -trim_limit_hz, trim_limit_hz) : 0.0;
    active_output_hz_ = nominal_output_hz + existing_trim_hz;
    if (coherent_observations >= 3 && phase_search_seed_) {
        bool seed_fill = false;
        std::cout << "PHASE_LOCAL_RESUME phase="
                  << phase_search_seed_->phase
                  << " frequency_hz=" << input_frequency_hz << "\n"
                  << std::flush;
        const auto resumed = refine_phase_locally(
            mode, phase_search_seed_->phase, phase_search_seed_->metrics,
            4, true, &seed_fill, 8);
        if (seed_fill) {
            if (frequency_mismatch) *frequency_mismatch = true;
            phase_search_seed_.reset();
            return std::nullopt;
        }
        if (resumed && resumed->valid) {
            phase_search_seed_.reset();
            return resumed;
        }
        if (resumed) phase_search_seed_ = resumed;
        return std::nullopt;
    }
    std::vector<PhaseTrackSample> phase_track;
    const auto track_origin = Clock::now();
    ShapeWindowStats phase0_stats;
    const auto phase0 = measure_shape(
        mode, 0, initial_line_circle ? 900 : config_.shape_settle_ms,
        config_.shape_window_ms, true, &phase_track, track_origin,
        &phase0_stats, !initial_line_circle);
    if (!phase0 || !expected_mode(mode)) return std::nullopt;
    if (reject_frequency_mismatch(*phase0)) return std::nullopt;
    record_window_frequency_evidence(0, *phase0, phase0_stats);
    const auto phase0_good = good_from_window(*phase0, phase0_stats);
    ShapeWindowStats phase64_stats;
    const auto phase64 = measure_shape(
        mode, 64, initial_line_circle ? 900 : config_.shape_settle_ms,
        config_.shape_window_ms, true, &phase_track, track_origin,
        &phase64_stats, !initial_line_circle);
    if (!phase64 || !expected_mode(mode)) return std::nullopt;
    if (reject_frequency_mismatch(*phase64)) return std::nullopt;
    record_window_frequency_evidence(64, *phase64, phase64_stats);
    const auto phase64_good = good_from_window(*phase64, phase64_stats);
    // A third independent phase improves the time-aware fit and completes
    // the calibration sequence before any line/circle can be accepted.
    ShapeWindowStats phase128_stats;
    const auto phase128 = measure_shape(
        mode, 128, initial_line_circle ? 900 : config_.shape_settle_ms,
        config_.shape_window_ms, true, &phase_track, track_origin,
        &phase128_stats, !initial_line_circle);
    if (!phase128 || !expected_mode(mode)) return std::nullopt;
    if (reject_frequency_mismatch(*phase128)) return std::nullopt;
    record_window_frequency_evidence(128, *phase128, phase128_stats);
    const auto phase128_good = good_from_window(*phase128, phase128_stats);

    // Always complete the three independent calibration phases. If one of
    // them already showed the requested target, return to the best such phase
    // and require two fresh windows there. This preserves calibration while
    // preventing the later fit from stepping away from a visible circle,
    // line, or figure eight.
    struct CalibrationTarget {
        int phase = 0;
        ShapeMetrics metrics;
        double score = -1.0e9;
    };
    std::optional<CalibrationTarget> calibration_target;
    const auto consider_calibration_target =
        [&](int candidate_phase,
            const std::optional<ShapeMetrics>& candidate) {
            if (!candidate) return;
            const double score = shape_observer_.search_score(
                mode, *candidate);
            if (!calibration_target || score > calibration_target->score)
                calibration_target = CalibrationTarget{
                    candidate_phase, *candidate, score};
        };
    consider_calibration_target(0, phase0_good);
    consider_calibration_target(64, phase64_good);
    consider_calibration_target(128, phase128_good);
    if (calibration_target) {
        if (coherent_observations >= 3) {
            if (!send_output(active_output_hz_, calibration_target->phase,
                             102, 0))
                return std::nullopt;
            phase_search_seed_.reset();
            std::cout << "PHASE_ACCEPTED phase="
                      << calibration_target->phase
                      << " score=" << std::fixed << std::setprecision(4)
                      << calibration_target->score
                      << " source=most_likely_calibration"
                      << " frequency_evidence=" << coherent_observations
                      << "/3 fine_adjustment=non_blocking\n" << std::flush;
            return ShapeChoice{calibration_target->phase,
                               calibration_target->metrics, true};
        }
        bool calibration_fill = false;
        std::cout << "PHASE_CALIBRATION_TARGET phase="
                  << calibration_target->phase
                  << " score=" << std::fixed << std::setprecision(4)
                  << calibration_target->score
                  << " action=return_and_confirm\n" << std::flush;
        const auto calibrated = refine_phase_locally(
            mode, calibration_target->phase, calibration_target->metrics,
            2, true, &calibration_fill, 8);
        if (calibration_fill) {
            if (frequency_mismatch) *frequency_mismatch = true;
            return std::nullopt;
        }
        if (calibrated && calibrated->valid) {
            phase_search_seed_.reset();
            return calibrated;
        }
        if (calibrated) phase_search_seed_ = calibrated;
    }

    const double response0 = mode == AutoMode::FigureEight ?
        phase0->phase_feature : phase0->correlation;
    const double response64 = mode == AutoMode::FigureEight ?
        phase64->phase_feature : phase64->correlation;
    const double track_reference_seconds = std::chrono::duration<double>(
        Clock::now() - track_origin).count();
    const auto track_fit = fit_phase_track(
        phase_track, track_reference_seconds);
    const bool track_fit_usable = track_fit &&
        track_fit->amplitude >= 0.15 &&
        track_fit->rms <= (initial_line_circle ? 0.30 : 0.34) &&
        std::abs(track_fit->frequency_error_hz) <= 1.95;
    const double track_error_magnitude_hz = track_fit ?
        std::abs(track_fit->frequency_error_hz) : 0.0;
    const bool track_fit_safe_for_phase = track_fit &&
        track_fit->amplitude >= 0.50 &&
        track_error_magnitude_hz <= 0.80 &&
        track_fit->rms <= 0.40;

    double applied_frequency_correction_hz = 0.0;
    bool trim_error_confirmed = false;
    // The calibration commands are acquired in monotonic 0/64/128 order.
    // Command phase and capture time are therefore strongly correlated: the
    // fit can predict the phase at the end of the sweep, but its frequency
    // sign is not independently observable. Feeding that value into the DDS
    // created the measured 48000 -> 47999.975 Hz runaway. Frequency control
    // is deferred to the lock servo, where several independently recovered
    // optimum phases establish an unambiguous signed slope.
    if (output_frequency_trim_hz)
        *output_frequency_trim_hz = existing_trim_hz;
    if (track_fit && track_fit->rms <= 0.30 &&
               std::abs(track_fit->frequency_error_hz) < 0.015) {
        pending_visual_error_hz_.reset();
    }

    std::optional<QuadraturePhaseSolution> solution;
    if (track_fit_usable) {
        // Predict the phase at the end of the sequential camera sweep. This
        // compensates phase motion during 0/64/128/192 acquisition without
        // feeding the short-window frequency estimate into the DDS.
        const double fit_response0 = std::cos(track_fit->relative_phase_rad);
        const double fit_response64 = -std::sin(track_fit->relative_phase_rad);
        solution = solve_quadrature_phase(
            mode, fit_response0, fit_response64);
        std::cout << "PHASE_TRACK samples=" << phase_track.size()
                  << " error_hz=" << std::fixed << std::setprecision(6)
                  << track_fit->frequency_error_hz
                  << " correction_hz="
                  << applied_frequency_correction_hz
                  << " trim_hz="
                  << (output_frequency_trim_hz ?
                          *output_frequency_trim_hz : 0.0)
                  << " trim_usable=0"
                  << " trim_confirmed=" << trim_error_confirmed
                  << " trim_pending_hz="
                  << (pending_visual_error_hz_ ?
                          *pending_visual_error_hz_ : 0.0)
                  << " rms=" << track_fit->rms
                  << " amplitude=" << track_fit->amplitude
                  << " relative_phase_rad="
                  << track_fit->relative_phase_rad
                  << " reference_s=" << track_reference_seconds
                  << " phase_fit_safe=" << track_fit_safe_for_phase
                  << " trim_suppressed=1"
                  << " trim_policy=locked_phase_trajectory_only"
                  << " usable=1\n" << std::flush;
    } else {
        solution = solve_quadrature_phase(mode, response0, response64);
        std::cout << "PHASE_TRACK samples=" << phase_track.size()
                  << " error_hz=" << std::fixed << std::setprecision(6)
                  << (track_fit ? track_fit->frequency_error_hz : 0.0)
                  << " correction_hz="
                  << applied_frequency_correction_hz
                  << " trim_hz="
                  << (output_frequency_trim_hz ?
                          *output_frequency_trim_hz : 0.0)
                  << " trim_usable=0"
                  << " trim_confirmed=" << trim_error_confirmed
                  << " trim_pending_hz="
                  << (pending_visual_error_hz_ ?
                          *pending_visual_error_hz_ : 0.0)
                  << " rms="
                  << (track_fit ? track_fit->rms : 1.0)
                  << " amplitude="
                  << (track_fit ? track_fit->amplitude : 0.0)
                  << " reference_s=" << track_reference_seconds
                  << " usable=0 strategy=raw_quadrature"
                  << " trim_policy=locked_phase_trajectory_only"
                  << "\n" << std::flush;
    }
    if (!solution) return std::nullopt;
    std::cout << "PHASE_QUADRATURE r0=" << std::fixed
              << std::setprecision(5) << response0 << " r64=" << response64
              << " amplitude=" << solution->response_amplitude
              << " predicted_phase=" << solution->phase << "\n"
              << std::flush;

    ShapeWindowStats predicted_stats;
    const auto predicted = measure_shape(
        mode, solution->phase,
        initial_line_circle ? 900 : config_.shape_settle_ms,
        config_.shape_window_ms, true, nullptr, {}, &predicted_stats,
        !initial_line_circle);
    if (!predicted || !expected_mode(mode))
        return std::nullopt;
    if (reject_frequency_mismatch(*predicted)) return std::nullopt;
    record_window_frequency_evidence(
        solution->phase, *predicted, predicted_stats);
    const auto predicted_good = good_from_window(*predicted, predicted_stats);
    std::cout << "PHASE_PREDICTED phase=" << solution->phase
              << " provisional_good=" << predicted_good.has_value()
              << " good_hits=" << predicted_stats.good_frames
              << " confirm=" << (coherent_observations >= 3 ? 0 : 1)
              << "\n" << std::flush;

    if (coherent_observations >= 3) {
        phase_search_seed_.reset();
        std::cout << "PHASE_ACCEPTED phase=" << solution->phase
                  << " score=" << std::fixed << std::setprecision(4)
                  << shape_observer_.search_score(mode, *predicted)
                  << " source=most_likely_calibration"
                  << " frequency_evidence=" << coherent_observations
                  << "/3 fine_adjustment=non_blocking\n" << std::flush;
        return ShapeChoice{solution->phase, *predicted,
                           predicted_good.has_value()};
    }

    // The predicted window can still contain persistence from the phase-64
    // calibration image. Keep the phase unchanged through an independent
    // confirmation window. A strict hit from either fresh window is retained
    // because a slow processor may only analyze one target frame.
    ShapeWindowStats confirmed_stats;
    const auto confirmed = measure_shape(
        mode, solution->phase, 300, 350, false, nullptr, {},
        &confirmed_stats);
    if (!confirmed || reject_frequency_mismatch(*confirmed))
        return std::nullopt;
    record_window_frequency_evidence(
        solution->phase, *confirmed, confirmed_stats);
    const auto confirmed_good = good_from_window(
        *confirmed, confirmed_stats);
    const bool two_window_good = predicted_good && confirmed_good;
    std::cout << "PHASE_CONFIRMED phase=" << solution->phase
              << " good=" << two_window_good
              << " source=" << (two_window_good ? "two_fresh_windows" :
                                                   "local_refinement")
              << " good_hits="
              << predicted_stats.good_frames + confirmed_stats.good_frames
              << "\n" << std::flush;
    if (two_window_good) {
        phase_search_seed_.reset();
        return ShapeChoice{solution->phase, *confirmed_good, true};
    }

    const ShapeMetrics& near_metrics =
        shape_observer_.search_score(mode, *predicted) >
                shape_observer_.search_score(mode, *confirmed) ?
            *predicted : *confirmed;
    bool local_fill = false;
    const auto refined = refine_phase_locally(
        mode, solution->phase, near_metrics, 4, true, &local_fill, 8);
    if (local_fill && frequency_mismatch) *frequency_mismatch = true;
    if (refined && refined->valid) {
        phase_search_seed_.reset();
        return refined;
    }
    if (refined) phase_search_seed_ = refined;
    return std::nullopt;
}

RuntimeController::ReacquireTarget RuntimeController::maintain_lock(
    AutoMode mode, double input_frequency_hz, int phase,
    bool frequency_confirmed) {
    if (!send_output(active_output_hz_, phase, 102, 0) ||
        !set_visual_lock(true))
        return ReacquireTarget::None;
    std::cout << "LOCKED mode=" << static_cast<int>(mode)
              << " input_hz=" << std::fixed << std::setprecision(1)
              << input_frequency_hz << " output_hz=" << active_output_hz_
              << " phase=" << phase
              << " led=steady minimum_hold_ms=5000\n" << std::flush;

    VisualLockStateMachine lock_state({
        std::chrono::milliseconds(5'000), 3,
        std::chrono::milliseconds(5'000)});
    lock_state.reset(Clock::now());
    int consecutive_frequency_mismatch = 0;
    const double nominal_output_hz = input_frequency_hz *
        (mode == AutoMode::FigureEight ? 2.0 : 1.0);
    const double trim_limit_hz = mode == AutoMode::FigureEight ? 1.0 :
        mode == AutoMode::Circle ? 0.05 : 0.25;
    const auto lock_started = Clock::now();
    const auto servo_origin = Clock::now();
    std::vector<VisualPhaseSample> phase_history;
    int last_observed_phase = phase & 0xff;
    int unwrapped_phase = last_observed_phase;
    phase_history.push_back({0.0, unwrapped_phase});
    while (!stopped() && expected_mode(mode)) {
        ShapeWindowStats hold_stats;
        const auto metrics = measure_shape(
            mode, phase, 0, config_.hold_window_ms, false, nullptr, {},
            &hold_stats);
        if (!metrics) {
            std::cout << "LOCK_HOLD_CONTINUE reason=no_valid_shape"
                      << " phase=" << (phase & 0xff) << "\n" << std::flush;
            continue;
        }
        const bool explicit_frequency_mismatch =
            shape_observer_.frequency_mismatch_fill(*metrics);
        if (explicit_frequency_mismatch)
            ++consecutive_frequency_mismatch;
        else
            consecutive_frequency_mismatch = 0;
        bool good = shape_observer_.shape_ok(mode, *metrics) ||
            (hold_stats.has_best_good &&
             (mode == AutoMode::Line ||
              (mode == AutoMode::Circle && hold_stats.good_frames >= 3 &&
               hold_stats.good_frames * 3 >=
                   std::max(1, hold_stats.valid_frames)) ||
              hold_stats.good_frames >= 2 || hold_stats.valid_frames <= 4));

        if (consecutive_frequency_mismatch >= 3) {
            set_visual_lock(false);
            std::cout << "REACQUIRE reason=confirmed_filled_rectangle"
                      << " phase_code=1 led=blink\n" << std::flush;
            return ReacquireTarget::Frequency;
        }

        // A circle can be geometrically valid while its correlation is near
        // the servo trigger (for example corr=0.245 on a real round trace).
        // Do not pull an already-good circle toward an ellipse just because
        // that noisy scalar moved slightly. Re-enter phase search only after
        // the shape itself fails; the next pass will still correct genuine
        // drift.
        const bool servo_needed = mode == AutoMode::Circle ? !good :
            (!good || shape_observer_.phase_servo_needed(mode, *metrics));
        const bool minimum_hold_elapsed = Clock::now() - lock_started >=
            std::chrono::seconds(5);
        // Once a figure eight has passed acquisition and confirmation, keep
        // that exact DDS command. Post-lock probing previously pulled a valid
        // target away or into a cusp; only an explicit filled rectangle may
        // force frequency reacquisition.
        const bool phase_servo_allowed =
            mode != AutoMode::FigureEight && minimum_hold_elapsed;
        if (phase_servo_allowed && servo_needed &&
            !explicit_frequency_mismatch) {
            bool local_fill = false;
            const int previous_phase = phase;
            std::optional<PhaseServoChoice> refined;
            if (mode == AutoMode::Circle) {
                refined = probe_circle_centered(
                    phase, *metrics, &local_fill);
            } else {
                // Line maintenance uses the drift-cancelling side probe.
                refined = probe_phase_abba(
                    mode, phase, *metrics, &local_fill);
            }
            if (local_fill) {
                ++consecutive_frequency_mismatch;
                continue;
            }
            if (refined) {
                phase = refined->phase;
                good = refined->target_good ||
                       (phase == previous_phase && good);
                std::cout << "LOCK_PHASE_ADJUST phase=" << (phase & 0xff)
                          << " target_good=" << good
                          << " direction_confident="
                          << refined->direction_confident
                          << " frequency_retained=1 led=steady\n"
                          << std::flush;
                const int phase_delta = signed_phase_delta(
                    phase, last_observed_phase);
                if (refined->direction_confident && phase_delta != 0) {
                    unwrapped_phase += signed_phase_delta(
                        phase, last_observed_phase);
                    last_observed_phase = phase;
                    phase_history.push_back({
                        std::chrono::duration<double>(
                            Clock::now() - servo_origin).count(),
                        unwrapped_phase});
                    const bool circle_trim_ready =
                        mode == AutoMode::Circle && phase_history.size() >= 4;
                    const bool other_trim_ready =
                        mode != AutoMode::Circle && phase_history.size() >= 3;
                    const auto trim = circle_trim_ready ?
                        estimate_visual_frequency_trim(
                            phase_history, 3.0, 6, 0.008) :
                        other_trim_ready ?
                            estimate_visual_frequency_trim(
                                phase_history, 1.5, 2, 0.012) :
                            std::nullopt;
                    if (trim) {
                        const double frequency_servo_gain =
                            mode == AutoMode::Circle ? 0.50 : 0.65;
                        const double requested_hz = std::clamp(
                            active_output_hz_ + frequency_servo_gain *
                                trim->output_correction_hz,
                            nominal_output_hz - trim_limit_hz,
                            nominal_output_hz + trim_limit_hz);
                        std::cout << "LOCK_FREQUENCY_TRIM phase_rate="
                                  << std::fixed << std::setprecision(5)
                                  << trim->phase_rate_codes_per_second
                                  << " correction_hz="
                                  << requested_hz - active_output_hz_
                                  << " requested_hz=" << requested_hz
                                  << " rms_codes="
                                  << trim->rms_phase_codes
                                  << " direction_consistency="
                                  << trim->direction_consistency
                                  << " source=visual_phase_trend\n"
                                  << std::flush;
                        if (!send_output(requested_hz, phase, 102, 250))
                            return ReacquireTarget::None;
                        phase_history.clear();
                        unwrapped_phase = phase;
                        last_observed_phase = phase;
                        phase_history.push_back({
                            std::chrono::duration<double>(
                                Clock::now() - servo_origin).count(),
                            unwrapped_phase});
                    }
                }
            }
        }

        VisualLockUpdate update = lock_state.update(good, Clock::now());
        if (!update.event.empty()) {
            std::cout << update.event << " mode=" << static_cast<int>(mode)
                      << " input_hz=" << input_frequency_hz << "\n"
                      << std::flush;
        }
        if (update.lost) {
            // Shape degradation is a control error, not evidence that a
            // visually confirmed frequency changed. Keep the success output
            // asserted and continue local correction from the best phase.
            std::cout << "LOCK_ADJUST_CONTINUE reason=shape_degraded"
                      << " frequency_retained=" << frequency_confirmed
                      << " phase_code=0 led=steady\n" << std::flush;
            lock_state.reset(Clock::now());
        }
    }
    set_visual_lock(false);
    return ReacquireTarget::None;
}

int RuntimeController::run(const std::atomic_bool* stop_requested) {
    stop_requested_ = stop_requested;
    if (!uart_.open(config_.serial)) {
        std::cerr << "cannot open serial " << config_.serial << "\n";
        return 3;
    }
    if (!camera_.open(config_.camera)) {
        std::cerr << "cannot open camera " << config_.camera << "\n";
        return 5;
    }

    if (config_.abba_grid_roi) {
        calibrator_.set_roi(*config_.abba_grid_roi);
        std::cout << "ABBA_GRID_ROI roi=" << config_.abba_grid_roi->x << ','
                  << config_.abba_grid_roi->y << ','
                  << config_.abba_grid_roi->width << ','
                  << config_.abba_grid_roi->height << "\n" << std::flush;
    }

    FpgaResponse status;
    if (!poll_status(&status)) {
        std::cerr << "cannot read initial FPGA mode\n";
        return 6;
    }
    // Start grid acquisition as soon as the camera and FPGA mode are known.
    // When the process itself starts in an automatic mode, the previous run
    // may have left an arbitrary ellipse on screen. Blank Y first so that the
    // horizontal X trace provides the absolute grid anchor; otherwise an
    // integer-cell-shifted lattice can freeze before the normal auto-mode path
    // gets a chance to blank it. Manual modes remain read-only here.
    std::cout << "GRID_SEARCH_START\n" << std::flush;
    if (is_auto_mode(current_mode_)) {
        if (const auto calibration = read_live_calibration()) {
            active_output_hz_ = 100'000.0;
            send_output(active_output_hz_, 0, 0, 0);
        }
    }
    if (!collect_grid_with_exposure_sweep(kExposureGridSearchMs))
        std::cout << "GRID_SEARCH_CONTINUE\n" << std::flush;

    std::cout << "RUNTIME_START mode=" << static_cast<int>(current_mode_)
              << " name=" << mode_name(current_mode_) << "\n" << std::flush;

    uint8_t announced_mode = 0xff;
    auto last_calibration_poll = Clock::time_point{};
    auto last_grid_progress = Clock::now();
    while (!stopped()) {
        if (!poll_status(&status)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Keep looking while the grid is not found, regardless of the FPGA
        // mode. Once locked, GridCalibrator freezes the homography forever.
        if (!calibrator_.locked()) {
            collect_grid(kGridSearchSliceMs);
            if (!calibrator_.locked() &&
                Clock::now() - last_grid_progress >=
                    std::chrono::seconds(1)) {
                std::cout << "GRID_SEARCH_ACTIVE mode="
                          << static_cast<int>(current_mode_) << "\n"
                          << std::flush;
                last_grid_progress = Clock::now();
            }
        }

        if (current_mode_ != announced_mode) {
            std::cout << "MODE_CHANGE mode=" << static_cast<int>(current_mode_)
                      << " name=" << mode_name(current_mode_) << "\n"
                      << std::flush;
            announced_mode = current_mode_;
        }

        if (!is_auto_mode(current_mode_)) {
            if (current_mode_ == 2 &&
                Clock::now() - last_calibration_poll >
                    std::chrono::seconds(1)) {
                read_live_calibration();
                last_calibration_poll = Clock::now();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        const AutoMode mode = static_cast<AutoMode>(current_mode_);
        if (!read_live_calibration()) {
            std::cerr << "WAIT_CALIBRATION mode="
                      << static_cast<int>(current_mode_) << "\n" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        if (!calibrator_.locked()) {
            active_output_hz_ = 100'000.0;
            if (!send_output(active_output_hz_, 0, 0, 0) ||
                !collect_grid_with_exposure_sweep(kExposureGridSearchMs)) {
                std::cerr << "GRID_SEARCH_WAIT\n" << std::flush;
                continue;
            }
        }

        if (config_.abba_probe_frequency_hz > 0.0) {
            const bool probe_ok = run_abba_probe(
                mode, config_.abba_probe_frequency_hz);
            set_visual_lock(false);
            std::cout << "ABBA_PROBE_DONE ok=" << probe_ok << "\n"
                      << std::flush;
            return probe_ok ? 0 : 8;
        }

        if (config_.control_probe_frequency_hz > 0.0) {
            bool frequency_mismatch = false;
            bool frequency_confirmed = false;
            int frequency_evidence = 0;
            double output_frequency_trim_hz = 0.0;
            const auto shape = acquire_shape(
                mode, config_.control_probe_frequency_hz,
                &frequency_mismatch, &frequency_confirmed,
                &frequency_evidence, &output_frequency_trim_hz);
            std::cout << "CONTROL_PROBE_ACQUIRE hz=" << std::fixed
                      << std::setprecision(4)
                      << config_.control_probe_frequency_hz
                      << " shape=" << shape.has_value()
                      << " mismatch=" << frequency_mismatch
                      << " frequency_confirmed=" << frequency_confirmed
                      << " evidence=" << frequency_evidence
                      << " trim_hz=" << output_frequency_trim_hz << "\n"
                      << std::flush;
            if (!shape || !expected_mode(mode)) return 9;
            const ReacquireTarget target = maintain_lock(
                mode, config_.control_probe_frequency_hz, shape->phase,
                true);
            std::cout << "CONTROL_PROBE_DONE reacquire="
                      << static_cast<int>(target) << "\n" << std::flush;
            return 0;
        }

        std::optional<double> retained_frequency_hz;
        std::vector<double> fallback_frequency_hz;
        bool frequency_visually_confirmed = false;
        int frequency_shape_evidence = 0;
        double output_frequency_trim_hz = 0.0;
        int provisional_rephase_attempts = 0;
        int explicit_mismatch_failures = 0;
        int frequency_attempt_index = 0;
        bool startup_prior_enabled = true;
        bool frequency_prior_active = false;
        auto acquisition_started = Clock::now();
        while (!stopped() && expected_mode(mode)) {
            if (!retained_frequency_hz) {
                acquisition_started = Clock::now();
                ++frequency_attempt_index;
                const bool use_prior = startup_prior_enabled &&
                    use_startup_frequency_prior(frequency_attempt_index);
                if (use_prior) {
                    retained_frequency_hz =
                        startup_frequency_prior(mode);
                    frequency_prior_active = true;
                    fallback_frequency_hz.clear();
                    std::cout << "FREQUENCY_PRIOR mode="
                              << static_cast<int>(mode) << " hz="
                              << std::fixed << std::setprecision(1)
                              << *retained_frequency_hz
                              << " attempt=" << frequency_attempt_index
                              << " reason=startup_attempt_2_through_5\n"
                              << std::flush;
                } else {
                    const auto frequencies = acquire_frequency(mode);
                    if (!frequencies || !expected_mode(mode)) {
                        if (expected_mode(mode)) {
                            wait_frames(500, mode);
                        }
                        continue;
                    }
                    retained_frequency_hz = frequencies->primary_hz;
                    fallback_frequency_hz = frequencies->fallback_hz;
                    frequency_prior_active = false;
                }
                frequency_visually_confirmed = false;
                frequency_shape_evidence = 0;
                output_frequency_trim_hz = 0.0;
                provisional_rephase_attempts = 0;
                explicit_mismatch_failures = 0;
                std::cout << "FREQUENCY_CONFIRM_START hz="
                          << std::fixed << std::setprecision(1)
                          << *retained_frequency_hz
                          << " source="
                          << (frequency_prior_active ? "startup_prior" :
                                                       "visual_consensus")
                          << "\n" << std::flush;
            }

            std::optional<ShapeChoice> shape;
            bool frequency_mismatch = false;
            bool shape_confirmed_frequency = false;
            shape = acquire_shape(
                mode, *retained_frequency_hz, &frequency_mismatch,
                &shape_confirmed_frequency, &frequency_shape_evidence,
                &output_frequency_trim_hz);
            frequency_visually_confirmed = frequency_visually_confirmed ||
                                           shape_confirmed_frequency;
            // A runner-up is authorized only by a distinctive visual
            // frequency mismatch, not by an ordinary failed phase search.
            while (!shape && frequency_mismatch &&
                   !frequency_visually_confirmed &&
                   !fallback_frequency_hz.empty() &&
                   expected_mode(mode)) {
                const double fallback_hz = fallback_frequency_hz.front();
                fallback_frequency_hz.erase(fallback_frequency_hz.begin());
                std::cout << "FREQUENCY_FALLBACK from_hz="
                          << *retained_frequency_hz << " to_hz="
                          << fallback_hz
                          << " reason=visual_frequency_mismatch"
                             " reuse_phase_code=1\n"
                          << std::flush;
                retained_frequency_hz = fallback_hz;
                std::cout << "FREQUENCY_CONFIRM_START hz="
                          << std::fixed << std::setprecision(1)
                          << *retained_frequency_hz
                          << " source=visual_fallback\n" << std::flush;
                frequency_shape_evidence = 0;
                output_frequency_trim_hz = 0.0;
                provisional_rephase_attempts = 0;
                bool alternate_mismatch = false;
                bool alternate_confirmed_frequency = false;
                shape = acquire_shape(
                    mode, *retained_frequency_hz, &alternate_mismatch,
                    &alternate_confirmed_frequency,
                    &frequency_shape_evidence, &output_frequency_trim_hz);
                frequency_mismatch = alternate_mismatch;
                frequency_visually_confirmed =
                    alternate_confirmed_frequency;
            }
            if (!shape || !expected_mode(mode)) {
                if (!expected_mode(mode)) break;
                if (frequency_mismatch)
                    ++explicit_mismatch_failures;
                else
                    explicit_mismatch_failures = 0;

                const bool provisional_rephase =
                    !frequency_visually_confirmed && !frequency_mismatch &&
                    frequency_shape_evidence >= 2 &&
                    provisional_rephase_attempts < 1;
                const bool retain_frequency =
                    (frequency_visually_confirmed &&
                     (!frequency_mismatch || explicit_mismatch_failures < 2)) ||
                    provisional_rephase;
                if (retain_frequency) {
                    if (provisional_rephase) ++provisional_rephase_attempts;
                    std::cout << "REPHASE reason=shape_acquisition_failed"
                              << " frequency_hz=" << *retained_frequency_hz
                              << " frequency_retained=1 phase_code=0"
                              << " evidence=" << frequency_shape_evidence
                              << "/3\n"
                              << std::flush;
                    wait_frames(250, mode);
                    continue;
                }
                if (frequency_prior_active) {
                    std::cout << "FREQUENCY_PRIOR_REJECT mode="
                              << static_cast<int>(mode) << " hz="
                              << std::fixed << std::setprecision(1)
                              << *retained_frequency_hz << " reason="
                              << (frequency_mismatch
                                      ? "confirmed_frequency_mismatch"
                                      : "unconfirmed_frequency")
                              << " continue=startup_retry_policy\n"
                              << std::flush;
                }
                std::cout << "RETRY reason="
                          << (frequency_mismatch
                                  ? "confirmed_frequency_mismatch"
                                  : "unconfirmed_frequency")
                          << " phase_code=1\n" << std::flush;
                retained_frequency_hz.reset();
                fallback_frequency_hz.clear();
                frequency_visually_confirmed = false;
                frequency_shape_evidence = 0;
                output_frequency_trim_hz = 0.0;
                provisional_rephase_attempts = 0;
                explicit_mismatch_failures = 0;
                frequency_prior_active = false;
                continue;
            }
            // A successful target shape ends the startup-only prior policy.
            // Later visual frequency reacquisitions must remain unconstrained.
            startup_prior_enabled = false;
            frequency_prior_active = false;
            if (!frequency_visually_confirmed) {
                frequency_visually_confirmed = true;
                frequency_shape_evidence = 3;
                std::cout << "FREQUENCY_VISUAL_CONFIRMED hz="
                          << *retained_frequency_hz
                          << " reason=strict_target_shape\n"
                          << std::flush;
            }
            fallback_frequency_hz.clear();
            explicit_mismatch_failures = 0;
            const double elapsed = std::chrono::duration<double>(
                Clock::now() - acquisition_started).count();
            std::cout << "CONTROL_READY elapsed_s=" << std::fixed
                      << std::setprecision(3) << elapsed << "\n" << std::flush;
            const ReacquireTarget target = maintain_lock(
                mode, *retained_frequency_hz, shape->phase,
                frequency_visually_confirmed);
            if (target == ReacquireTarget::None) break;
            if (target == ReacquireTarget::Frequency) {
                retained_frequency_hz.reset();
                frequency_visually_confirmed = false;
                frequency_shape_evidence = 0;
                output_frequency_trim_hz = 0.0;
                provisional_rephase_attempts = 0;
                explicit_mismatch_failures = 0;
                frequency_prior_active = false;
            } else {
                std::cout << "REPHASE reason=lock_lost frequency_hz="
                          << *retained_frequency_hz
                          << " frequency_retained=1 phase_code=0\n"
                          << std::flush;
            }
        }
    }

    if (live_calibration_ && is_auto_mode(current_mode_))
        set_visual_lock(false);
    std::cout << "RUNTIME_STOP\n" << std::flush;
    return 0;
}

}  // namespace task5
