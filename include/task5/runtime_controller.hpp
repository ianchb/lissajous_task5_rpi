#pragma once

#include "task5/camera.hpp"
#include "task5/clock_calibration.hpp"
#include "task5/command_scheduler.hpp"
#include "task5/grid_calibrator.hpp"
#include "task5/lock_state.hpp"
#include "task5/phase_capture.hpp"
#include "task5/shape_observer.hpp"
#include "task5/trace_segmenter.hpp"
#include "task5/uart.hpp"
#include "task5/visual_frequency_servo.hpp"

#include <atomic>
#include <optional>
#include <string>
#include <vector>

namespace task5 {

struct RuntimeConfig {
    std::string camera = "/dev/video19";
    std::string serial = "/dev/ttyUSB0";
    int frequency_windows = 3;
    int frequency_window_ms = 1'000;
    int phase_code_settle_ms = 400;
    // The scope trace settles much faster than the old conservative 900 ms
    // delay. Keep a bounded delay before consuming fresh frames after each
    // phase command.
    int shape_settle_ms = 600;
    int shape_window_ms = 300;
    int hold_window_ms = 500;
    // Lower exposure preserves the fine phase-code texture under strong
    // ambient light. Grid search falls back upward for darker installations.
    std::vector<int> grid_exposures{40, 60, 80};
    double abba_probe_frequency_hz = 0.0;
    int abba_center_phase = 0;
    int abba_delta_phase = 12;
    int abba_settle_ms = 300;
    int abba_measure_ms = 240;
    int abba_rounds = 1;
    std::optional<cv::Rect> abba_grid_roi;
    double control_probe_frequency_hz = 0.0;
    bool control_probe_suppress_frequency_trim = false;
};

class RuntimeController {
public:
    explicit RuntimeController(RuntimeConfig config = {});
    int run(const std::atomic_bool* stop_requested = nullptr);

private:
    struct ShapeChoice {
        int phase = 0;
        ShapeMetrics metrics;
        bool valid = false;
    };

    struct ShapeWindowStats {
        int valid_frames = 0;
        int good_frames = 0;
        bool has_best_good = false;
        ShapeMetrics best_good;
        double best_good_score = -1.0e9;
    };

    struct PhaseServoChoice {
        int phase = 0;
        ShapeMetrics metrics;
        bool target_good = false;
        bool direction_confident = false;
    };

    struct FrequencyCandidates {
        double primary_hz = 0.0;
        std::optional<double> alternate_hz;
        std::vector<double> fallback_hz;
        int primary_support = 0;
        int valid_windows = 0;
    };

    enum class ReacquireTarget {
        None,
        Phase,
        Frequency,
    };

    bool stopped() const;
    bool poll_status(FpgaResponse* response = nullptr);
    bool expected_mode(AutoMode mode);
    bool read_frame(Frame* frame);
    bool send_output(double frequency_hz, int phase, int amplitude,
                     int display_settle_ms = 0);
    bool set_visual_lock(bool locked);
    bool collect_grid(int budget_ms);
    bool collect_grid_with_exposure_sweep(int budget_per_exposure_ms);
    bool wait_frames(int duration_ms, AutoMode mode);
    std::optional<ClockCalibration> read_live_calibration();
    std::optional<FrequencyCandidates> acquire_frequency(AutoMode mode);
    std::optional<ShapeMetrics> measure_shape(AutoMode mode, int phase,
                                               int settle_ms, int measure_ms,
                                               bool send_command = true,
                                               std::vector<PhaseTrackSample>*
                                                   phase_track = nullptr,
                                               Clock::time_point track_origin = {},
                                               ShapeWindowStats* window_stats =
                                                   nullptr,
                                               bool stop_on_frequency_mismatch =
                                                   false);
    bool run_abba_probe(AutoMode mode, double input_frequency_hz);
    std::optional<ShapeChoice> acquire_shape(AutoMode mode,
                                              double input_frequency_hz,
                                              bool* frequency_mismatch,
                                              bool* frequency_confirmed,
                                              int* frequency_evidence_count,
                                              double* output_frequency_trim_hz);
    ReacquireTarget maintain_lock(AutoMode mode, double input_frequency_hz,
                                  int phase, bool frequency_confirmed);
    std::optional<ShapeChoice> refine_phase_locally(
        AutoMode mode, int starting_phase, const ShapeMetrics& starting_metrics,
        int maximum_rounds, bool require_two_fresh_windows,
        bool* filled_rectangle = nullptr, int maximum_phase_step = 8,
        int candidate_settle_ms = 350, int candidate_measure_ms = -1);
    std::optional<PhaseServoChoice> probe_phase_abba(
        AutoMode mode, int center_phase,
        const ShapeMetrics& center_metrics,
        bool* filled_rectangle = nullptr);
    std::optional<PhaseServoChoice> probe_circle_centered(
        int center_phase, const ShapeMetrics& center_metrics,
        bool* filled_rectangle = nullptr);
    static ShapeMetrics median_metrics(std::vector<ShapeMetrics> samples);

    RuntimeConfig config_;
    CameraSource camera_;
    UartTransport uart_;
    DdsCommandScheduler scheduler_{&uart_};
    GridCalibrator calibrator_;
    TraceSegmenter segmenter_;
    ShapeObserver shape_observer_;
    const std::atomic_bool* stop_requested_ = nullptr;
    uint8_t current_mode_ = 0xff;
    Clock::time_point last_status_poll_{};
    Clock::time_point last_grid_analysis_{};
    std::optional<ClockCalibration> live_calibration_;
    std::optional<CommandTicket> active_ticket_;
    std::optional<double> pending_visual_error_hz_;
    std::optional<double> visual_trim_context_hz_;
    uint8_t visual_trim_context_mode_ = 0xff;
    std::optional<ShapeChoice> phase_search_seed_;
    std::optional<double> phase_search_context_hz_;
    uint8_t phase_search_context_mode_ = 0xff;
    uint64_t last_frame_sequence_ = 0;
    double active_output_hz_ = 0.0;
};

}  // namespace task5
