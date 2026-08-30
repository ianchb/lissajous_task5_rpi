#include "task5/frequency_observer.hpp"
#include "task5/candidate_bank_matcher.hpp"
#include "task5/clock_calibration.hpp"
#include "task5/controller_state.hpp"
#include "task5/pll.hpp"
#include "task5/phase_code_matcher.hpp"
#include "task5/sine_matcher.hpp"
#include "task5/shape_observer.hpp"
#include "task5/uart.hpp"
#include "task5/trace_segmenter.hpp"
#include "task5/visual_frequency_servo.hpp"
#include "task5/command_scheduler.hpp"
#include "task5/lock_state.hpp"
#include "task5/phase_capture.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cv::Mat make_ramp_trace(double frequency_hz, double scan_duration_s) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    std::vector<cv::Point> curve;
    for (int i = 0; i <= 1400; ++i) {
        const double u = static_cast<double>(i) / 1400.0;
        const double t = u * scan_duration_s;
        const int x = static_cast<int>(std::lround(320 + 245 *
                                                    std::sin(2 * kPi * frequency_hz * t + 0.3)));
        const int y = static_cast<int>(std::lround(90 + 300 * u));
        curve.emplace_back(x, y);
    }
    cv::polylines(image, curve, false, cv::Scalar(0, 220, 255), 3,
                  cv::LINE_AA);
    return image;
}

cv::Mat make_phase_code_trace(double frequency_hz,
                              const task5::PhaseCodeSpec& code) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    const double phase = 0.37;
    for (size_t band = 0; band < code.band_centers.size(); ++band) {
        std::vector<cv::Point> curve;
        const double center = code.band_centers[band] * (image.rows - 1);
        const double height = code.band_height * (image.rows - 1);
        const double start = code.divisors[band] == 1
            ? 0.0
            : code.frame_period_s / code.divisors[band];
        for (int i = 0; i <= 160; ++i) {
            const double u = static_cast<double>(i) / 160.0;
            const double t = start + u * code.marker_duration_s;
            const int x = static_cast<int>(std::lround(
                code.center_x + code.amplitude_x *
                    std::sin(2 * kPi * frequency_hz * t + phase)));
            const int y = static_cast<int>(std::lround(
                center - 0.5 * height + u * height));
            curve.emplace_back(x, y);
        }
        cv::polylines(image, curve, false, cv::Scalar(0, 220, 255), 2,
                      cv::LINE_8);
    }
    return image;
}

cv::Mat make_quantized_phase_code_trace(
    double frequency_hz, const task5::PhaseCodeSpec& code,
    double visual_scale = 1.0, double visual_offset = 0.0,
    int visible_band = -1) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    constexpr int amplitude = 102;
    constexpr std::array<int, 4> starts{89, 38, -12, -63};
    const int span = amplitude / 3;
    const double phase = 0.37;
    for (std::size_t band = 0; band < starts.size(); ++band) {
        if (visible_band >= 0 &&
            static_cast<int>(band) != visible_band)
            continue;
        const double start_s = code.divisors[band] == 1
            ? 0.0 : code.frame_period_s / code.divisors[band];
        for (int sample = 0; sample < 4096; ++sample) {
            const double u = (sample + 0.5) / 4096.0;
            const int fraction = sample * 256 / 4096;
            const int offset = span * fraction >> 8;
            const int physical_value = -starts[band] + offset;
            const double nominal_y = static_cast<double>(
                physical_value + amplitude) / (2 * amplitude);
            const double y_fraction = visual_offset +
                                      visual_scale * nominal_y;
            const int x = static_cast<int>(std::lround(
                320.0 + 245.0 * std::sin(2 * kPi * frequency_hz *
                    (start_s + u * code.marker_duration_s) + phase)));
            const int y = static_cast<int>(std::lround(
                y_fraction * (image.rows - 1)));
            cv::circle(image, {x, y}, 1, cv::Scalar(0, 220, 255),
                       cv::FILLED, cv::LINE_8);
        }
    }
    return image;
}

cv::Mat make_multirate_trace(double frequency_hz,
                             const task5::MultiRateSpec& spec) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    double start_s = 0.0;
    const double phase = 0.41;
    for (size_t band = 0; band < spec.band_centers.size(); ++band) {
        std::vector<cv::Point> curve;
        const int points = std::max(180, spec.samples_per_band[band] * 2);
        const double top = (spec.band_centers[band] -
                            0.5 * spec.band_heights[band]) * (image.rows - 1);
        const double height = spec.band_heights[band] * (image.rows - 1);
        for (int index = 0; index <= points; ++index) {
            const double u = static_cast<double>(index) / points;
            const double time_s = start_s + u * spec.durations_s[band];
            const int x = static_cast<int>(std::lround(
                320.0 + 245.0 * std::sin(
                    2.0 * kPi * frequency_hz * time_s + phase)));
            const int y = static_cast<int>(std::lround(top + u * height));
            curve.emplace_back(x, y);
        }
        cv::polylines(image, curve, false, cv::Scalar(0, 220, 255), 2,
                      cv::LINE_8);
        start_s += spec.durations_s[band];
    }
    return image;
}

void test_frequency_observer() {
    const double expected = 12'300.0;
    const double duration = 1.0e-3;
    task5::TraceSegmenter segmenter({12, 48, 35, 55, 145, 105, 0, 10});
    const cv::Mat image = make_ramp_trace(expected, duration);
    const task5::TraceObservation observation =
        segmenter.process(image, 0, task5::Clock::now());
    task5::FrequencyObserver observer;
    task5::ProbeSpec probe;
    probe.scan_duration_s = duration;
    const task5::FrequencyEstimate estimate =
        observer.observe_ramp(observation, probe);
    std::cerr << "synthetic estimate=" << estimate.frequency_hz
              << " score=" << estimate.score
              << " runner=" << estimate.runner_up_hz
              << " margin=" << estimate.margin
              << " samples=" << estimate.valid_samples << "\n";
    require(estimate.valid, "ramp estimate invalid");
    // This is deliberately a coarse range observation. Trace thickness alone
    // moves the apparent scan duration enough to shift a few 100 Hz bins.
    require(std::abs(estimate.frequency_hz - expected) <= 300.0,
            "ramp estimate outside coarse range");
    require(estimate.margin > 0.0, "ramp candidate has no runner-up margin");
}

void test_phase_code_observer() {
    task5::PhaseCodeSpec code;
    code.center_x = 320.0;
    code.amplitude_x = 245.0;
    code.phase_trials = 256;
    task5::TraceSegmenter segmenter({12, 48, 35, 55, 145, 105, 0, 4});
    task5::FrequencyObserver observer;
    for (double expected : {1'000.0, 1'100.0, 12'300.0, 50'100.0,
                            99'900.0, 100'000.0}) {
        const cv::Mat image = make_phase_code_trace(expected, code);
        const task5::TraceObservation observation =
            segmenter.process(image, 0, task5::Clock::now());
        const task5::FrequencyEstimate estimate =
            observer.observe_phase_code(observation, code);
        std::cerr << "phase-code expected=" << expected
                  << " estimate=" << estimate.frequency_hz
                  << " score=" << estimate.score
                  << " margin=" << estimate.margin << "\n";
        require(estimate.valid, "phase-code estimate invalid");
        require(std::abs(estimate.frequency_hz - expected) < 1.0,
                "phase-code estimate is not on the legal bin");
        require(estimate.margin > 1.0e-4,
                "phase-code candidate has no separation");
    }
}

void test_quantized_phase_code_matcher() {
    task5::PhaseCodeSpec code;
    for (double expected : {1'000.0, 1'100.0, 8'000.0, 12'300.0, 16'000.0,
                            55'500.0,
                            68'400.0, 73'100.0, 86'800.0, 92'700.0,
                            99'900.0, 100'000.0}) {
        const cv::Mat image = make_quantized_phase_code_trace(expected, code);
        task5::TraceObservation observation;
        observation.normalized_bgr = image;
        task5::PhaseCodeMatcher matcher(code);
        for (int frame = 0; frame < 30; ++frame)
            require(matcher.add(observation),
                    "quantized phase-code frame rejected");
        const task5::FrequencyEstimate estimate = matcher.estimate();
        std::cerr << "quantized phase-code expected=" << expected
                  << " estimate=" << estimate.frequency_hz
                  << " runner=" << estimate.runner_up_hz
                  << " margin=" << estimate.margin
                  << " spectral=" << estimate.candidate_first_bin * 100
                  << " spectral_ratio="
                  << (estimate.candidate_scores.empty() ? 0.0f :
                      estimate.candidate_scores.front()) << "\n";
        require(estimate.valid, "quantized phase-code estimate invalid");
        require(std::abs(estimate.frequency_hz - expected) < 1.0,
                "quantized phase-code estimate is wrong");
    require(estimate.margin > 1.0e-4,
                "quantized phase-code candidate has no separation");
    }

    // A 500 us/div acquisition can expose only part of the coherent code in
    // each camera frame. The matcher must combine independently visible bands
    // across time instead of requiring all four in one image.
    {
        constexpr double partial_expected = 82'700.0;
        task5::PhaseCodeMatcher partial_matcher(code);
        for (int frame = 0; frame < 32; ++frame) {
            task5::TraceObservation partial;
            partial.normalized_bgr = make_quantized_phase_code_trace(
                partial_expected, code, 1.0, 0.0, frame % 4);
            require(partial_matcher.add(partial),
                    "partial-band phase-code frame rejected");
        }
        const task5::FrequencyEstimate partial_estimate =
            partial_matcher.estimate();
        require(partial_estimate.valid &&
                    std::abs(partial_estimate.frequency_hz -
                             partial_expected) < 1.0,
                "cross-frame partial-band phase code is wrong");
    }

    constexpr double expected = 50'000.0;
    constexpr double visual_scale = 0.944;
    constexpr double visual_offset = 0.025;
    const cv::Mat compressed = make_quantized_phase_code_trace(
        expected, code, visual_scale, visual_offset);
    task5::TraceObservation observation;
    observation.normalized_bgr = compressed;
    task5::PhaseCodeMatcher matcher(code);
    for (int frame = 0; frame < 30; ++frame)
        require(matcher.add(observation),
                "compressed phase-code frame rejected");
    const task5::FrequencyEstimate estimate = matcher.estimate();
    std::cerr << "compressed phase-code expected=" << expected
              << " estimate=" << estimate.frequency_hz
              << " visual_scale=" << estimate.visual_scale
              << " visual_offset=" << estimate.visual_offset << "\n";
    require(estimate.valid, "compressed phase-code estimate invalid");
    require(std::abs(estimate.frequency_hz - expected) < 1.0,
            "compressed phase-code estimate is wrong");
    require(estimate.visual_geometry_valid,
            "compressed phase-code geometry was not applied");
    require(std::abs(estimate.visual_scale - visual_scale) < 0.02,
            "compressed phase-code scale fit is inaccurate");
    require(estimate.geometry_alternate_hz > 0.0,
            "unscaled geometry alternate was not generated");
    require(!estimate.geometry_alternate_scaled,
            "alternate should use the unscaled model when scaled geometry "
            "wins by default");
}

void test_multirate_observer() {
    task5::MultiRateSpec spec;
    task5::TraceSegmenter segmenter({12, 48, 35, 55, 145, 105, 0, 4});
    task5::FrequencyObserver observer;
    for (double expected : {1'000.0, 3'000.0, 12'300.0, 62'300.0,
                            100'000.0}) {
        const cv::Mat image = make_multirate_trace(expected, spec);
        const task5::TraceObservation observation =
            segmenter.process(image, 0, task5::Clock::now());
        const task5::FrequencyEstimate estimate =
            observer.observe_multirate(observation, spec);
        std::cerr << "multirate expected=" << expected
                  << " estimate=" << estimate.frequency_hz
                  << " score=" << estimate.score
                  << " margin=" << estimate.margin
                  << " cycles=" << estimate.observed_cycles << "\n";
        require(estimate.valid, "multirate estimate invalid");
        require(std::abs(estimate.frequency_hz - expected) <= 1'000.0,
                "multirate estimate outside coarse candidate window");
    }
}

void test_phase_slope() {
    task5::PhaseSlopeEstimator estimator(24);
    const auto start = task5::Clock::now();
    for (int i = 0; i < 16; ++i) {
        const double t = 0.02 * i;
        // Once the CRT bin is correct, the remaining timebase/DDS residual is
        // below the camera Nyquist rate. A 100 Hz wrong bin is rejected by the
        // phase code instead of being aliased by this estimator.
        const double phase = 2 * kPi * 4.0 * t + 0.25;
        estimator.add({start + std::chrono::duration_cast<task5::Clock::duration>(
                                  std::chrono::duration<double>(t)),
                       phase, 0.95});
    }
    const task5::PhaseRateEstimate estimate = estimator.estimate();
    require(estimate.valid, "phase slope estimate invalid");
    require(std::abs(estimate.delta_frequency_hz - 4.0) < 1.0e-6,
            "phase slope estimate is wrong");
}

void test_pll_is_bounded() {
    task5::PhasePll pll;
    auto output = pll.output();
    for (int i = 0; i < 200; ++i) {
        output = pll.update(0.10, 0.9, 1.0 / 30.0);
        require(std::abs(output.frequency_correction_hz) <= 20.0,
                "PLL frequency correction escaped its bound");
        require(std::abs(output.phase_correction_rad) <= kPi,
                "PLL phase correction escaped its bound");
    }
    require(output.locked, "PLL did not enter its qualified lock state");
}

void test_controller_deadline_and_reacquire() {
    task5::ControllerStateMachine machine;
    auto now = task5::Clock::now();
    auto output = machine.update({true, false, false, 0.0, false, false, false}, now);
    require(output.state == task5::ControllerState::GridReady,
            "controller did not accept grid");
    output = machine.update({true, true, false, 0.0, false, false, false}, now);
    require(output.state == task5::ControllerState::FrequencyCoarse,
            "controller did not enter coarse acquisition");
    output = machine.update({true, true, false, 0.0, false, false, false}, now);
    require(output.state == task5::ControllerState::FrequencyCode,
            "controller did not enter code acquisition");
    for (int i = 0; i < 3; ++i) {
        output = machine.update({true, true, true, 0.2, false, false, false}, now);
    }
    require(output.state == task5::ControllerState::CandidateConfirm,
            "candidate was not confirmed");
    output = machine.update({true, true, true, 0.2, false, false, true}, now);
    require(output.event == "REACQUIRE_SOURCE_CHANGE" &&
                output.request_frequency_probe,
            "source change did not force frequency reacquisition");

    task5::ControllerStateMachine timeout_machine;
    output = timeout_machine.update(
        {}, now + std::chrono::milliseconds(10'001));
    require(output.failed && output.event == "TIMEOUT_TOTAL",
            "controller did not enforce the 10 second budget");
}

void test_uart_packet() {
    const task5::DdsCommand command{0x12345678u, 0x9a, 0x66, 0x82};
    const auto packet = task5::UartTransport::encode_set_dds(command);
    require(packet[0] == 0xa5 && packet[1] == 0x5a && packet[2] == 0x01,
            "UART header is wrong");
    require(packet[3] == 0x78 && packet[4] == 0x56 && packet[5] == 0x34 &&
                packet[6] == 0x12,
            "UART FTW is not little endian");
    uint8_t checksum = 0;
    for (size_t i = 0; i < 10; ++i) checksum ^= packet[i];
    require(checksum == packet[10], "UART checksum is wrong");
    const auto request = task5::UartTransport::encode_request(0x03);
    checksum = 0;
    for (size_t i = 0; i < 10; ++i) checksum ^= request[i];
    require(request[2] == 0x03 && checksum == request[10],
            "UART request packet is wrong");

    const task5::FineDdsCommand fine{0x123456789aULL, 0xbc, 0x66};
    const auto fine_packet = task5::UartTransport::encode_set_fine_dds(fine);
    require(fine_packet[2] == 0x05 && fine_packet[3] == 0x9a &&
                fine_packet[4] == 0x78 && fine_packet[5] == 0x56 &&
                fine_packet[6] == 0x34 && fine_packet[7] == 0x12 &&
                fine_packet[8] == 0xbc && fine_packet[9] == 0x66,
            "fine DDS packet field mapping is wrong");
    checksum = 0;
    for (size_t i = 0; i < 10; ++i) checksum ^= fine_packet[i];
    require(checksum == fine_packet[10], "fine DDS checksum is wrong");
    const auto lock_packet = task5::UartTransport::encode_set_locked(true);
    require(lock_packet[2] == 0x06 && lock_packet[3] == 1,
            "independent lock packet is wrong");

    constexpr double clock_hz = 19'999'854.335;
    constexpr double requested_hz = 66'400.0;
    const uint64_t ftw_q8 = task5::DdsCommandScheduler::frequency_to_ftw_q8(
        requested_hz, clock_hz);
    const double realized_hz = task5::DdsCommandScheduler::ftw_q8_to_frequency(
        ftw_q8, clock_hz);
    require(std::abs(realized_hz - requested_hz) < 0.00002,
            "Q8 FTW resolution is insufficient");
}

void test_visual_lock_state() {
    task5::VisualLockStateMachine state({std::chrono::milliseconds(5'000), 3});
    const auto start = task5::Clock::now();
    state.reset(start);
    auto update = state.update(true, start);
    require(update.event == "STABLE_START" && !update.stable_5s,
            "visual lock did not start qualification");
    update = state.update(true, start + std::chrono::milliseconds(5'001));
    require(update.stable_5s && update.event == "STABLE_5S",
            "visual lock did not qualify five seconds");
    update = state.update(false, start + std::chrono::milliseconds(5'500));
    require(!update.lost && update.event == "STABLE_RESET",
            "one bad window incorrectly caused lock loss");
    state.update(false, start + std::chrono::milliseconds(6'000));
    update = state.update(false, start + std::chrono::milliseconds(6'500));
    require(update.lost && update.event == "LOCK_LOST",
            "confirmed visual failure did not lose lock");

    task5::VisualLockStateMachine protected_state({
        std::chrono::milliseconds(5'000), 3,
        std::chrono::milliseconds(5'000)});
    protected_state.reset(start);
    protected_state.update(false, start + std::chrono::milliseconds(500));
    protected_state.update(false, start + std::chrono::milliseconds(1'000));
    update = protected_state.update(
        false, start + std::chrono::milliseconds(4'999));
    require(!update.lost,
            "bad windows broke the minimum five-second lock hold");
    protected_state.update(false,
                           start + std::chrono::milliseconds(5'001));
    protected_state.update(false,
                           start + std::chrono::milliseconds(5'501));
    update = protected_state.update(
        false, start + std::chrono::milliseconds(6'001));
    require(update.lost && update.event == "LOCK_LOST",
            "visual failure was not counted after the minimum lock hold");
}

void test_quadrature_phase_capture() {
    constexpr double delta = 0.42;
    const double response0 = std::cos(delta);
    const double response64 = -std::sin(delta);
    const auto line = task5::solve_quadrature_phase(
        task5::AutoMode::Line, response0, response64);
    const int expected_line = static_cast<int>(std::lround(
        (kPi - delta) * 256.0 / (2.0 * kPi))) & 0xff;
    require(line.has_value() &&
                std::abs(task5::signed_phase_delta(
                    line->phase, expected_line)) <= 1,
            "quadrature line phase has the wrong sign");

    const auto circle = task5::solve_quadrature_phase(
        task5::AutoMode::Circle, response0, response64);
    const int expected_circle = static_cast<int>(std::lround(
        (0.5 * kPi - delta) * 256.0 / (2.0 * kPi))) & 0xff;
    require(circle.has_value() &&
                std::abs(task5::signed_phase_delta(
                    circle->phase, expected_circle)) <= 1,
            "quadrature circle phase has the wrong sign");

    constexpr double feature_phase = -0.31;
    const double feature0 = std::cos(feature_phase);
    const double feature64 = std::sin(feature_phase);
    const auto figure8 = task5::solve_quadrature_phase(
        task5::AutoMode::FigureEight, feature0, feature64);
    require(figure8.has_value(),
            "quadrature figure-eight phase was rejected");
    const double q = 2.0 * kPi * figure8->phase / 256.0;
    require(std::abs(feature0 * std::cos(q) +
                     feature64 * std::sin(q)) < 0.03,
            "quadrature figure-eight solution is not a feature zero");
}

void test_clock_calibration() {
    task5::FpgaResponse response;
    response.flags = 0x80;
    response.payload = 2'560'034'016u;
    response.phase = 0x8f;
    response.amplitude = 0xc1;
    response.command = 0x03;
    response.checksum_valid = true;
    const auto calibration = task5::derive_clock_calibration(response);
    require(calibration.has_value(),
            "valid live clock calibration was rejected");
    require(std::abs(calibration->effective_dds_clock_hz -
                     19'999'862.12) < 0.1,
            "live clock calibration value is wrong");

    response.flags = 0;
    require(!task5::derive_clock_calibration(response),
            "invalid live clock calibration was accepted");
}

void test_sine_matcher() {
    task5::SineMatcher matcher;
    task5::TraceObservation thin;
    thin.mask = cv::Mat::zeros(480, 640, CV_8U);
    cv::ellipse(thin.mask, {320, 240}, {230, 150}, 0.0, 0.0, 360.0,
                cv::Scalar(255), 3, cv::LINE_AA);
    const task5::SineMatchScore thin_score = matcher.score(thin);

    task5::TraceObservation filled;
    filled.mask = cv::Mat::zeros(480, 640, CV_8U);
    cv::ellipse(filled.mask, {320, 240}, {230, 150}, 0.0, 0.0, 360.0,
                cv::Scalar(255), cv::FILLED, cv::LINE_AA);
    const task5::SineMatchScore filled_score = matcher.score(filled);
    require(thin_score.valid && filled_score.valid,
            "sine matcher rejected synthetic traces");
    require(thin_score.score < filled_score.score * 0.35,
            "sine matcher did not separate thin and filled traces");
}

void test_candidate_bank_matcher() {
    task5::TraceObservation observation;
    observation.mask = cv::Mat::zeros(480, 640, CV_8U);
    for (int band = 0; band < 16; ++band) {
        const int top = band * 30;
        const int center = top + 15;
        if (band == 11) {
            cv::ellipse(observation.mask, {320, center}, {230, 7}, 0.0,
                        0.0, 360.0, cv::Scalar(255), 2, cv::LINE_8);
        } else {
            cv::rectangle(observation.mask, {90, top + 4},
                          {550, top + 26}, cv::Scalar(255), cv::FILLED);
        }
    }
    task5::CandidateBankMatcher matcher;
    const task5::CandidateBankResult result = matcher.score(observation);
    require(result.valid && result.best_band == 11,
            "candidate bank did not select the thin band");
    require(result.margin > 0.1,
            "candidate bank has insufficient synthetic separation");
}

void test_shape_observer() {
    task5::ShapeObserver observer;

    task5::TraceObservation line;
    line.mask = cv::Mat::zeros(480, 640, CV_8U);
    cv::line(line.mask, {0, 0}, {639, 479}, cv::Scalar(255), 5,
             cv::LINE_AA);
    const task5::ShapeMetrics line_metrics = observer.analyze(line);
    require(line_metrics.valid && line_metrics.thinness < 0.03 &&
                line_metrics.correlation > 0.99,
            "shape observer rejected a synthetic line");

    task5::TraceObservation compact_locked_line;
    compact_locked_line.mask = cv::Mat::zeros(480, 640, CV_8U);
    cv::line(compact_locked_line.mask, {160, 369}, {560, 111},
             cv::Scalar(255), 5, cv::LINE_AA);
    const task5::ShapeMetrics compact_line_metrics =
        observer.analyze(compact_locked_line);
    require(compact_line_metrics.span_div.x >= 4.5 &&
                compact_line_metrics.span_div.y >= 3.8 &&
                compact_line_metrics.span_div.y < 7.75,
            "compact line fixture does not exercise non-full-grid extent");
    require(observer.shape_ok(task5::AutoMode::Line,
                              compact_line_metrics),
            "a compact high-quality locked line was rejected");
    require(observer.frequency_shape_coherent(
                task5::AutoMode::Line, compact_line_metrics),
            "a compact bounded line did not confirm frequency coherence");

    task5::ShapeMetrics camera_full_grid_line;
    camera_full_grid_line.valid = true;
    camera_full_grid_line.coverage = 0.10;
    camera_full_grid_line.span_div = {7.95, 7.93};
    camera_full_grid_line.correlation = -0.9699;
    camera_full_grid_line.thinness = 0.1181;
    camera_full_grid_line.minor_rms_div = 0.3888;
    require(observer.shape_ok(task5::AutoMode::Line,
                              camera_full_grid_line),
            "a crisp camera-thick full-grid line was rejected");
    require(!observer.phase_servo_needed(
                task5::AutoMode::Line, camera_full_grid_line),
            "a crisp locked line unnecessarily started the phase servo");

    task5::ShapeMetrics camera_ellipse = camera_full_grid_line;
    camera_ellipse.correlation = -0.8590;
    camera_ellipse.thinness = 0.2602;
    camera_ellipse.minor_rms_div = 0.8647;
    require(!observer.shape_ok(task5::AutoMode::Line, camera_ellipse),
            "a camera-observed ellipse was accepted as a locked line");
    require(observer.phase_servo_needed(
                task5::AutoMode::Line, camera_ellipse),
            "an emerging line ellipse did not start the phase servo");
    task5::TraceObservation short_line;
    short_line.mask = cv::Mat::zeros(480, 640, CV_8U);
    cv::line(short_line.mask, {240, 300}, {400, 180}, cv::Scalar(255), 5,
             cv::LINE_AA);
    const task5::ShapeMetrics short_line_metrics = observer.analyze(short_line);
    require(!observer.shape_ok(task5::AutoMode::Line, short_line_metrics),
            "a short trace fragment was accepted as a locked line");

    task5::TraceObservation circle;
    circle.mask = cv::Mat::zeros(480, 640, CV_8U);
    cv::ellipse(circle.mask, {320, 240}, {319, 239}, 0.0, 0.0, 360.0,
                cv::Scalar(255), 5, cv::LINE_AA);
    const task5::ShapeMetrics circle_metrics = observer.analyze(circle);
    require(circle_metrics.valid && circle_metrics.radial_cv < 0.03 &&
                std::abs(circle_metrics.correlation) < 0.03,
            "shape observer rejected a synthetic circle");
    require(observer.shape_ok(task5::AutoMode::Circle, circle_metrics),
            "circle acceptance thresholds reject a true circle");
    require(!observer.phase_servo_needed(
                task5::AutoMode::Circle, circle_metrics),
            "a true circle unnecessarily started the phase servo");
    require(observer.frequency_shape_coherent(
                task5::AutoMode::Circle, circle_metrics),
            "a bounded circle did not confirm frequency coherence");
    require(observer.frequency_shape_coherent(
                task5::AutoMode::Line, line_metrics),
            "a bounded line did not confirm frequency coherence");

    task5::ShapeMetrics harmonic_peak;
    harmonic_peak.valid = true;
    harmonic_peak.coverage = 0.20;
    harmonic_peak.correlation = -0.0062;
    harmonic_peak.thinness = 0.6408;
    harmonic_peak.radial_cv = 0.2122;
    harmonic_peak.symmetry_x = 0.1387;
    harmonic_peak.symmetry_y = 0.9996;
    harmonic_peak.crossing_fill = 0.0;
    harmonic_peak.phase_feature = 0.1562;
    harmonic_peak.span_div = {8.0, 7.9667};
    require(observer.frequency_mismatch_harmonic(
                task5::AutoMode::Circle, harmonic_peak),
            "8-to-16 kHz harmonic peak was not rejected");
    require(!observer.shape_ok(task5::AutoMode::Circle, harmonic_peak),
            "harmonic peak was accepted as a circle");

    task5::TraceObservation figure_eight;
    figure_eight.mask = cv::Mat::zeros(480, 640, CV_8U);
    std::vector<cv::Point> curve;
    for (int index = 0; index <= 1000; ++index) {
        const double phase = 2.0 * 3.14159265358979323846 * index / 1000.0;
        curve.emplace_back(
            static_cast<int>(std::lround(320.0 + 300.0 * std::sin(phase))),
            static_cast<int>(std::lround(240.0 + 237.0 * std::sin(2.0 * phase))));
    }
    cv::polylines(figure_eight.mask, curve, false, cv::Scalar(255), 5,
                  cv::LINE_AA);
    const task5::ShapeMetrics figure_eight_metrics =
        observer.analyze(figure_eight);
    require(figure_eight_metrics.valid &&
                figure_eight_metrics.symmetry_x > 0.85 &&
                figure_eight_metrics.symmetry_y > 0.85 &&
                figure_eight_metrics.crossing_fill > 0.05,
            "shape observer rejected a synthetic figure eight");
    require(observer.frequency_shape_coherent(
                task5::AutoMode::FigureEight, figure_eight_metrics),
            "a bounded figure eight did not retain visual frequency");
    require(std::abs(figure_eight_metrics.crossing_offset_y_div) < 0.10,
            "centered figure-eight crossing is not at the grid centre");
    require(!observer.phase_servo_needed(
                task5::AutoMode::FigureEight, figure_eight_metrics),
            "a centered figure eight unnecessarily started the phase servo");
    task5::ShapeMetrics shifted_crossing = figure_eight_metrics;
    shifted_crossing.crossing_offset_y_div = 0.45;
    require(!observer.shape_ok(task5::AutoMode::FigureEight,
                               shifted_crossing),
            "off-centre figure-eight crossing was accepted");
    require(observer.search_score(task5::AutoMode::FigureEight,
                                  shifted_crossing) <
                observer.search_score(task5::AutoMode::FigureEight,
                                      figure_eight_metrics) - 0.5,
            "figure-eight search does not penalize crossing displacement");

    task5::TraceObservation filled;
    filled.mask = cv::Mat(480, 640, CV_8U, cv::Scalar(255));
    const task5::ShapeMetrics filled_metrics = observer.analyze(filled);
    require(observer.frequency_mismatch_fill(filled_metrics),
            "filled frequency-mismatch rectangle was not detected");
    task5::ShapeMetrics persistence_rectangle = filled_metrics;
    persistence_rectangle.coverage = 0.48;
    persistence_rectangle.span_div = {8.0, 4.85};
    persistence_rectangle.symmetry_x = 0.8565;
    persistence_rectangle.symmetry_y = 0.6640;
    persistence_rectangle.crossing_fill = 1.0;
    require(observer.frequency_mismatch_fill(persistence_rectangle),
            "asymmetric filled persistence was not rejected");
    require(!observer.frequency_shape_coherent(
                task5::AutoMode::Line, filled_metrics),
            "a filled rectangle incorrectly confirmed frequency coherence");
    require(!observer.frequency_mismatch_fill(circle_metrics),
            "circle was mistaken for a frequency-mismatch rectangle");
}

void test_visual_frequency_servo() {
    const std::vector<task5::VisualPhaseSample> positive{
        {0.0, 0}, {2.0, 2}, {4.0, 4}};
    const auto positive_trim = task5::estimate_visual_frequency_trim(positive);
    require(positive_trim.has_value(),
            "visual frequency servo rejected a valid phase slope");
    require(std::abs(positive_trim->phase_rate_codes_per_second - 1.0) <
                1.0e-9 &&
            std::abs(positive_trim->output_correction_hz - 1.0 / 256.0) <
                1.0e-9,
            "visual frequency servo used the wrong sign or scale");

    const std::vector<task5::VisualPhaseSample> saturated{
        {0.0, 0}, {2.0, -20}};
    const auto saturated_trim = task5::estimate_visual_frequency_trim(saturated);
    require(saturated_trim.has_value() &&
                std::abs(saturated_trim->output_correction_hz + 0.025) <
                    1.0e-12,
            "visual frequency servo correction was not bounded");
    require(task5::signed_phase_delta(2, 254) == 4 &&
                task5::signed_phase_delta(254, 2) == -4,
            "phase code unwrap failed across zero");

    const std::vector<task5::VisualPhaseSample> inconsistent{
        {0.0, 0}, {0.5, 5}, {1.0, 1}, {1.5, 7}, {2.0, 2}};
    require(!task5::estimate_visual_frequency_trim(
                 inconsistent, 1.0, 2, 0.010),
            "inconsistent phase-search directions produced a DDS trim");

    std::vector<task5::PhaseTrackSample> track;
    constexpr double frequency_error = 0.037;
    constexpr double relative_phase = -0.42;
    for (int index = 0; index < 20; ++index) {
        const double time = 0.12 * index;
        const double command_phase = (index / 5) * 0.5 * kPi;
        track.push_back({time, command_phase,
            std::cos(2.0 * kPi * frequency_error * time +
                     command_phase + relative_phase)});
    }
    const auto track_fit = task5::fit_phase_track(track, 2.5);
    require(track_fit.has_value() &&
                std::abs(track_fit->frequency_error_hz - frequency_error) <
                    0.0011 && track_fit->rms < 0.01 &&
                std::abs(track_fit->amplitude - 1.0) < 0.01,
            "time-aware phase track fit failed synthetic data");

    std::vector<task5::PhaseTrackSample> abba;
    constexpr double abba_error_hz = 0.043;
    constexpr double abba_phase_at_zero = -0.61;
    constexpr double dither = 12.0 * 2.0 * kPi / 256.0;
    const std::array<double, 4> abba_commands{
        -dither, dither, dither, -dither};
    for (int block = 0; block < 4; ++block) {
        for (int frame = 0; frame < 6; ++frame) {
            const double time = 0.48 * block + 0.03 * frame;
            const double noise = ((block + frame) % 2 == 0) ? 0.008 : -0.008;
            abba.push_back({time, abba_commands[block],
                0.88 * std::cos(2.0 * kPi * abba_error_hz * time +
                                abba_commands[block] +
                                abba_phase_at_zero) + noise});
        }
    }
    const double abba_reference = 2.0;
    const auto abba_fit = task5::fit_phase_track(
        abba, abba_reference, 0.25, 0.0005);
    const double expected_abba_phase = abba_phase_at_zero +
        2.0 * kPi * abba_error_hz * abba_reference;
    std::cout << "abba expected_error=" << abba_error_hz
              << " fit_error="
              << (abba_fit ? abba_fit->frequency_error_hz : 999.0)
              << " expected_phase=" << expected_abba_phase
              << " fit_phase="
              << (abba_fit ? abba_fit->relative_phase_rad : 999.0)
              << " rms=" << (abba_fit ? abba_fit->rms : 999.0)
              << " amplitude=" << (abba_fit ? abba_fit->amplitude : 0.0)
              << '\n';
    require(abba_fit.has_value() &&
                std::abs(abba_fit->frequency_error_hz - abba_error_hz) <
                    0.002 &&
                std::abs(std::atan2(
                    std::sin(abba_fit->relative_phase_rad -
                             expected_abba_phase),
                    std::cos(abba_fit->relative_phase_rad -
                             expected_abba_phase))) < 0.03,
            "ABBA phase schedule did not recover signed phase drift");
}
}  // namespace

int main() {
    test_frequency_observer();
    test_phase_code_observer();
    test_quantized_phase_code_matcher();
    test_multirate_observer();
    test_phase_slope();
    test_pll_is_bounded();
    test_controller_deadline_and_reacquire();
    test_uart_packet();
    test_visual_lock_state();
    test_quadrature_phase_capture();
    test_clock_calibration();
    test_sine_matcher();
    test_candidate_bank_matcher();
    test_shape_observer();
    test_visual_frequency_servo();
    std::cout << "TASK5_CPP_SYNTHETIC_PASS\n";
    return 0;
}
