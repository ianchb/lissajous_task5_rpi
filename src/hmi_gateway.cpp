// Boot gateway for a Nextion serial screen and the FPGA controller.
//
// The runtime keeps exclusive ownership of the FPGA serial device.  This
// process discovers the screen first, waits for BTN0, then starts the runtime
// and turns its existing structured log lines into screen state updates.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <optional>
#include <poll.h>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr std::array<uint8_t, 3> kNextionEnd{0xff, 0xff, 0xff};
constexpr uint8_t kFpgaStatusCommand = 0x02;
volatile std::sig_atomic_t stop_requested = 0;
volatile std::sig_atomic_t child_pid = -1;

void handle_signal(int) {
    stop_requested = 1;
    if (child_pid > 0) ::kill(static_cast<pid_t>(child_pid), SIGTERM);
}

uint8_t checksum(const uint8_t* data, size_t size) {
    uint8_t value = 0;
    for (size_t index = 0; index < size; ++index) value ^= data[index];
    return value;
}

std::array<uint8_t, 11> fpga_status_request() {
    std::array<uint8_t, 11> packet{};
    packet[0] = 0xa5;
    packet[1] = 0x5a;
    packet[2] = kFpgaStatusCommand;
    packet[10] = checksum(packet.data(), 10);
    return packet;
}

class SerialPort {
public:
    ~SerialPort() { close(); }

    bool open(const std::string& path, int baud) {
        close();
        fd_ = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) return false;
        termios options{};
        if (tcgetattr(fd_, &options) != 0) {
            close();
            return false;
        }
        cfmakeraw(&options);
        const speed_t speed = baud == 9600 ? B9600 : B115200;
        cfsetispeed(&options, speed);
        cfsetospeed(&options, speed);
        options.c_cflag |= CLOCAL | CREAD;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CRTSCTS;
        if (tcsetattr(fd_, TCSANOW, &options) != 0) {
            close();
            return false;
        }
        tcflush(fd_, TCIOFLUSH);
        return true;
    }

    void close() {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        input_.clear();
    }

    bool is_open() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    const std::string& input() const { return input_; }
    void clear_input() { input_.clear(); }

    bool write_all(const uint8_t* data, size_t size) {
        for (size_t written = 0; written < size;) {
            const ssize_t count = ::write(fd_, data + written, size - written);
            if (count > 0) written += static_cast<size_t>(count);
            else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                usleep(1000);
            else return false;
        }
        return true;
    }

    bool read_available() {
        if (!is_open()) return false;
        uint8_t bytes[256];
        while (true) {
            const ssize_t count = ::read(fd_, bytes, sizeof(bytes));
            if (count > 0) input_.append(reinterpret_cast<const char*>(bytes),
                                         static_cast<size_t>(count));
            else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                return true;
            else if (count == 0) return true;
            else {
                close();
                return false;
            }
        }
    }

private:
    int fd_ = -1;
    std::string input_;
};

struct Arguments {
    std::string screen = "/dev/ttyACM0";
    std::string fpga = "/dev/ttyUSB0";
    int screen_baud = 115200;
    std::string camera = "/dev/video0";
    std::string runtime = "/usr/local/bin/task5_runtime";
    std::vector<std::string> runtime_args;
};

bool parse_arguments(int argc, char** argv, Arguments* args) {
    bool forward = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (forward) {
            args->runtime_args.push_back(argument);
            continue;
        }
        if (argument == "--") {
            forward = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const std::string value(argv[++index]);
        if (argument == "--screen") {
            args->screen = value;
        } else if (argument == "--fpga") {
            args->fpga = value;
        } else if (argument == "--screen-baud") {
            try {
                args->screen_baud = std::stoi(value);
            } catch (...) {
                return false;
            }
            if (args->screen_baud != 9600 && args->screen_baud != 115200)
                return false;
        } else if (argument == "--camera") {
            args->camera = value;
        } else if (argument == "--runtime") {
            args->runtime = value;
        } else {
            return false;
        }
    }
    return !args->screen.empty() && !args->fpga.empty() &&
           !args->camera.empty() && !args->runtime.empty();
}

[[noreturn]] void usage(const char* name, int result) {
    std::cerr << "usage: " << name
              << " [--screen /dev/ttyACM0] [--fpga /dev/ttyUSB0]"
                 " [--screen-baud 115200] [--camera /dev/video0]"
                 " [--runtime /usr/local/bin/task5_runtime]"
                 " [-- runtime arguments...]\n";
    std::exit(result);
}

bool fpga_response_present(SerialPort* port) {
    const auto request = fpga_status_request();
    if (!port->write_all(request.data(), request.size())) return false;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        port->read_available();
        const std::string& bytes = port->input();
        for (size_t i = 0; i + 11 < bytes.size(); ++i) {
            const auto* packet = reinterpret_cast<const uint8_t*>(bytes.data() + i);
            if (packet[0] == 0x5a && packet[1] == 0xa5 &&
                packet[10] == kFpgaStatusCommand &&
                checksum(packet, 11) == packet[11])
                return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

std::string bytes_to_hex(const std::string& bytes) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (size_t index = 0; index < bytes.size(); ++index) {
        if (index != 0) output << ' ';
        output << std::setw(2) << static_cast<unsigned>(
            static_cast<uint8_t>(bytes[index]));
    }
    return output.str();
}

bool has_button0_event(const std::string& bytes) {
    if (bytes.find("BTN0") != std::string::npos) return true;
    // Standard TJC/Nextion touch return: 0x65, page, component, event,
    // 0xff, 0xff, 0xff.  Component 0 is button0; accept press or release.
    for (size_t index = 0; index + 6 < bytes.size(); ++index) {
        const auto* event = reinterpret_cast<const uint8_t*>(bytes.data() + index);
        if (event[0] == 0x65 && event[2] == 0x00 &&
            event[4] == 0xff && event[5] == 0xff && event[6] == 0xff)
            return true;
    }
    return false;
}

constexpr std::string_view kHmiHeartbeat = "T5HMI\n";

struct ScreenState {
    int n0 = 0;
    int n1 = -1;
    int n2 = -1;
    int n3 = 0;
    int n4 = -1;
    std::array<int, 5> sent{std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::min()};
};

bool set_number(SerialPort* screen, const char* name, int value) {
    const std::string command = std::string(name) + ".val=" +
                                std::to_string(value);
    return screen->write_all(reinterpret_cast<const uint8_t*>(command.data()),
                             command.size()) &&
           screen->write_all(kNextionEnd.data(), kNextionEnd.size());
}

bool acknowledge_hmi_heartbeat(SerialPort* screen,
                               std::string* receive_buffer) {
    if (!screen || !receive_buffer) return false;
    const size_t position = receive_buffer->find(kHmiHeartbeat);
    if (position == std::string::npos) return false;
    // The screen's local timer increments host_wd. Resetting it here makes
    // the screen able to report a missing host even when the ACM node is gone.
    set_number(screen, "host_wd", 0);
    receive_buffer->erase(0, position + kHmiHeartbeat.size());
    return true;
}

bool publish(SerialPort* screen, ScreenState* state, bool force = false) {
    const std::array<std::pair<const char*, int>, 5> values{{
        {"n0", state->n0}, {"n1", state->n1}, {"n2", state->n2},
        {"n3", state->n3}, {"n4", state->n4},
    }};
    for (size_t index = 0; index < values.size(); ++index) {
        if (!force && state->sent[index] == values[index].second) continue;
        if (!set_number(screen, values[index].first, values[index].second))
            return false;
        state->sent[index] = values[index].second;
    }
    return true;
}

std::optional<int> parse_int_after(const std::string& line,
                                   const std::string& key) {
    const size_t begin = line.find(key);
    if (begin == std::string::npos) return std::nullopt;
    const char* first = line.c_str() + begin + key.size();
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(first, &end, 10);
    if (errno != 0 || end == first) return std::nullopt;
    return static_cast<int>(value);
}

std::optional<int> parse_hz_after(const std::string& line,
                                  const std::string& key) {
    const size_t begin = line.find(key);
    if (begin == std::string::npos) return std::nullopt;
    const char* first = line.c_str() + begin + key.size();
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(first, &end);
    if (errno != 0 || end == first || !std::isfinite(value) ||
        value < -1.0 || value > 1'000'000.0)
        return std::nullopt;
    return static_cast<int>(std::lround(value));
}

void apply_runtime_log(const std::string& line, ScreenState* state) {
    if (line.rfind("MODE_CHANGE mode=", 0) == 0 ||
        line.rfind("RUNTIME_START mode=", 0) == 0) {
        const auto mode = parse_int_after(line, "mode=");
        if (!mode) return;
        state->n1 = *mode;
        state->n2 = -1;
        state->n4 = *mode >= 3 ? 0 : -1;
        return;
    }
    if (line.rfind("GRID_LOCKED ", 0) == 0) {
        state->n3 = 1;
        return;
    }
    // FREQUENCY_LOCK is only the phase-code consensus. The visual verifier
    // can still reject it and walk through several fallback candidates, so
    // publishing it here leaves the HMI showing a discarded frequency.
    if (line.rfind("FREQUENCY_LOCK hz=", 0) == 0) return;
    if (line.rfind("FREQUENCY_CONFIRM_START ", 0) == 0 ||
        line.rfind("FREQUENCY_VISUAL_CONFIRMED ", 0) == 0) {
        // A startup prior is only a probe. Do not expose it on the screen
        // until the visual verifier has accepted the requested frequency.
        if (line.rfind("FREQUENCY_CONFIRM_START ", 0) == 0 &&
            line.find("source=startup_prior") != std::string::npos)
            return;
        const auto frequency = parse_hz_after(line, "hz=");
        if (frequency) state->n2 = *frequency;
        return;
    }
    if (line.rfind("LOCKED ", 0) == 0 ||
        line.rfind("STABLE_5S ", 0) == 0) {
        const auto frequency = parse_hz_after(line, "input_hz=");
        if (frequency) state->n2 = *frequency;
        return;
    }
    if (line.rfind("LOCK_INDICATOR value=", 0) == 0) {
        const auto locked = parse_int_after(line, "value=");
        if (locked && state->n1 >= 3) state->n4 = *locked ? 1 : 0;
        return;
    }
    if (line.rfind("RETRY ", 0) == 0 ||
        line.rfind("REACQUIRE reason=confirmed_frequency_mismatch", 0) == 0 ||
        line.rfind("REACQUIRE reason=confirmed_filled_rectangle", 0) == 0) {
        state->n2 = -1;
        return;
    }
}

pid_t launch_runtime(const Arguments& args, const std::string& fpga,
                     int* output_read_fd) {
    int pipe_fds[2]{};
    if (::pipe2(pipe_fds, O_CLOEXEC) != 0) return -1;
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        return -1;
    }
    if (pid == 0) {
        ::dup2(pipe_fds[1], STDOUT_FILENO);
        ::dup2(pipe_fds[1], STDERR_FILENO);
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        std::vector<std::string> values;
        values.push_back(args.runtime);
        values.push_back("--camera");
        values.push_back(args.camera);
        values.push_back("--serial");
        values.push_back(fpga);
        values.insert(values.end(), args.runtime_args.begin(),
                      args.runtime_args.end());
        std::vector<char*> argv;
        argv.reserve(values.size() + 1);
        for (std::string& value : values) argv.push_back(value.data());
        argv.push_back(nullptr);
        ::execv(args.runtime.c_str(), argv.data());
        std::cerr << "cannot exec runtime " << args.runtime << ": "
                  << std::strerror(errno) << "\n";
        _exit(127);
    }
    ::close(pipe_fds[1]);
    const int flags = ::fcntl(pipe_fds[0], F_GETFL, 0);
    if (flags < 0 || ::fcntl(pipe_fds[0], F_SETFL, flags | O_NONBLOCK) != 0) {
        ::close(pipe_fds[0]);
        return -1;
    }
    *output_read_fd = pipe_fds[0];
    return pid;
}

int supervise_runtime(const Arguments& args, SerialPort* screen,
                      const std::string& fpga) {
    int runtime_output = -1;
    const pid_t pid = launch_runtime(args, fpga, &runtime_output);
    if (pid < 0) {
        std::cerr << "RUNTIME_LAUNCH_FAILED error=" << std::strerror(errno)
                  << "\n";
        return 127;
    }
    child_pid = pid;
    ScreenState state;
    state.n0 = 1;
    if (!publish(screen, &state, true))
        std::cerr << "HMI_WRITE_LOST\n";
    std::cout << "RUNTIME_LAUNCHED fpga=" << fpga << " pid=" << pid << "\n"
              << std::flush;

    std::string pending;
    std::string hmi_receive_buffer;
    bool heartbeat_acknowledged = false;
    bool child_done = false;
    int child_status = 0;
    while (!stop_requested && !child_done) {
        pollfd fds[2]{{runtime_output, POLLIN, 0}, {screen->fd(), POLLIN, 0}};
        const int count = ::poll(fds, screen->is_open() ? 2 : 1, 100);
        if (count > 0 && (fds[0].revents & (POLLIN | POLLHUP))) {
            char buffer[512];
            const ssize_t read_count = ::read(runtime_output, buffer,
                                              sizeof(buffer));
            if (read_count > 0) {
                pending.append(buffer, static_cast<size_t>(read_count));
                while (true) {
                    const size_t end = pending.find('\n');
                    if (end == std::string::npos) break;
                    const std::string line = pending.substr(0, end);
                    pending.erase(0, end + 1);
                    std::cout << line << "\n" << std::flush;
                    apply_runtime_log(line, &state);
                    if (screen->is_open() && !publish(screen, &state)) {
                        std::cerr << "HMI_WRITE_LOST\n";
                        screen->close();
                    }
                }
            }
        }
        if (screen->is_open() && count > 0 && (fds[1].revents & POLLIN)) {
            if (!screen->read_available()) screen->close();
            else {
                hmi_receive_buffer.append(screen->input());
                screen->clear_input();  // BTN0 has no action while running.
                while (acknowledge_hmi_heartbeat(
                    screen, &hmi_receive_buffer)) {
                    if (!heartbeat_acknowledged) {
                        std::cout << "HMI_HEARTBEAT_ACK state=running\n"
                                  << std::flush;
                        heartbeat_acknowledged = true;
                    }
                }
                if (hmi_receive_buffer.size() > 64)
                    hmi_receive_buffer.erase(
                        0, hmi_receive_buffer.size() - 16);
            }
        }
        const pid_t waited = ::waitpid(pid, &child_status, WNOHANG);
        child_done = waited == pid;
    }
    if (!child_done) {
        ::kill(pid, SIGTERM);
        while (::waitpid(pid, &child_status, 0) < 0 && errno == EINTR) {}
    }
    child_pid = -1;
    ::close(runtime_output);
    if (WIFEXITED(child_status)) return WEXITSTATUS(child_status);
    return 128;
}

}  // namespace

int main(int argc, char** argv) {
    Arguments args;
    if (!parse_arguments(argc, argv, &args)) usage(argv[0], 2);
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::cerr << "HMI_GATEWAY_WAIT screen=" << args.screen
              << " fpga=" << args.fpga << "\n";
    while (!stop_requested) {
        SerialPort screen;
        if (!screen.open(args.screen, args.screen_baud)) {
            std::cerr << "HMI_SCREEN_WAIT serial=" << args.screen << "\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        SerialPort fpga_probe;
        if (!fpga_probe.open(args.fpga, 115200) ||
            !fpga_response_present(&fpga_probe)) {
            std::cerr << "FPGA_WAIT serial=" << args.fpga << "\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        fpga_probe.close();

        ScreenState idle;
        publish(&screen, &idle, true);
        std::cout << "HMI_CONNECTED serial=" << args.screen
                  << " baud=" << args.screen_baud << "\n"
                  << "FPGA_CONNECTED serial=" << args.fpga << "\n"
                  << "HMI_WAIT_BUTTON button=0\n" << std::flush;
        // UART reads are not message reads: a 6-byte `BTN0\n` command can
        // arrive as e.g. `BT` then `N0\n`. Keep a small rolling buffer so a
        // partial screen transmission is never discarded between poll calls.
        std::string hmi_receive_buffer;
        bool heartbeat_acknowledged = false;
        while (!stop_requested) {
            pollfd descriptor{screen.fd(), POLLIN, 0};
            const int poll_result = ::poll(&descriptor, 1, 200);
            if (poll_result > 0 && (descriptor.revents & POLLIN)) {
                if (!screen.read_available()) break;
                const std::string received = screen.input();
                screen.clear_input();
                hmi_receive_buffer.append(received);
                if (has_button0_event(hmi_receive_buffer)) {
                    std::cout << "HMI_BUTTON button=0\n" << std::flush;
                    return supervise_runtime(args, &screen, args.fpga);
                }
                while (acknowledge_hmi_heartbeat(
                    &screen, &hmi_receive_buffer)) {
                    if (!heartbeat_acknowledged) {
                        std::cout << "HMI_HEARTBEAT_ACK state=idle\n"
                                  << std::flush;
                        heartbeat_acknowledged = true;
                    }
                }
                if (!received.empty()) {
                    std::cout << "HMI_RX bytes="
                              << bytes_to_hex(received) << "\n"
                              << std::flush;
                }
                // The longest supported binary touch frame is 7 bytes.
                // Retain a little more than that to allow any split point,
                // while bounding the buffer under the 500 ms heartbeat.
                if (hmi_receive_buffer.size() > 64)
                    hmi_receive_buffer.erase(
                        0, hmi_receive_buffer.size() - 16);
            }
            if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) break;
        }
        std::cerr << "HMI_SCREEN_RECONNECT\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
