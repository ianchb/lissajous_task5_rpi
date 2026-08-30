// Wait for the FPGA's unsolicited boot/mode packet before starting runtime.
// This intentionally does not use UartTransport::open(), because that method
// flushes the input queue and can erase the one-shot boot packet.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <termios.h>

namespace {

constexpr uint8_t kResponseMagic0 = 0x5a;
constexpr uint8_t kResponseMagic1 = 0xa5;
constexpr uint8_t kCommandStatus = 0x02;
constexpr uint8_t kCommandStartup = 0x80;

volatile std::sig_atomic_t stop_requested = 0;

void handle_signal(int) { stop_requested = 1; }

struct Response {
    uint8_t mode = 0;
    uint8_t flags = 0;
    uint8_t command = 0;
};

uint8_t checksum(const uint8_t* data, size_t count) {
    uint8_t result = 0;
    for (size_t index = 0; index < count; ++index) result ^= data[index];
    return result;
}

std::array<uint8_t, 11> status_request() {
    std::array<uint8_t, 11> packet{};
    packet[0] = 0xa5;
    packet[1] = 0x5a;
    packet[2] = kCommandStatus;
    packet[10] = checksum(packet.data(), 10);
    return packet;
}

class SerialProbe {
public:
    ~SerialProbe() { close(); }

    bool open(const std::string& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            last_error_ = errno;
            return false;
        }

        termios options{};
        if (tcgetattr(fd_, &options) != 0) {
            last_error_ = errno;
            close();
            return false;
        }
        cfmakeraw(&options);
        cfsetispeed(&options, B115200);
        cfsetospeed(&options, B115200);
        options.c_cflag |= CLOCAL | CREAD;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CRTSCTS;
        if (tcsetattr(fd_, TCSANOW, &options) != 0) {
            last_error_ = errno;
            close();
            return false;
        }
        // Do not call tcflush here: the FPGA can have sent its boot packet
        // before this process managed to open the device.
        input_.clear();
        last_error_ = 0;
        return true;
    }

    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        input_.clear();
    }

    bool is_open() const { return fd_ >= 0; }
    int last_error() const { return last_error_; }

    bool send_status_request() {
        if (fd_ < 0) return false;
        const auto packet = status_request();
        size_t offset = 0;
        while (offset < packet.size()) {
            const ssize_t count = ::write(fd_, packet.data() + offset,
                                          packet.size() - offset);
            if (count > 0) {
                offset += static_cast<size_t>(count);
            } else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                usleep(1000);
            } else {
                return false;
            }
        }
        return true;
    }

    // Returns one valid FPGA response, preserving incomplete bytes for the
    // next read. Invalid bytes are discarded one at a time for resync.
    bool receive(Response* response) {
        if (fd_ < 0 || !response) return false;
        uint8_t chunk[256];
        const ssize_t count = ::read(fd_, chunk, sizeof(chunk));
        if (count > 0) input_.insert(input_.end(), chunk, chunk + count);
        else if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            close();
            return false;
        } else if (count == 0) {
            close();
            return false;
        }

        constexpr std::array<uint8_t, 2> magic_bytes{
            kResponseMagic0, kResponseMagic1};
        while (input_.size() >= 12) {
            auto magic = std::search(input_.begin(), input_.end(),
                                     magic_bytes.begin(), magic_bytes.end());
            if (magic == input_.end()) {
                input_.erase(input_.begin(), input_.end() - 1);
                return false;
            }
            if (magic != input_.begin()) input_.erase(input_.begin(), magic);
            if (input_.size() < 12) return false;
            if (checksum(input_.data(), 11) != input_[11]) {
                input_.erase(input_.begin());
                continue;
            }
            response->mode = input_[2];
            response->flags = input_[3];
            response->command = input_[10];
            input_.erase(input_.begin(), input_.begin() + 12);
            return true;
        }
        return false;
    }

private:
    int fd_ = -1;
    int last_error_ = 0;
    std::vector<uint8_t> input_;
};

struct Arguments {
    std::string serial = "/dev/ttyUSB0";
    std::string camera = "/dev/video0";
    std::string runtime = "/usr/local/bin/task5_runtime";
    int start_mode = 1;
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
        } else if (argument == "--serial" || argument == "--camera" ||
                   argument == "--runtime" || argument == "--start-mode") {
            if (index + 1 >= argc) return false;
            const std::string value(argv[++index]);
            if (argument == "--serial") args->serial = value;
            else if (argument == "--camera") args->camera = value;
            else if (argument == "--runtime") args->runtime = value;
            else {
                try {
                    args->start_mode = std::stoi(value);
                } catch (...) {
                    return false;
                }
            }
        } else if (argument == "--help" || argument == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << argument << "\n";
            return false;
        }
    }
    return !args->serial.empty() && !args->runtime.empty() &&
           args->start_mode >= 0 && args->start_mode <= 7;
}

[[noreturn]] void usage(const char* name, int result) {
    std::cerr << "usage: " << name
              << " [--serial /dev/ttyUSB0] [--camera /dev/video0]"
                 " [--runtime /usr/local/bin/task5_runtime]"
                 " [--start-mode 1]"
                 " [-- runtime arguments...]\n";
    std::exit(result);
}

int launch_runtime(const Arguments& args) {
    std::vector<std::string> values;
    values.reserve(5 + args.runtime_args.size());
    values.push_back(args.runtime);
    values.push_back("--camera");
    values.push_back(args.camera);
    values.push_back("--serial");
    values.push_back(args.serial);
    values.insert(values.end(), args.runtime_args.begin(), args.runtime_args.end());

    std::vector<char*> pointers;
    pointers.reserve(values.size() + 1);
    for (std::string& value : values) pointers.push_back(value.data());
    pointers.push_back(nullptr);
    ::execv(args.runtime.c_str(), pointers.data());
    std::cerr << "cannot exec runtime " << args.runtime << ": "
              << std::strerror(errno) << "\n";
    return 127;
}

}  // namespace

int main(int argc, char** argv) {
    Arguments args;
    if (!parse_arguments(argc, argv, &args)) usage(argv[0], 2);
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    SerialProbe serial;
    auto next_status_probe = std::chrono::steady_clock::now();
    auto next_open_error_report = std::chrono::steady_clock::now();
    int last_open_error = -1;
    int last_reported_mode = -1;
    std::cerr << "FPGA_WAIT serial=" << args.serial
              << " start_mode=" << args.start_mode << "\n";
    while (!stop_requested) {
        if (!serial.is_open()) {
            if (!serial.open(args.serial)) {
                const auto now = std::chrono::steady_clock::now();
                if (serial.last_error() != last_open_error ||
                    now >= next_open_error_report) {
                    last_open_error = serial.last_error();
                    std::cerr << "FPGA_SERIAL_WAIT error="
                              << std::strerror(last_open_error) << "\n";
                    next_open_error_report = now + std::chrono::seconds(5);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            std::cerr << "FPGA_SERIAL_OPEN\n";
            next_status_probe = std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(250);
        }

        // SerialProbe deliberately keeps the descriptor private. A short
        // sleep is sufficient here and avoids exposing another fd API.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Response response;
        if (serial.receive(&response)) {
            if (response.command == kCommandStartup ||
                response.command == kCommandStatus) {
                if (response.mode == args.start_mode) {
                    std::cerr << "FPGA_HANDSHAKE command=0x" << std::hex
                              << static_cast<int>(response.command) << std::dec
                              << " mode=" << static_cast<int>(response.mode)
                              << "\n";
                    serial.close();
                    return launch_runtime(args);
                }
                if (response.mode != last_reported_mode) {
                    last_reported_mode = response.mode;
                    std::cerr << "FPGA_MODE_WAIT current="
                              << static_cast<int>(response.mode)
                              << " expected=" << args.start_mode << "\n";
                }
            }
        }
        if (std::chrono::steady_clock::now() >= next_status_probe) {
            if (!serial.send_status_request()) serial.close();
            next_status_probe = std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(1000);
        }
    }
    serial.close();
    return 0;
}
