cmake_minimum_required(VERSION 3.20)
project(lissajous_task5_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d)

add_library(task5_core
    src/candidate_bank_matcher.cpp
    src/camera.cpp
    src/clock_calibration.cpp
    src/command_scheduler.cpp
    src/controller_state.cpp
    src/frequency_observer.cpp
    src/grid_calibrator.cpp
    src/lock_state.cpp
    src/phase_code_matcher.cpp
    src/phase_capture.cpp
    src/pll.cpp
    src/runtime_controller.cpp
    src/shape_observer.cpp
    src/sine_matcher.cpp
    src/trace_segmenter.cpp
    src/uart.cpp
    src/visual_frequency_servo.cpp
)
target_include_directories(task5_core PUBLIC include)
target_link_libraries(task5_core PUBLIC ${OpenCV_LIBS})
target_compile_options(task5_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(task5_controller src/main.cpp)
target_link_libraries(task5_controller PRIVATE task5_core)
target_compile_options(task5_controller PRIVATE -Wall -Wextra -Wpedantic)

add_executable(task5_runtime src/runtime_main.cpp)
target_link_libraries(task5_runtime PRIVATE task5_core)
target_compile_options(task5_runtime PRIVATE -Wall -Wextra -Wpedantic)

if(UNIX)
    add_executable(task5_startup_wait src/startup_wait.cpp)
    target_compile_options(task5_startup_wait PRIVATE -Wall -Wextra -Wpedantic)
    add_executable(task5_hmi_gateway src/hmi_gateway.cpp)
    target_compile_options(task5_hmi_gateway PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(task5_synthetic_test tests/synthetic_test.cpp)
target_link_libraries(task5_synthetic_test PRIVATE task5_core)
target_compile_options(task5_synthetic_test PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME task5_synthetic_test COMMAND task5_synthetic_test)

if(UNIX)
    include(GNUInstallDirs)
    install(TARGETS task5_runtime task5_startup_wait task5_hmi_gateway
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
