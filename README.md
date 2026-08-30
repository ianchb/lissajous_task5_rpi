# Visual Lissajous Frequency Recognition and Phase Control

2026年电赛 F 题：李萨如图形显示控制装置的主机程序

程序通过摄像头观察示波器 X-Y 画面，
与配套 FPGA 协作完成示波器网格定位、1 kHz--100 kHz 视觉测频以及直线、圆形和
“∞”图形的相位控制。

本仓库只包含 Linux 侧软件。

> [!WARNING]
> 本仓库中的代码不是最终采用的版本，后期开发内容没有完整保存在本仓库中。
> 由于比赛时间有限，开发过程中有多处代码使用 AI 工具辅助生成或修改，赛后也未进行
> 系统维护、审计和完整复测。本项目按“原样”提供，不保证其正确性、性能、适用性，
> 也不保证能够复现比赛现场效果，仅供参考学习。

## 工作原理

### 四带视觉测频

普通静态李萨如图只能反映频率比和相位关系，无法给出输入信号的绝对频率。配套 FPGA
因此以 100 Hz 为基准构造 10 ms 相干帧，并在以下四个时刻向示波器 Y 轴输出短阶梯扫描：

```text
0, T/7, T/11, T/13    (T = 10 ms)
```

外部待测正弦信号直接进入示波器 X 轴。若输入频率为 `f = n * 100 Hz`，后三条扫描带
相对第一条带的相位分别包含 `n mod 7`、`n mod 11` 和 `n mod 13`。由于
`7 * 11 * 13 = 1001`，它们能够唯一覆盖 `n = 10..1000`，即 1 kHz--100 kHz 的
全部 100 Hz 档位。

主机程序执行以下处理：

1. 检测示波器 8x8 网格并计算透视单应矩阵；
2. 提取黄色轨迹并归一化四条扫描带；
3. 按 FPGA 的真实 DAC 阶梯和驻留时间生成候选模板；
4. 联合枚举候选频率与未知公共初相位，比较四带左右轮廓；
5. 使用多个独立观察窗确认同一 100 Hz 频点。

摄像头不需要直接采样几十千赫兹的电信号。FPGA 已经把高速时间信息转换成可由普通
摄像头读取的空间条带图案。

### 图形相位控制

频率确认后，程序使用本次上电获得的 FPGA 时钟校准值计算细分 DDS 控制字：

- 直线和圆形模式输出与输入同频的正弦；
- “∞”模式输出输入频率二倍频的正弦；
- 通过已知相位试探和图像几何特征估计目标相位；
- 根据相关性、轨迹短轴、径向误差、对称性和中心交叉等指标选择最可能相位；
- 锁定后保持输出，并在明确失配时重新捕获相位或频率。

自动模式下，频率识别和相位控制只使用摄像头画面，不从输入信号向 FPGA 建立电气测量
通路。

## 目录结构

```text
include/task5/                 公共接口和数据结构
src/                           视觉、串口、频率匹配和运行控制实现
tests/synthetic_test.cpp       合成回归测试
docs/task5-autostart.service   systemd 服务模板
CMakeLists.txt                 构建入口
install.sh                     构建、测试和服务安装脚本
```

主要可执行程序：

- `task5_runtime`：完整运行流程；
- `task5_hmi_gateway`：串口屏握手、状态显示及运行程序管理；
- `task5_startup_wait`：不使用串口屏时的旧式启动等待程序；
- `task5_controller`：底层采集和控制诊断入口；
- `task5_synthetic_test`：不依赖硬件的合成测试。

## 依赖

构建需要支持 C++20 的编译器、CMake 3.20 及 OpenCV。我们使用的发行版：`Debian GNU/Linux Forky & Trixie`。

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## 直接运行

使用摄像头 `/dev/video0` 和 FPGA 串口 `/dev/ttyUSB0`：

```bash
./build/task5_runtime \
  --camera /dev/video0 \
  --serial /dev/ttyUSB0 \
  --frequency-windows 3 \
  --frequency-window-ms 1000
```

常用参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--camera` | `/dev/video19` | 摄像头设备节点 |
| `--serial` | `/dev/ttyUSB0` | FPGA 串口设备节点 |
| `--frequency-windows` | `3` | 测频独立窗口数，最少为 3 |
| `--frequency-window-ms` | `1000` | 单个测频窗口时长 |
| `--shape-settle-ms` | `600` | 相位命令后的图像稳定等待时间 |
| `--grid-exposures` | `40,60,80` | 网格搜索时依次尝试的曝光值 |

程序持续轮询 FPGA 模式：

```text
0  TASK1
1  TASK2
2  TASK3
3  AUTO_LINE
4  AUTO_CIRCLE
5  AUTO_FIGURE8
```

前三个模式用于前置题目和时钟校准；后三个模式执行视觉测频与目标图形控制。

## 串口屏启动

串口屏默认为 `/dev/ttyACM0`、波特率 115200：

```bash
./build/task5_hmi_gateway \
  --screen /dev/ttyACM0 \
  --fpga /dev/ttyUSB0 \
  --screen-baud 115200 \
  --camera /dev/video0 \
  --runtime "$PWD/build/task5_runtime" \
  -- --frequency-windows 3 --frequency-window-ms 1000
```

`--` 后的参数会原样传递给 `task5_runtime`。

## 安装开机服务

服务模板默认假定：

```text
用户              siergtc
工程目录          /home/siergtc/task5_cpp
摄像头            /dev/video0
串口屏            /dev/ttyACM0
FPGA 串口         /dev/ttyUSB0
```

如果设备环境不同，先修改 `docs/task5-autostart.service`。随后执行：

```bash
chmod +x install.sh
./install.sh
```

查看服务状态和日志：

```bash
systemctl status task5-autostart.service
journalctl -u task5-autostart.service -f
```

停止并取消开机启动：

```bash
sudo systemctl disable --now task5-autostart.service
```

## License

本项目采用 [Apache License 2.0](LICENSE) 许可。除许可证明确规定外，软件按“原样”
提供，不附带任何明示或默示担保。
