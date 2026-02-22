#include <GL/glew.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <json/json.h>
#include <zstd.h>
#include <zmq.h>

#include "System.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

std::atomic<bool> g_running{true};

void HandleSignal(int) { g_running = false; }

double MonotonicNowSec() {
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

struct Options {
  std::string vocab_path;
  std::string settings_template_path;
  std::string endpoint = "tcp://0.0.0.0:5555";
  std::string topic = "rgbd";
  bool bind = true;
  bool viewer = true;
  int rcv_hwm = 2;
  int poll_timeout_ms = 100;
  std::string trajectory_path;
};

struct FrameMeta {
  int seq = -1;
  double ts = 0.0;
  int w = 0;
  int h = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::string rgb_encoding = "BGR";
  std::string rgb_codec = "jpeg";
  std::string depth_encoding = "f32_meters";
  std::string depth_codec = "zstd";
  double depth_scale = 1.0;
  std::size_t depth_uncompressed_bytes = 0;
};

struct RawPacket {
  std::string header_json;
  std::vector<std::uint8_t> rgb_bytes;
  std::vector<std::uint8_t> depth_bytes;
};

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " --vocab <ORBvoc.txt> --settings-template <template.yaml> [options]\n"
      << "Options:\n"
      << "  --endpoint <zmq_endpoint>        Default: tcp://0.0.0.0:5555\n"
      << "  --topic <topic>                  Default: rgbd\n"
      << "  --connect                        Connect SUB socket (default is bind)\n"
      << "  --no-viewer                      Disable Pangolin viewer\n"
      << "  --rcv-hwm <int>                  Default: 2\n"
      << "  --poll-timeout-ms <int>          Default: 100\n"
      << "  --trajectory <path.txt>          Save TUM trajectory on shutdown\n";
}

bool ParseArgs(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string& flag, std::string* out) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      *out = argv[++i];
      return true;
    };
    auto need_int = [&](const std::string& flag, int* out) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      try {
        *out = std::stoi(argv[++i]);
      } catch (...) {
        std::cerr << "Invalid integer for " << flag << "\n";
        return false;
      }
      return true;
    };

    if (arg == "--help" || arg == "-h") {
      return false;
    } else if (arg == "--vocab") {
      if (!need_value(arg, &options->vocab_path)) {
        return false;
      }
    } else if (arg == "--settings-template") {
      if (!need_value(arg, &options->settings_template_path)) {
        return false;
      }
    } else if (arg == "--endpoint") {
      if (!need_value(arg, &options->endpoint)) {
        return false;
      }
    } else if (arg == "--topic") {
      if (!need_value(arg, &options->topic)) {
        return false;
      }
    } else if (arg == "--trajectory") {
      if (!need_value(arg, &options->trajectory_path)) {
        return false;
      }
    } else if (arg == "--rcv-hwm") {
      if (!need_int(arg, &options->rcv_hwm)) {
        return false;
      }
    } else if (arg == "--poll-timeout-ms") {
      if (!need_int(arg, &options->poll_timeout_ms)) {
        return false;
      }
    } else if (arg == "--connect") {
      options->bind = false;
    } else if (arg == "--no-viewer") {
      options->viewer = false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (options->vocab_path.empty() || options->settings_template_path.empty()) {
    std::cerr << "--vocab and --settings-template are required\n";
    return false;
  }
  return true;
}

bool ReceivePart(void* socket, std::vector<std::uint8_t>* out) {
  zmq_msg_t msg;
  zmq_msg_init(&msg);
  const int rc = zmq_msg_recv(&msg, socket, 0);
  if (rc < 0) {
    zmq_msg_close(&msg);
    return false;
  }
  const auto* begin = static_cast<const std::uint8_t*>(zmq_msg_data(&msg));
  out->assign(begin, begin + zmq_msg_size(&msg));
  zmq_msg_close(&msg);
  return true;
}

bool HasMoreParts(void* socket) {
  int more = 0;
  size_t more_size = sizeof(more);
  if (zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &more_size) != 0) {
    return false;
  }
  return more != 0;
}

bool ReceiveRawPacket(void* socket, const Options& options, RawPacket* packet) {
  zmq_pollitem_t item{};
  item.socket = socket;
  item.events = ZMQ_POLLIN;
  const int poll_rc = zmq_poll(&item, 1, options.poll_timeout_ms);
  if (poll_rc <= 0 || !(item.revents & ZMQ_POLLIN)) {
    return false;
  }

  std::vector<std::uint8_t> topic_bytes;
  if (!ReceivePart(socket, &topic_bytes)) {
    return false;
  }
  const std::string topic(topic_bytes.begin(), topic_bytes.end());
  if (!HasMoreParts(socket)) {
    return false;
  }
  if (topic != options.topic) {
    while (HasMoreParts(socket)) {
      std::vector<std::uint8_t> ignored;
      if (!ReceivePart(socket, &ignored)) {
        return false;
      }
    }
    return false;
  }

  std::vector<std::uint8_t> header_bytes;
  std::vector<std::uint8_t> rgb_bytes;
  std::vector<std::uint8_t> depth_bytes;
  if (!ReceivePart(socket, &header_bytes) || !HasMoreParts(socket)) {
    return false;
  }
  if (!ReceivePart(socket, &rgb_bytes) || !HasMoreParts(socket)) {
    return false;
  }
  if (!ReceivePart(socket, &depth_bytes)) {
    return false;
  }
  while (HasMoreParts(socket)) {
    std::vector<std::uint8_t> ignored;
    if (!ReceivePart(socket, &ignored)) {
      break;
    }
  }

  packet->header_json.assign(header_bytes.begin(), header_bytes.end());
  packet->rgb_bytes = std::move(rgb_bytes);
  packet->depth_bytes = std::move(depth_bytes);
  return true;
}

bool ParseHeader(const std::string& header_json, FrameMeta* meta) {
  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::istringstream iss(header_json);
  if (!Json::parseFromStream(builder, iss, &root, &errs)) {
    std::cerr << "[receiver] invalid header JSON: " << errs << "\n";
    return false;
  }

  meta->seq = root.get("seq", -1).asInt();
  meta->ts = root.get("ts", 0.0).asDouble();
  meta->w = root.get("w", 0).asInt();
  meta->h = root.get("h", 0).asInt();
  meta->fx = root.get("fx", 0.0).asDouble();
  meta->fy = root.get("fy", 0.0).asDouble();
  meta->cx = root.get("cx", 0.0).asDouble();
  meta->cy = root.get("cy", 0.0).asDouble();
  meta->rgb_encoding = root.get("rgb_encoding", "BGR").asString();
  meta->rgb_codec = root.get("rgb_codec", "jpeg").asString();
  meta->depth_encoding = root.get("depth_encoding", "f32_meters").asString();
  meta->depth_codec = root.get("depth_codec", "zstd").asString();
  meta->depth_scale = root.get("depth_scale", 1.0).asDouble();
  meta->depth_uncompressed_bytes = static_cast<std::size_t>(
      root.get("depth_uncompressed_bytes", 0).asLargestUInt());

  if (meta->w <= 0 || meta->h <= 0 || meta->fx <= 0.0 || meta->fy <= 0.0) {
    std::cerr << "[receiver] invalid metadata values in header\n";
    return false;
  }
  if (meta->ts <= 0.0) {
    meta->ts = MonotonicNowSec();
  }
  return true;
}

bool DecodeRgb(const std::vector<std::uint8_t>& rgb_bytes, const FrameMeta& meta, cv::Mat* rgb_out) {
  if (rgb_bytes.empty()) {
    return false;
  }
  cv::Mat encoded(1, static_cast<int>(rgb_bytes.size()), CV_8UC1,
                  const_cast<std::uint8_t*>(rgb_bytes.data()));
  cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (decoded.empty()) {
    std::cerr << "[receiver] failed to decode RGB frame\n";
    return false;
  }
  if (decoded.cols != meta.w || decoded.rows != meta.h) {
    std::cerr << "[receiver] RGB size mismatch. expected " << meta.w << "x" << meta.h << " got "
              << decoded.cols << "x" << decoded.rows << "\n";
    return false;
  }

  if (meta.rgb_encoding == "RGB") {
    cv::cvtColor(decoded, *rgb_out, cv::COLOR_BGR2RGB);
  } else {
    *rgb_out = decoded;
  }
  return true;
}

bool InflateDepth(const std::vector<std::uint8_t>& depth_bytes, const FrameMeta& meta,
                  std::vector<std::uint8_t>* uncompressed) {
  if (meta.w <= 0 || meta.h <= 0) {
    return false;
  }
  std::size_t expected = 0;
  if (meta.depth_encoding == "f32_meters") {
    expected = static_cast<std::size_t>(meta.w) * static_cast<std::size_t>(meta.h) * sizeof(float);
  } else if (meta.depth_encoding == "u16_mm") {
    expected =
        static_cast<std::size_t>(meta.w) * static_cast<std::size_t>(meta.h) * sizeof(std::uint16_t);
  } else {
    std::cerr << "[receiver] unsupported depth_encoding: " << meta.depth_encoding << "\n";
    return false;
  }

  if (meta.depth_uncompressed_bytes > 0) {
    expected = meta.depth_uncompressed_bytes;
  }
  uncompressed->resize(expected);

  if (meta.depth_codec == "zstd") {
    const std::size_t out_size =
        ZSTD_decompress(uncompressed->data(), uncompressed->size(), depth_bytes.data(), depth_bytes.size());
    if (ZSTD_isError(out_size)) {
      std::cerr << "[receiver] depth zstd decompress error: " << ZSTD_getErrorName(out_size) << "\n";
      return false;
    }
    if (out_size != uncompressed->size()) {
      std::cerr << "[receiver] depth size mismatch after decompress: " << out_size
                << " expected " << uncompressed->size() << "\n";
      return false;
    }
    return true;
  }

  if (meta.depth_codec == "raw") {
    if (depth_bytes.size() != uncompressed->size()) {
      std::cerr << "[receiver] raw depth size mismatch: " << depth_bytes.size() << " expected "
                << uncompressed->size() << "\n";
      return false;
    }
    *uncompressed = depth_bytes;
    return true;
  }

  std::cerr << "[receiver] unsupported depth_codec: " << meta.depth_codec << "\n";
  return false;
}

bool DecodeDepth(const std::vector<std::uint8_t>& depth_bytes, const FrameMeta& meta, cv::Mat* depth_out) {
  std::vector<std::uint8_t> raw;
  if (!InflateDepth(depth_bytes, meta, &raw)) {
    return false;
  }

  if (meta.depth_encoding == "f32_meters") {
    cv::Mat depth(meta.h, meta.w, CV_32FC1, raw.data());
    *depth_out = depth.clone();
  } else if (meta.depth_encoding == "u16_mm") {
    cv::Mat depth_u16(meta.h, meta.w, CV_16UC1, raw.data());
    const double scale = meta.depth_scale > 0.0 ? meta.depth_scale : 0.001;
    depth_u16.convertTo(*depth_out, CV_32FC1, scale);
  } else {
    return false;
  }

  cv::patchNaNs(*depth_out, 0.0);
  cv::threshold(*depth_out, *depth_out, 0.0, 0.0, cv::THRESH_TOZERO);
  return true;
}

std::string ReadFileText(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

void WriteFileText(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write file: " + path.string());
  }
  output << text;
}

void ReplaceAll(std::string* text, const std::string& key, const std::string& value) {
  std::size_t pos = 0;
  while ((pos = text->find(key, pos)) != std::string::npos) {
    text->replace(pos, key.size(), value);
    pos += value.size();
  }
}

std::string ToFloatString(double value) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(8) << value;
  return ss.str();
}

std::filesystem::path MaterializeRuntimeSettings(const std::filesystem::path& template_path,
                                                 const FrameMeta& meta) {
  std::string content = ReadFileText(template_path);
  ReplaceAll(&content, "__WIDTH__", std::to_string(meta.w));
  ReplaceAll(&content, "__HEIGHT__", std::to_string(meta.h));
  ReplaceAll(&content, "__FX__", ToFloatString(meta.fx));
  ReplaceAll(&content, "__FY__", ToFloatString(meta.fy));
  ReplaceAll(&content, "__CX__", ToFloatString(meta.cx));
  ReplaceAll(&content, "__CY__", ToFloatString(meta.cy));
  ReplaceAll(&content, "__CAMERA_RGB__", meta.rgb_encoding == "RGB" ? "1" : "0");

  const auto runtime_path =
      std::filesystem::temp_directory_path() /
      ("orbslam3_rgbd_runtime_" + std::to_string(getpid()) + ".yaml");
  WriteFileText(runtime_path, content);
  return runtime_path;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (!std::filesystem::exists(options.vocab_path)) {
    std::cerr << "[init] missing vocabulary file: " << options.vocab_path << "\n";
    return 1;
  }
  if (!std::filesystem::exists(options.settings_template_path)) {
    std::cerr << "[init] missing settings template: " << options.settings_template_path << "\n";
    return 1;
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  void* zmq_ctx = zmq_ctx_new();
  if (!zmq_ctx) {
    std::cerr << "[init] failed to create ZMQ context\n";
    return 1;
  }

  void* sub = zmq_socket(zmq_ctx, ZMQ_SUB);
  if (!sub) {
    std::cerr << "[init] failed to create ZMQ SUB socket\n";
    zmq_ctx_term(zmq_ctx);
    return 1;
  }

  const int hwm = std::max(1, options.rcv_hwm);
  zmq_setsockopt(sub, ZMQ_RCVHWM, &hwm, sizeof(hwm));
  zmq_setsockopt(sub, ZMQ_SUBSCRIBE, options.topic.data(), options.topic.size());

  int bind_rc = 0;
  if (options.bind) {
    bind_rc = zmq_bind(sub, options.endpoint.c_str());
    std::cout << "[zmq] SUB bind " << options.endpoint << " topic=" << options.topic << "\n";
  } else {
    bind_rc = zmq_connect(sub, options.endpoint.c_str());
    std::cout << "[zmq] SUB connect " << options.endpoint << " topic=" << options.topic << "\n";
  }
  if (bind_rc != 0) {
    std::cerr << "[zmq] endpoint error: " << zmq_strerror(zmq_errno()) << "\n";
    zmq_close(sub);
    zmq_ctx_term(zmq_ctx);
    return 1;
  }

  std::unique_ptr<ORB_SLAM3::System> slam;
  std::filesystem::path runtime_settings;
  bool slam_initialized = false;

  std::size_t frames = 0;
  double t_log = MonotonicNowSec();

  while (g_running) {
    RawPacket raw;
    if (!ReceiveRawPacket(sub, options, &raw)) {
      continue;
    }

    FrameMeta meta;
    if (!ParseHeader(raw.header_json, &meta)) {
      continue;
    }

    cv::Mat rgb;
    cv::Mat depth_m;
    if (!DecodeRgb(raw.rgb_bytes, meta, &rgb)) {
      continue;
    }
    if (!DecodeDepth(raw.depth_bytes, meta, &depth_m)) {
      continue;
    }

    if (!slam_initialized) {
      try {
        runtime_settings = MaterializeRuntimeSettings(options.settings_template_path, meta);
      } catch (const std::exception& exc) {
        std::cerr << "[init] failed to generate runtime settings: " << exc.what() << "\n";
        continue;
      }
      std::cout << "[init] runtime settings: " << runtime_settings << "\n";
      std::cout << "[init] intrinsics fx=" << meta.fx << " fy=" << meta.fy << " cx=" << meta.cx
                << " cy=" << meta.cy << " size=" << meta.w << "x" << meta.h << "\n";

      slam = std::make_unique<ORB_SLAM3::System>(
          options.vocab_path, runtime_settings.string(), ORB_SLAM3::System::RGBD, options.viewer);
      slam_initialized = true;
    }

    slam->TrackRGBD(rgb, depth_m, meta.ts);

    ++frames;
    const double now = MonotonicNowSec();
    if (now - t_log >= 1.0) {
      const double fps = static_cast<double>(frames) / (now - t_log);
      std::cout << "[track] fps=" << std::fixed << std::setprecision(1) << fps << " last_seq=" << meta.seq
                << " rgb=" << rgb.cols << "x" << rgb.rows << "\n";
      frames = 0;
      t_log = now;
    }
  }

  if (slam) {
    slam->Shutdown();
    if (!options.trajectory_path.empty()) {
      slam->SaveTrajectoryTUM(options.trajectory_path);
      std::cout << "[shutdown] trajectory saved: " << options.trajectory_path << "\n";
    }
  }

  zmq_close(sub);
  zmq_ctx_term(zmq_ctx);
  std::cout << "[shutdown] done\n";
  return 0;
}
