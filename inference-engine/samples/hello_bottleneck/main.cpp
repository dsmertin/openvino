//  Copyright (C) 2020 Intel Corporation
//  SPDX-License-Identifier: Apache-2.0

#include "infer_request_wrap.hpp"

#include <inference_engine.hpp>

#include <map>
#include <string>
#include <vector>

using namespace InferenceEngine;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage : ./hello_bottleneck <model_path> <device> <nireq> "
                 "<optional: timeout (secs)>"
              << std::endl;
    return EXIT_FAILURE;
  }

  /* read network */
  const std::string model_path = argv[1];
  const std::string device = argv[2];
  const size_t batch_size = 1;
  const size_t num_of_requests = std::stoi(argv[3]);

  InferenceEngine::Core core;
  InferenceEngine::CNNNetwork network = core.ReadNetwork(model_path);

  network.setBatchSize(batch_size);

  /* pre-process */
  InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
  std::string input_name = network.getInputsInfo().begin()->first;

  input_info->getPreProcess().setResizeAlgorithm(
      ResizeAlgorithm::RESIZE_BILINEAR);
  input_info->getPreProcess().setColorFormat(ColorFormat::BGR);

  input_info->setLayout(Layout::NCHW);
  input_info->setPrecision(Precision::U8);

  DataPtr output_info = network.getOutputsInfo().begin()->second;
  std::string output_name = network.getOutputsInfo().begin()->first;

  output_info->setPrecision(Precision::FP16);

  /* load network */
  std::map<std::string, std::string> networkConfig;
  ExecutableNetwork executable_network =
      core.LoadNetwork(network, device, networkConfig);

  std::vector<InferReqWrap::Ptr> inferRequests;
  inferRequests.reserve(num_of_requests);

  /* inference */
  for (size_t i = 0; i < num_of_requests; i++) {
    inferRequests.push_back(
        std::make_shared<InferReqWrap>(executable_network, input_name));
  }

  long long currentInference = 0LL;
  long long previousInference = 1LL - num_of_requests;

  using namespace std::chrono;
  size_t seconds_count = (argc == 5) ? std::stoi(argv[4]) : 120;
  seconds working_time{seconds_count};
  std::cout << "Exec-time = " << seconds_count << " sec." << std::endl;

  size_t processed_frames_count = 0;

  std::cout << "Starting..." << std::endl;

  auto start = high_resolution_clock::now();
  while (duration_cast<seconds>(high_resolution_clock::now() - start) <
         working_time) {
    // start new inference
    inferRequests[currentInference]->startAsync();

    // wait the latest inference execution if exists
    if (previousInference >= 0) {
      inferRequests[previousInference]->wait();
      ++processed_frames_count;
    }

    currentInference++;
    if (currentInference >= num_of_requests) {
      currentInference = 0;
    }

    previousInference++;
    if (previousInference >= num_of_requests) {
      previousInference = 0;
    }
  }

  // wait the latest inference executions
  for (size_t notCompletedIndex = 0ULL;
       notCompletedIndex < (num_of_requests - 1); ++notCompletedIndex) {
    if (previousInference >= 0) {
      inferRequests[previousInference]->wait();
      ++processed_frames_count;
    }

    previousInference++;
    if (previousInference >= num_of_requests) {
      previousInference = 0LL;
    }
  }

  auto stop = high_resolution_clock::now();

  uint64_t fps =
      processed_frames_count / duration_cast<seconds>(stop - start).count();

  std::cout << "Result fps: " << fps << std::endl;

  return 0;
}
