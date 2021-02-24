// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <random>
#include <string>

#include "inference_engine.hpp"
#include <opencv2/opencv.hpp>

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

std::string getErrorMsg(InferenceEngine::StatusCode code) {
  switch (code) {

  case InferenceEngine::StatusCode::OK:
    return std::string("OK");

  case InferenceEngine::StatusCode::GENERAL_ERROR:
    return std::string("GENERAL_ERROR");

  case InferenceEngine::StatusCode::NOT_IMPLEMENTED:
    return std::string("NOT_IMPLEMENTED");

  case InferenceEngine::StatusCode::NETWORK_NOT_LOADED:
    return std::string("NETWORK_NOT_LOADED");

  case InferenceEngine::StatusCode::PARAMETER_MISMATCH:
    return std::string("PARAMETER_MISMATCH");

  case InferenceEngine::StatusCode::NOT_FOUND:
    return std::string("NOT_FOUND");

  case InferenceEngine::StatusCode::OUT_OF_BOUNDS:
    return std::string("OUT_OF_BOUNDS");

  case InferenceEngine::StatusCode::UNEXPECTED:
    return std::string("UNEXPECTED");

  case InferenceEngine::StatusCode::REQUEST_BUSY:
    return std::string("REQUEST_BUSY");

  case InferenceEngine::StatusCode::RESULT_NOT_READY:
    return std::string("RESULT_NOT_READY");

  case InferenceEngine::StatusCode::NOT_ALLOCATED:
    return std::string("NOT_ALLOCATED");

  case InferenceEngine::StatusCode::INFER_NOT_STARTED:
    return std::string("INFER_NOT_STARTED");

  case InferenceEngine::StatusCode::NETWORK_NOT_READ:
    return std::string("NETWORK_NOT_READ");
  }

  return std::string("UNKNOWN_IE_STATUS_CODE");
}

class InferReqWrap {
public:
  using Ptr = std::shared_ptr<InferReqWrap>;

  explicit InferReqWrap(InferenceEngine::ExecutableNetwork &net,
                        std::string input_name)
      : _request(net.CreateInferRequest()), _input_name(input_name) {
    _input_mat = createMat();
  }

  void startAsync() {
    auto blob = createRandomBlob(_input_mat);
    _request.SetBlob(_input_name, blob);
    _request.StartAsync();
  }

  void infer() {
    auto blob = createRandomBlob(_input_mat);
    _request.SetBlob(_input_name, blob);
    _request.Infer();
  }

  void wait() {
    InferenceEngine::StatusCode code =
        _request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    if (code != InferenceEngine::StatusCode::OK &&
        code != InferenceEngine::StatusCode::INFER_NOT_STARTED) {
      throw std::logic_error("Wait");
    }
  }

  InferenceEngine::Blob::Ptr getBlob(const std::string &name) {
    return _request.GetBlob(name);
  }

  cv::Mat createMat() {
    cv::Mat frame(224, 224, CV_8UC3);
    cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    return frame;
  }

  InferenceEngine::Blob::Ptr createRandomBlob(cv::Mat &frame) {
    size_t channels = frame.channels();
    size_t height = frame.size().height;
    size_t width = frame.size().width;

    size_t strideH = frame.step.buf[0];
    size_t strideW = frame.step.buf[1];

    bool is_dense = strideW == channels && strideH == channels * width;

    if (!is_dense)
      THROW_IE_EXCEPTION << "Doesn't support conversion from not dense cv::Mat";

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

    InferenceEngine::Blob::Ptr image_blob =
        InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data);

    return image_blob;
  }

private:
  InferenceEngine::InferRequest _request;
  std::string _input_name;
  cv::Mat _input_mat;
};
