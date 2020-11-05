#include <map>
#include <memory>
#include <samples/common.hpp>
#include <string>
#include <vector>

#include <inference_engine.hpp>

using namespace InferenceEngine;

int main(int argc, char *argv[]) {

  const std::string input_model =
      "/mnt/vdp_tests/models/internal/int8/AccuracyAwareQuantization/"
      "performance_preset/2021.2.0-1516-166ab89b95e/FP32/caffe/ssd_mobilenet/"
      "ssd_mobilenet_i8.xml";
  const std::string device_name = "CPU";
  const std::map<std::string, std::string> inference_config = {
      {"CPU_THREADS_NUM", "1"}};

  Core ie;

  CNNNetwork network = ie.ReadNetwork(input_model);
  network.setBatchSize(1);

  std::cout << "model loading..." << std::endl;

  ExecutableNetwork executable_network =
      ie.LoadNetwork(network, device_name, inference_config);

  std::cout << "model was loaded." << std::endl;

  return EXIT_SUCCESS;
}
