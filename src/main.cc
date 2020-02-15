#include <vulkan/vulkan.h>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <cstdlib>

class HelloTriangleApplication {
public:
  void run() {
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initVulkan() {
  }

  void mainLoop() {
  }

  void cleanup() {
  }


};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (std::exception const & ex) {
    std::cerr << ex.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
