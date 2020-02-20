#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>

#include <cstdlib>
#include <cstring>
#include <cstdint>

static std::vector<char>
readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  auto const fileSize = file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

const std::vector<const char*> validationLayers = {
  // TODO: The tutorial uses this: "VK_LAYER_KHRONOS_validation"
  "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#define NDEBUG 1

const bool enableValidationLayers =
#ifdef NDEBUG
  true;
#else
  false;
#endif

static VkResult
CreateDebugUtilsMessengerEXT(VkInstance instance,
  const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
  const VkAllocationCallbacks* pAllocator,
  VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(instance,
      "vkCreateDebugUtilsMessengerEXT");
  if (!func) {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }

  return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
}

static void
DestroyDebugUtilsMessengerEXT(VkInstance instance,
  const VkDebugUtilsMessengerEXT debugMessenger,
  const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(instance,
      "vkDestroyDebugUtilsMessengerEXT");

  func(instance, debugMessenger,  pAllocator);
}

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(WIDTH, HEIGHT, "example",
      nullptr, nullptr);
  }

  void createSwapChain() {
    auto const swapChainSupport = querySwapChainSupport(physicalDevice_);

    auto const surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    auto const presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

    auto const & capabilities = swapChainSupport.capabilities;
    auto const extent = chooseSwapExtent(capabilities);

    auto imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
      imageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto const indices = findQueueFamilies(physicalDevice_);
    uint32_t const queueFamilyIndices[] = {indices.graphicsFamily.value(),
      indices.presentFamily.value()};
    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;
      createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapChain_)
      != VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain.");
    }


    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
    swapChainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount,
      swapChainImages_.data());
    // std::cout << "debug: " << swapChainImages_.size() << std::endl;

    swapChainImageFormat_ = surfaceFormat.format;
    swapChainExtent_ = extent;
  }

  void createImageViews() {
    swapChainImageViews_.resize(swapChainImages_.size());
    for (auto i = 0u; i < swapChainImages_.size(); ++i) {
      VkImageViewCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages_[i];
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat_;

      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      if (vkCreateImageView(device_, &createInfo, nullptr,
        &swapChainImageViews_[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image view");
      }
    }
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createGraphicsPipeline();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window_)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    for (auto imageView : swapChainImageViews_) {
      vkDestroyImageView(device_, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device_, swapChain_, nullptr);
    swapChain_ = {};

    vkDestroyDevice(device_, nullptr);
    device_ = {};

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
      debugMessenger_ = {};
    }

    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    surface_ = {};

    vkDestroyInstance(instance_, nullptr);
    instance_ = {};

    glfwDestroyWindow(window_);
    window_ = {};

    glfwTerminate();
  }

  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error("validation layers requested but not available.");
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "example";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto const extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    createInfo.enabledLayerCount = 0;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (enableValidationLayers) {
      createInfo.enabledLayerCount = validationLayers.size();
      createInfo.ppEnabledLayerNames = validationLayers.data();

      populateDebugMessengerCreateInfo(debugCreateInfo);
      // TODO: Crashes: createInfo.pNext = &debugCreateInfo;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance.");
    }

    /*
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
      nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::cout << "available extensions:" << std::endl;
    for (auto const ext : extensions) {
      std::cout << "\t" << ext.extensionName << std::endl;
    }
    */
  }

  static bool
  checkValidationLayerSupport() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    std::cout << "available layers:" << std::endl;
    for (auto const layer : availableLayers) {
      std::cout << "\t" << layer.layerName << std::endl;
    }

    for (auto const layer : validationLayers) {
      for (const auto & layerProperties : availableLayers) {
        if (strcmp(layer, layerProperties.layerName) == 0) {
          return true;
        }
      }
    }

    return false;
  }

  static std::vector<const char*>
  getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    auto const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers) {
      return;
    }

    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr,
      &debugMessenger_) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger.");
    }
  }

  void createSurface() {
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create surface");
    }
  }

  void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = &debugCallback;
    createInfo.pUserData = nullptr;
  }

  static VKAPI_ATTR VkBool32
  VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /* messageSeverity */,
    VkDebugUtilsMessageTypeFlagsEXT /* messageType */,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /* pUserData */) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE; // Don't abort.
  }

  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
  };

  SwapChainSupportDetails
  querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_,
     &details.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount,
      nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_,
        &formatCount, details.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount,
      nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_,
        &presentModeCount, details.presentModes.data());
    }

    return details;
  }

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    if (availableFormats.empty()) {
      throw std::runtime_error("empty availableFormats");
    }

    for (auto const & availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    return availableFormats[0];
  }

  VkPresentModeKHR chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (auto const & availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return availablePresentMode;
      }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
  }

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
      return capabilities.currentExtent;
    } else {
      VkExtent2D actualExtent = {WIDTH, HEIGHT};
      actualExtent.width =
	std::max(capabilities.minImageExtent.width,
	    std::min(capabilities.maxImageExtent.width,
	      actualExtent.width));
      actualExtent.height =
	std::max(capabilities.minImageExtent.height,
	    std::min(capabilities.maxImageExtent.height,
	      actualExtent.height));
      return actualExtent;
    }
  }

  static bool
  checkDeviceExtensionsSupported(VkPhysicalDevice device) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr,
      &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr,
      &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
      deviceExtensions.end());
    for (auto const & extension : availableExtensions) {
      // std::cout << "Available extension: " << extension.extensionName << std::endl;
      requiredExtensions.erase(extension.extensionName);
    }

    /*
    if (!requiredExtensions.empty()) {
      for (auto const & extension : requiredExtensions) {
        std::cout << "Missing extension: " << extension << std::endl;
      }

    }
    */

    return requiredExtensions.empty();
  }

  bool
  isDeviceSuitable(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    auto const indices = findQueueFamilies(device);

    auto const swapChainSupport = querySwapChainSupport(device);

    return /* deviceProperties.deviceType ==
      VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && */
      deviceFeatures.geometryShader &&
      indices.isComplete() &&
      checkDeviceExtensionsSupported(device) &&
      !swapChainSupport.formats.empty() &&
      !swapChainSupport.presentModes.empty();
  }

  void pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (auto const & device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice_  = device;
        return;
      }
    }

    throw std::runtime_error("failed to find a suitable GPU.");
  }

  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
      return graphicsFamily.has_value() &&
        presentFamily.has_value();;
    }
  };

  QueueFamilyIndices
  findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (auto const queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }

      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
      if (presentSupport) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }

      ++i;
    }

    return indices;
  }

  void createLogicalDevice() {
    auto const indices = findQueueFamilies(physicalDevice_);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
      indices.graphicsFamily.value(), indices.presentFamily.value()};

    float queuePriority = 1.0f;
    for (auto const queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo = {};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;

      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.emplace_back(queueCreateInfo);
    }

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount = queueCreateInfos.size();
    createInfo.pQueueCreateInfos = queueCreateInfos.data();;

    VkPhysicalDeviceFeatures deviceFeatures = {};
    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = deviceExtensions.size();
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount = validationLayers.size();
      createInfo.ppEnabledLayerNames = validationLayers.data();
    }

    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device.");
    }

    vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, indices.presentFamily.value(), 0, &presentQueue_);
  }

  VkShaderModule createShaderModule(std::vector<char> const & code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device_, &createInfo, nullptr,
      &shaderModule) != VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module.");
    }

    return shaderModule;
  }

  void createGraphicsPipeline() {
    auto vertShaderCode = readFile("src/shaders/vert.spv");
    auto fragShaderCode = readFile("src/shaders/frag.spv");

    auto vertShaderModule = createShaderModule(vertShaderCode);
    auto fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] =
     {vertShaderStageInfo, fragShaderStageInfo};
    std::cout << "debug to avoid unused variable: " << std::size(shaderStages) << std::endl;

    vkDestroyShaderModule(device_, fragShaderModule, nullptr);
    vkDestroyShaderModule(device_, vertShaderModule, nullptr);
  }

  GLFWwindow* window_{};

  VkInstance instance_{};
  VkDebugUtilsMessengerEXT debugMessenger_{};
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice device_{};
  VkQueue graphicsQueue_{};
  VkSurfaceKHR surface_{};
  VkQueue presentQueue_{};
  VkSwapchainKHR swapChain_{};
  std::vector<VkImage> swapChainImages_;
  VkFormat swapChainImageFormat_;
  VkExtent2D swapChainExtent_;
  std::vector<VkImageView> swapChainImageViews_;

  static constexpr int WIDTH = 800;
  static constexpr int HEIGHT = 600;
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
