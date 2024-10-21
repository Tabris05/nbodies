#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#define _CRT_SECURE_NO_WARNINGS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <variant>

class Application {
	public:
		Application();
		~Application();

		void run();

		void onResize(int width, int height);

	private:
		static constexpr size_t m_framesInFlight = 2;
		static constexpr size_t m_localGroupSize = 32;
		static constexpr size_t m_numBodies = m_localGroupSize * 2048;

		struct PushConstants {
			glm::mat4 projMatrix;
			vk::DeviceAddress readBuffer;
			vk::DeviceAddress writeBuffer;
			float deltaTime;
		};

		struct Body {
			glm::vec2 position;
			glm::vec2 velocity;
			float mass;
		};

		struct Buffer {
			vk::Buffer buffer;
			vk::DeviceMemory memory;
			std::variant<std::byte*, vk::DeviceAddress> ptr;
		};

		struct {
			vk::CommandPool cmdPool;
			vk::CommandBuffer cmdBuffer;
			vk::Semaphore acquireSem;
			vk::Semaphore presentSem;
			vk::Fence fence;
		} m_perFrameData[m_framesInFlight];

		GLFWwindow* m_window;

		uint32_t m_queueFamily;
		vk::Instance m_instance;
		vk::PhysicalDevice m_physicalDevice;
		vk::PhysicalDeviceMemoryProperties m_memProps;
		vk::Device m_device;
		vk::DispatchLoaderDynamic m_deviceDispatcher;
		vk::Queue m_queue;
		vk::SurfaceKHR m_surface;
		vk::SwapchainKHR m_swapchain;
		vk::Format m_swapchainFormat;
		std::vector<vk::Image> m_swapchainImages;
		std::vector<vk::ImageView> m_swapchainImageViews;
		vk::PipelineLayout m_pipelineLayout;
		vk::Pipeline m_pipeline;

		Buffer m_pingBuffer;
		Buffer m_pongBuffer;

		int m_width, m_height;
		size_t m_frameIndex = 0;
		bool m_resizePending = false;

		std::vector<const char*> m_layers{
#ifdef _DEBUG
		"VK_LAYER_KHRONOS_validation"
#endif
		};

		bool hasExtension(const std::vector<vk::ExtensionProperties>& properties, const char* extension);

		void createSwapchain(int width, int height);
		void recreateSwapchain();
		void destroySwapchain();

		std::vector<uint32_t> getShaderSource(const char* path);

		uint32_t getMemoryIndex(vk::MemoryPropertyFlags flags, uint32_t mask);
		Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, uint32_t queueFamily, bool staging);
		void destroyBuffer(Buffer buffer);

		void transitionImage(vk::CommandBuffer& cmdBuffer, vk::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::PipelineStageFlags srcMask, vk::PipelineStageFlags dstMask);
};

#endif

