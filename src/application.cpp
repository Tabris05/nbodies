#include "application.hpp"
#include <ranges>
#include <fstream>
#include <random>
#include <numbers>

Application::Application() {

	// glfw
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		m_width = mode->width * 3 / 4;
		m_height = mode->height * 3 / 4;

		m_window = glfwCreateWindow(m_width, m_height, "N Bodies", nullptr, nullptr);

		glfwSetWindowUserPointer(m_window, this);
		glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int width, int height) {
			reinterpret_cast<Application*>(glfwGetWindowUserPointer(window))->onResize(width, height);
		});
	}

	// vk::Instance
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::span instanceExtensions(glfwExtensions, glfwExtensionCount);

		vk::ApplicationInfo appInfo({}, {}, {}, {}, vk::ApiVersion13);
		m_instance = vk::createInstance({ {}, &appInfo, m_layers, instanceExtensions });
	}

	// vk::PhysicalDevice and vk::PhysicalDeviceMemoryProperties
	{
		for(auto device : m_instance.enumeratePhysicalDevices()) {
			const auto featuresChain = device.getFeatures2<
				vk::PhysicalDeviceFeatures2,
				vk::PhysicalDeviceVulkan12Features,
				vk::PhysicalDeviceVulkan13Features
			>();

			const auto& features12 = featuresChain.get<vk::PhysicalDeviceVulkan12Features>();
			const auto& features13 = featuresChain.get<vk::PhysicalDeviceVulkan13Features>();

			if(
				features12.bufferDeviceAddress &&
				features12.scalarBlockLayout &&
				features13.dynamicRendering &&
				hasExtension(device.enumerateDeviceExtensionProperties(), vk::EXTMeshShaderExtensionName)
			) {
				m_physicalDevice = device;
				break;
			}
		}

		m_memProps = m_physicalDevice.getMemoryProperties();
	}

	// vk::Device and vk::Queue
	{
		for(auto [idx, queueFamily] : std::views::enumerate(m_physicalDevice.getQueueFamilyProperties())) {
			if(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
				m_queueFamily = idx;
				break;
			}
		}

		float priority = 1.0f;
		vk::DeviceQueueCreateInfo queueInfo({}, m_queueFamily, 1, &priority);
		std::vector deviceExtensions{ vk::KHRSwapchainExtensionName, vk::EXTMeshShaderExtensionName };

		vk::StructureChain<
			vk::DeviceCreateInfo,
			vk::PhysicalDeviceVulkan12Features,
			vk::PhysicalDeviceVulkan13Features,
			vk::PhysicalDeviceMeshShaderFeaturesEXT
		> featuresChain({ {}, queueInfo, {}, deviceExtensions }, {}, {}, {});
		featuresChain.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress = true;
		featuresChain.get<vk::PhysicalDeviceVulkan12Features>().scalarBlockLayout = true;
		featuresChain.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering = true;
		featuresChain.get<vk::PhysicalDeviceVulkan13Features>().maintenance4 = true;
		featuresChain.get<vk::PhysicalDeviceMeshShaderFeaturesEXT>().meshShader = true;

		m_device = m_physicalDevice.createDevice(featuresChain.get<vk::DeviceCreateInfo>());
		m_queue = m_device.getQueue(m_queueFamily, 0);

		m_deviceDispatcher = vk::DispatchLoaderDynamic(m_instance, vkGetInstanceProcAddr, m_device, vkGetDeviceProcAddr);
	}

	// vk::Surface, vk::Swapchain, and Swapchain Image Views
	{
		glfwCreateWindowSurface(m_instance, m_window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&m_surface));
		createSwapchain(m_width, m_height);
	}

	// vk::PipelineLayout, vk::Pipeline
	{
		vk::PushConstantRange constantRange(vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(PushConstants));
		m_pipelineLayout = m_device.createPipelineLayout({ {}, {}, constantRange });

		auto msCode = getShaderSource("shaders/nbodies.mesh.spv");
		auto fsCode = getShaderSource("shaders/nbodies.frag.spv");
		vk::ShaderModule meshShader = m_device.createShaderModule({ {}, msCode });
		vk::ShaderModule fragmentShader = m_device.createShaderModule({ {}, fsCode });
		std::vector<vk::PipelineShaderStageCreateInfo> shaderStages{
			vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eMeshEXT, meshShader, "main"),
			vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragmentShader, "main")
		};

		std::vector<vk::DynamicState> dynamicStates{ vk::DynamicState::eViewport, vk::DynamicState::eScissor };

		vk::PipelineViewportStateCreateInfo viewportInfo({}, 1, {}, 1);
		vk::PipelineRasterizationStateCreateInfo rasterizerInfo;
		vk::PipelineMultisampleStateCreateInfo multisampleInfo;
		vk::PipelineDepthStencilStateCreateInfo depthInfo;
		vk::PipelineColorBlendAttachmentState attachmentState;
		vk::PipelineColorBlendStateCreateInfo colorBlendInfo({}, {}, {}, attachmentState);
		vk::PipelineDynamicStateCreateInfo dynamicInfo({}, dynamicStates);
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineInfo{
			{ {}, shaderStages, {}, {}, {}, &viewportInfo, &rasterizerInfo, &multisampleInfo, &depthInfo, &colorBlendInfo, &dynamicInfo, m_pipelineLayout },
			{ {}, m_swapchainFormat } // swapchain out of date events could invalidate this pipeline if the preferred format changes for some reason, seems unlikely to happen though
		};

		attachmentState
			.setBlendEnable(true)
			.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
			.setDstColorBlendFactor(vk::BlendFactor::eOne)
			.setAlphaBlendOp(vk::BlendOp::eAdd)
			.setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
			.setDstAlphaBlendFactor(vk::BlendFactor::eZero)
			.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
		rasterizerInfo.setLineWidth(1.0f);
		m_pipeline = m_device.createGraphicsPipeline({}, pipelineInfo.get<vk::GraphicsPipelineCreateInfo>()).value;

		m_device.destroyShaderModule(meshShader);
		m_device.destroyShaderModule(fragmentShader);
	}

	// per-frame data (vk::CommandPool, vk::CommandBuffer, vk::Semaphores, vk::Fence)
	{
		for(size_t i = 0; i < m_framesInFlight; i++) {
			m_perFrameData[i].cmdPool = m_device.createCommandPool({ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_queueFamily });
			m_perFrameData[i].cmdBuffer = m_device.allocateCommandBuffers({ m_perFrameData[i].cmdPool, vk::CommandBufferLevel::ePrimary, 1 }).front();
			m_perFrameData[i].acquireSem = m_device.createSemaphore({});
			m_perFrameData[i].presentSem = m_device.createSemaphore({});
			m_perFrameData[i].fence = m_device.createFence({ vk::FenceCreateFlagBits::eSignaled });
		}
	}

	// generate bodies
	{
		Buffer stagingBuffer = createBuffer(m_numBodies * sizeof(Body), vk::BufferUsageFlagBits::eTransferSrc, m_queueFamily, true);
		m_pingBuffer = createBuffer(m_numBodies * sizeof(Body), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress, m_queueFamily, false);
		m_pongBuffer = createBuffer(m_numBodies * sizeof(Body), vk::BufferUsageFlagBits::eShaderDeviceAddress, m_queueFamily, false);

		std::mt19937_64 generator;
		std::uniform_real_distribution<float> angle(0.0f, 2.0f * std::numbers::pi_v<float>);
		std::uniform_real_distribution<float> radius(0.0f, 0.5f);

		//*reinterpret_cast<Body*>(std::get<std::byte*>(stagingBuffer.ptr)) = Body{
		//	glm::vec2(0.0f),
		//	glm::vec2(0.0f),
		//	100000000.0f
		//};
		//
		//for(size_t i = 1; i < m_numBodies; i++) {
		//	float theta = angle(generator);
		//	float r = (1.0f - std::sqrtf(radius(generator))) * 375.0f;
		//
		//	glm::vec2 position(r * std::cosf(theta), r * std::sinf(theta));
		//	glm::vec2 velocity = glm::normalize(glm::vec2(position.y, -position.x)) * 1500000.0f / glm::dot(position, position);
		//
		//	reinterpret_cast<Body*>(std::get<std::byte*>(stagingBuffer.ptr))[i] = Body{
		//		position,
		//		velocity,
		//		10.0f
		//	};
		//}

		std::bernoulli_distribution dist;
		reinterpret_cast<Body*>(std::get<std::byte*>(stagingBuffer.ptr))[0] = Body{
			glm::vec2(-750.0f, 0.0f),
			glm::vec2(15.0f, 15.0f),
			500000000.0f
		};
		
		reinterpret_cast<Body*>(std::get<std::byte*>(stagingBuffer.ptr))[1] = Body{
			glm::vec2(750.0f, 0.0f),
			glm::vec2(-15.0f, -15.0f),
			500000000.0f
		};
		
		for(size_t i = 2; i < m_numBodies; i++) {
			float theta = angle(generator);
			float r = (1.0f - std::sqrtf(radius(generator))) * 500.0f;
		
			glm::vec2 position(r * std::cosf(theta), r * std::sinf(theta));
			glm::vec2 velocity = glm::normalize(glm::vec2(position.y, -position.x)) * 3000000.0f / glm::dot(position, position);
		
			reinterpret_cast<Body*>(std::get<std::byte*>(stagingBuffer.ptr))[i] = Body{
				position + (dist(generator) ? glm::vec2(750.0f, 0.0f) : glm::vec2(-750.0f, 0.0f)),
				velocity,
				10.0f
			};
		}

		m_perFrameData->cmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
		m_perFrameData->cmdBuffer.copyBuffer(stagingBuffer.buffer, m_pingBuffer.buffer, vk::BufferCopy{ 0, 0, m_numBodies * sizeof(Body) });
		m_perFrameData->cmdBuffer.end();

		m_queue.submit(vk::SubmitInfo{ {}, {}, m_perFrameData->cmdBuffer });
		m_queue.waitIdle();

		m_device.resetCommandPool(m_perFrameData->cmdPool);

		m_device.unmapMemory(stagingBuffer.memory);
		destroyBuffer(stagingBuffer);
	}
}

Application::~Application() {
	m_device.waitIdle();

	destroyBuffer(m_pingBuffer);
	destroyBuffer(m_pongBuffer);

	for(size_t i = 0; i < m_framesInFlight; i++) {
		m_device.destroyCommandPool(m_perFrameData[i].cmdPool);
		m_device.destroySemaphore(m_perFrameData[i].acquireSem);
		m_device.destroySemaphore(m_perFrameData[i].presentSem);
		m_device.destroyFence(m_perFrameData[i].fence);
	}

	m_device.destroyPipeline(m_pipeline);
	m_device.destroyPipelineLayout(m_pipelineLayout);

	destroySwapchain();

	m_device.destroy();
	m_instance.destroySurfaceKHR(m_surface);
	m_instance.destroy();

	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Application::run() {

	float lastTime = glfwGetTime();

	while(!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();

		uint32_t imageIndex;
		auto& frameData = m_perFrameData[m_frameIndex];

		m_device.waitForFences(frameData.fence, vk::True, std::numeric_limits<uint64_t>::max());

		auto acquireResult = m_device.acquireNextImageKHR(m_swapchain, std::numeric_limits<uint64_t>::max(), frameData.acquireSem);
		if(acquireResult.result == vk::Result::eErrorOutOfDateKHR) {
			recreateSwapchain();
			continue;
		}
		else {
			imageIndex = acquireResult.value;
		}

		m_device.resetFences(frameData.fence);

		vk::ImageSubresourceRange subresource{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

		m_device.resetCommandPool(frameData.cmdPool);
		frameData.cmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

		frameData.cmdBuffer.setViewport(0, vk::Viewport(0, 0, m_width, m_height, 0, 1));
		frameData.cmdBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, { static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height) }));

		float currentTime = glfwGetTime();

		glm::mat4 projection(0.0f);

		if(m_width > m_height) {
			float aspectRatio = static_cast<float>(m_width) / m_height;
			projection = glm::ortho(-1000.0f * aspectRatio, 1000.0f * aspectRatio, -1000.0f, 1000.0f, 0.0f, 1.0f);
		}
		else {
			float aspectRatio = static_cast<float>(m_height) / m_width;
			projection = glm::ortho(-1000.0f, 1000.0f, -1000.0f * aspectRatio, 1000.0f * aspectRatio, 0.0f, 1.0f);
		}

		vk::ArrayProxy<const PushConstants> pcs{ {
			projection,
			std::get<vk::DeviceAddress>(m_pingOrPong ? m_pingBuffer.ptr : m_pongBuffer.ptr),
			std::get<vk::DeviceAddress>(m_pingOrPong ? m_pongBuffer.ptr : m_pingBuffer.ptr),
			std::min(currentTime - lastTime, 0.01f)
		} };

		lastTime = currentTime;

		frameData.cmdBuffer.pushConstants(m_pipelineLayout, vk::ShaderStageFlagBits::eMeshEXT, 0u, pcs);

		transitionImage(
			frameData.cmdBuffer,
			m_swapchainImages[imageIndex],
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eColorAttachmentOutput
		);

		frameData.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

		vk::RenderingAttachmentInfo colorAttachment(
			m_swapchainImageViews[imageIndex],
			vk::ImageLayout::eColorAttachmentOptimal,
			{}, {}, {},
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eStore,
			{ { 0.0f, 0.0f, 0.0f, 0.0f } }
		);
		frameData.cmdBuffer.beginRendering({ {}, { { 0, 0 }, { static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height) } }, 1, {}, colorAttachment });
		frameData.cmdBuffer.drawMeshTasksEXT(m_numBodies / m_localGroupSize / m_bodiesPerInvocation, 1, 1, m_deviceDispatcher);
		frameData.cmdBuffer.endRendering();
		
		transitionImage(
			frameData.cmdBuffer,
			m_swapchainImages[imageIndex],
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::PipelineStageFlagBits::eBottomOfPipe
		);

		vk::BufferMemoryBarrier barrier;
		barrier
			.setSrcAccessMask(vk::AccessFlagBits::eMemoryWrite)
			.setDstAccessMask(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite)
			.setBuffer(m_pingOrPong ? m_pongBuffer.buffer : m_pingBuffer.buffer)
			.setSize(vk::WholeSize);
		
		frameData.cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eMeshShaderEXT, vk::PipelineStageFlagBits::eMeshShaderEXT, {}, {}, barrier, {});

		frameData.cmdBuffer.end();

		vk::PipelineStageFlags waitMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		m_queue.submit(vk::SubmitInfo{ frameData.acquireSem, waitMask, frameData.cmdBuffer, frameData.presentSem }, frameData.fence);

		vk::Result result = m_queue.presentKHR({ frameData.presentSem, m_swapchain, imageIndex });
		if(result != vk::Result::eSuccess || m_resizePending) {
			recreateSwapchain();
		}

		m_frameIndex = (m_frameIndex + 1) % m_framesInFlight;
		m_pingOrPong ^= 1;
	}
}

void Application::onResize(int width, int height) {
	m_resizePending = true;
}

bool Application::hasExtension(const std::vector<vk::ExtensionProperties>& properties, const char* extension) {
	for(const auto& property : properties) {
		if(strcmp(property.extensionName, extension) == 0) {
			return true;
		}
	}

	return false;
}

void Application::createSwapchain(int width, int height) {
	auto surfaceFormat = m_physicalDevice.getSurfaceFormatsKHR(m_surface).front();
	auto presentMode = vk::PresentModeKHR::eFifo;
	for(auto mode : m_physicalDevice.getSurfacePresentModesKHR(m_surface)) {
		if(mode == vk::PresentModeKHR::eMailbox) {
			presentMode = vk::PresentModeKHR::eMailbox;
			break;
		}
		else if(mode == vk::PresentModeKHR::eImmediate && presentMode != vk::PresentModeKHR::eMailbox) {
			presentMode = vk::PresentModeKHR::eImmediate;
		}
	}

	vk::SwapchainCreateInfoKHR createInfo;
	createInfo
		.setSurface(m_surface)
		.setMinImageCount(3)
		.setImageFormat(surfaceFormat.format)
		.setImageColorSpace(surfaceFormat.colorSpace)
		.setImageExtent({ static_cast<uint32_t>(width), static_cast<uint32_t>(height) })
		.setImageArrayLayers(1)
		.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst)
		.setPresentMode(presentMode);
	m_swapchain = m_device.createSwapchainKHR(createInfo);

	m_swapchainImages = m_device.getSwapchainImagesKHR(m_swapchain);
	for(auto image : m_swapchainImages) {
		m_swapchainImageViews.emplace_back(
			m_device.createImageView({ {}, image, vk::ImageViewType::e2D, surfaceFormat.format, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } })
		);
	}

	m_swapchainFormat = surfaceFormat.format;
}

void Application::recreateSwapchain() {
	glfwGetFramebufferSize(m_window, &m_width, &m_height);
	while(m_width == 0 || m_height == 0) {
		glfwGetFramebufferSize(m_window, &m_width, &m_height);
		glfwWaitEvents();
	}

	m_device.waitIdle();

	destroySwapchain();
	createSwapchain(m_width, m_height);

	m_resizePending = false;
}

void Application::destroySwapchain() {
	for(auto view : m_swapchainImageViews) {
		m_device.destroyImageView(view);
	}
	m_swapchainImageViews.clear();

	m_device.destroySwapchainKHR(m_swapchain);
}

std::vector<uint32_t> Application::getShaderSource(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	std::vector<uint32_t> ret(file.tellg() / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(ret.data()), ret.size() * sizeof(uint32_t));
	return ret;
}

uint32_t Application::getMemoryIndex(vk::MemoryPropertyFlags flags, uint32_t mask) {
	for(uint32_t idx = 0; idx < m_memProps.memoryTypeCount; idx++) {
		if(((1 << idx) & mask) && (m_memProps.memoryTypes[idx].propertyFlags & flags) == flags) {
			return idx;
		}
	}
}

auto Application::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, uint32_t queueFamily, bool staging) -> Buffer {
	Buffer ret;
	ret.buffer = m_device.createBuffer({ {}, size, usage, {}, queueFamily });

	vk::MemoryPropertyFlags flags = staging ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent : vk::MemoryPropertyFlagBits::eDeviceLocal;
	vk::MemoryRequirements mrq = m_device.getBufferMemoryRequirements(ret.buffer);

	if(usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
		vk::StructureChain<vk::MemoryAllocateInfo, vk::MemoryAllocateFlagsInfo> info{
			{ mrq.size, getMemoryIndex(flags, mrq.memoryTypeBits)},
			{ vk::MemoryAllocateFlagBits::eDeviceAddress }
		};
		ret.memory = m_device.allocateMemory(info.get<vk::MemoryAllocateInfo>());
	}
	else {
		ret.memory = m_device.allocateMemory({ mrq.size, getMemoryIndex(flags, mrq.memoryTypeBits) });
	}

	m_device.bindBufferMemory(ret.buffer, ret.memory, 0);

	if(usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
		ret.ptr = m_device.getBufferAddress({ ret.buffer });
	}
	else if(staging) {
		ret.ptr = reinterpret_cast<std::byte*>(m_device.mapMemory(ret.memory, 0, vk::WholeSize));
	}

	return ret;
}

void Application::destroyBuffer(Buffer buffer) {
	m_device.destroyBuffer(buffer.buffer);
	m_device.freeMemory(buffer.memory);
}

void Application::transitionImage(vk::CommandBuffer& cmdBuffer, vk::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::PipelineStageFlags srcMask, vk::PipelineStageFlags dstMask) {
	vk::ImageMemoryBarrier barrier;
	barrier
		.setSrcAccessMask(vk::AccessFlagBits::eMemoryWrite)
		.setDstAccessMask(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite)
		.setOldLayout(oldLayout)
		.setNewLayout(newLayout)
		.setImage(image)
		.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
	cmdBuffer.pipelineBarrier( srcMask, dstMask, {}, {}, {}, barrier);
}