use {
    anyhow::{anyhow, Context, Error, Result},
    ash::{
        extensions::{
            ext::DebugUtils,
            khr::{Surface, Swapchain},
        },
        vk,
        vk::{
            AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference,
            AttachmentStoreOp, BlendFactor, BlendOp, Buffer, BufferCopy, BufferCreateInfo,
            BufferUsageFlags, ClearColorValue, ClearValue, ColorComponentFlags, ColorSpaceKHR,
            CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
            CommandBufferUsageFlags, CommandPool, CommandPoolCreateInfo, ComponentMapping,
            ComponentSwizzle, CompositeAlphaFlagsKHR, CullModeFlags,
            DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
            DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT,
            DebugUtilsMessengerEXT, DeviceMemory, DeviceSize, Extent2D, Fence, FenceCreateFlags,
            FenceCreateInfo, Format, Framebuffer, FramebufferCreateInfo, FrontFace,
            GraphicsPipelineCreateInfo, Image, ImageAspectFlags, ImageLayout,
            ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType,
            IndexType, MemoryAllocateInfo, MemoryMapFlags, MemoryPropertyFlags, MemoryRequirements,
            PhysicalDevice, PhysicalDeviceMemoryProperties, Pipeline, PipelineBindPoint,
            PipelineCache, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
            PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineLayoutCreateInfo,
            PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo,
            PipelineShaderStageCreateInfo, PipelineStageFlags, PipelineVertexInputStateCreateInfo,
            PipelineViewportStateCreateInfo, PolygonMode, PresentInfoKHR, PresentModeKHR,
            PrimitiveTopology, Queue, Rect2D, RenderPass, RenderPassBeginInfo,
            RenderPassCreateInfo, SampleCountFlags, Semaphore, SemaphoreCreateInfo, ShaderModule,
            ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SubmitInfo, SubpassContents,
            SubpassDependency, SubpassDescription, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
            SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR, VertexInputAttributeDescription,
            VertexInputBindingDescription, VertexInputRate, Viewport, SUBPASS_EXTERNAL,
        },
        Device, Entry, Instance,
    },
    ash_window::enumerate_required_extensions,
    core::ffi::c_void,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle},
    std::{cmp, collections::HashSet, ffi::CStr, mem, ptr},
    winit::{
        dpi::{PhysicalSize, Size},
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        platform::run_return::EventLoopExtRunReturn,
        window::{Window, WindowBuilder},
    },
};

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

const MAX_FRAMES: usize = 2;

#[repr(C)]
struct Vertex {
    position: [f32; 3], // offset 0
    color: [f32; 3],    // offset 12
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    presentation_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.presentation_family.is_some()
    }
}

struct QueueHandles {
    graphics_queue: vk::Queue,
    presentation_queue: vk::Queue,
}

struct SurfaceInfo {
    present_modes: Vec<PresentModeKHR>,
    surface_formats: Vec<SurfaceFormatKHR>,
    surface_capabilities: SurfaceCapabilitiesKHR,
}

impl SurfaceInfo {
    fn is_valid(&self) -> bool {
        !self.present_modes.is_empty() && !self.surface_formats.is_empty()
    }

    fn choose_best_color_format(&self) -> SurfaceFormatKHR {
        const DESIRED_FORMAT: Format = Format::R8G8B8A8_UNORM;
        const DESIRED_FORMAT_ALT: Format = Format::B8G8R8A8_UNORM;
        const DESIRED_COLOR_SPACE: ColorSpaceKHR = ColorSpaceKHR::SRGB_NONLINEAR;

        // If the number of formats is 1, and that format's format is undefined, this is
        // a special case that Vulkan uses to indicate that **all** formats are supported.
        if self.surface_formats.len() == 1 && self.surface_formats[0].format == Format::UNDEFINED {
            SurfaceFormatKHR {
                format: DESIRED_FORMAT,
                color_space: DESIRED_COLOR_SPACE,
            }
        } else {
            for format in &self.surface_formats {
                // Early return if we find a format with the desired color space and on of the
                // desired formats.
                if (format.format == DESIRED_FORMAT || format.format == DESIRED_FORMAT_ALT)
                    && format.color_space == DESIRED_COLOR_SPACE
                {
                    return *format;
                }
            }

            // No desired format found, just take the first supported format.
            self.surface_formats[0]
        }
    }

    fn choose_best_present_mode(&self) -> PresentModeKHR {
        // If the desired mode is available we use that, else Vulkan
        // guarantees that FIFO is supported so fall back to that.
        const DESIRED_MODE: PresentModeKHR = PresentModeKHR::MAILBOX;
        if self.present_modes.contains(&DESIRED_MODE) {
            DESIRED_MODE
        } else {
            PresentModeKHR::FIFO
        }
    }

    fn choose_swapchain_extents(&self, window: &Window) -> Extent2D {
        let current_extent = self.surface_capabilities.current_extent;
        if current_extent.width != u32::MAX {
            current_extent
        } else {
            let window_size = window.inner_size();
            let (width, height) = (window_size.width, window_size.height);
            let (min_width, min_height) = (
                self.surface_capabilities.min_image_extent.width,
                self.surface_capabilities.min_image_extent.height,
            );
            let (max_width, max_height) = (
                self.surface_capabilities.max_image_extent.width,
                self.surface_capabilities.max_image_extent.height,
            );
            let width = cmp::min(cmp::max(width, min_width), max_width);
            let height = cmp::min(cmp::max(height, min_height), max_height);
            Extent2D { width, height }
        }
    }
}

fn main() -> Result<()> {
    {
        let shader_mapping = get_compiled_shader_mapping();
        let (event_loop, window) = create_event_loop_and_window()?;
        let entry = Entry::linked();
        let instance = create_vulkan_instance(&entry, window.raw_display_handle())?;
        let device_extensions =
            [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain\0") }];

        let surface = create_surface(&entry, &instance, &window)?;
        let debug_utils = DebugUtils::new(&entry, &instance);
        let messenger = create_debug_utils_messenger(&debug_utils)?;
        let surface_ext = Surface::new(&entry, &instance);
        let (physical_device, queue_family_indices, surface_info) =
            get_physical_device(&instance, &surface_ext, surface, &device_extensions)?;

        let (logical_device, queues) = create_logical_device(
            &instance,
            physical_device,
            &queue_family_indices,
            &device_extensions,
        )?;

        // create the swapchain
        let swapchain_ext = Swapchain::new(&instance, &logical_device);
        let (swapchain, format, swapchain_extent) = create_swapchain(
            &swapchain_ext,
            surface,
            &surface_info,
            &queue_family_indices,
            &window,
        )?;
        let swapchain_images =
            get_swapchain_images(&swapchain_ext, swapchain, format, &logical_device)?;

        // compile vertex and fragment shader modules from code.
        let vertex_shader = create_shader_module(&logical_device, shader_mapping["triangle.vert"])?;
        let fragment_shader =
            create_shader_module(&logical_device, shader_mapping["triangle.frag"])?;

        // create graphics pipeline etc.
        let pipeline_layout = create_pipeline_layout(&logical_device)?;
        let render_pass = create_render_pass(&logical_device, format)?;
        let graphics_pipeline = create_graphics_pipeline(
            vertex_shader,
            fragment_shader,
            swapchain_extent,
            pipeline_layout,
            render_pass,
            &logical_device,
        )?;

        // create framebuffers for swapchain images
        let framebuffers = swapchain_images
            .iter()
            .map(|image| {
                create_framebuffer(
                    &logical_device,
                    render_pass,
                    image.image_view,
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // create a command pool for the graphics queue family that we'll use for draw commands.
        let graphics_command_pool = create_command_pool(
            &logical_device,
            queue_family_indices.graphics_family.unwrap(),
        )?;

        // allocate command buffers from the graphics queue family.
        let command_buffers = allocate_command_buffers(
            &logical_device,
            graphics_command_pool,
            swapchain_images.len() as u32,
        )?;

        // vertex data
        let vertex_data = [
            Vertex {
                position: [-0.4, -0.4, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.4, 0.4, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: [-0.4, 0.4, 0.0],
                color: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [0.4, -0.4, 0.0],
                color: [1.0, 1.0, 0.0],
            },
        ];
        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(
            &instance,
            &logical_device,
            physical_device,
            &vertex_data,
            graphics_command_pool,
            queues.graphics_queue,
        )?;

        // index data
        let index_data = [0, 1, 2, 1, 0, 3];
        let (index_buffer, index_buffer_memory) = create_index_buffer(
            &instance,
            &logical_device,
            physical_device,
            &index_data,
            graphics_command_pool,
            queues.graphics_queue,
        )?;

        // record into the command buffers.
        record_command_buffers(
            &logical_device,
            &command_buffers,
            &framebuffers,
            render_pass,
            swapchain_extent,
            graphics_pipeline,
            vertex_buffer,
            index_buffer,
            index_data.len(),
        )?;

        // create semaphores and fences for the number of frames we're aiming at
        // defined by MAX_FRAMES constant.
        let (
            image_available_semaphores,
            queue_submit_complete_semaphores,
            queue_submit_complete_fences,
        ) = create_synchronization(&logical_device, MAX_FRAMES)?;

        let error_code = run_event_loop(
            event_loop,
            window,
            &logical_device,
            swapchain,
            &image_available_semaphores,
            &queue_submit_complete_semaphores,
            &queue_submit_complete_fences,
            &swapchain_ext,
            queues.graphics_queue,
            queues.presentation_queue,
            &command_buffers,
        );
        cleanup(
            instance,
            logical_device,
            debug_utils,
            messenger,
            surface_ext,
            surface,
            swapchain_ext,
            swapchain,
            swapchain_images,
            &[vertex_shader, fragment_shader],
            pipeline_layout,
            render_pass,
            graphics_pipeline,
            framebuffers,
            graphics_command_pool,
            image_available_semaphores,
            queue_submit_complete_semaphores,
            queue_submit_complete_fences,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
        );
        if error_code == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!(format!(
                "Application exited with error code {error_code}"
            )))
        }
    }
    .context("Error in main.")
}

////////////////////////// Buffers //////////////////////////////
fn create_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: PhysicalDevice,
    usage: BufferUsageFlags,
    memory_property_flags: MemoryPropertyFlags,
    size: DeviceSize,
) -> Result<(Buffer, DeviceMemory)> {
    unsafe {
        // create a buffer handle of the right size and type.
        let buffer = device.create_buffer(
            &BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(SharingMode::EXCLUSIVE),
            None,
        )?;

        // get buffer memory requirements plus the memory properties of our physical device.
        let memory_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);

        // find a valid memory type index to use.
        let memory_type_index = find_valid_memory_type_index(
            memory_properties,
            memory_requirements,
            memory_property_flags,
        )
        .ok_or_else(|| anyhow!("Failed to get a valid memory type for buffer."))?;

        // allocate memory.
        let buffer_memory = device
            .allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index as u32),
                None,
            )
            .context("Failed to allocate buffer memory.")?;

        // bind buffer memory.
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .context("Failed to bind buffer memory to the buffer.")?;

        // return.
        Ok::<_, Error>((buffer, buffer_memory))
    }
    .context("Error when trying to create a buffer of some type.")
}

fn create_staged_buffer<T>(
    instance: &Instance,
    device: &Device,
    physical_device: PhysicalDevice,
    elements: &[T],
    usage: BufferUsageFlags,
    transfer_command_pool: CommandPool,
    transfer_queue: Queue,
) -> Result<(Buffer, DeviceMemory)> {
    // determine the size needed
    let size = (mem::size_of::<T>() * elements.len()) as DeviceSize;

    // create the GPU buffer and CPU buffer
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        usage | BufferUsageFlags::TRANSFER_SRC,
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        size,
    )
    .context("Failed to create staging buffer.")?;
    let (gpu_buffer, gpu_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        usage | BufferUsageFlags::TRANSFER_DST,
        MemoryPropertyFlags::DEVICE_LOCAL,
        size,
    )
    .context("Failed to create GPU buffer.")?;

    // unsafe stuff now
    unsafe {
        // upload data to the staging buffer
        let write_ptr = device
            .map_memory(staging_buffer_memory, 0, size, MemoryMapFlags::empty())
            .context("Failed to map the staging buffer memory.")? as *mut T;
        ptr::copy_nonoverlapping(elements.as_ptr(), write_ptr, elements.len());
        device.unmap_memory(staging_buffer_memory);

        // allocate a (one time submit) command buffer from the pool to do the transfer
        let command_buffer = device
            .allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .command_pool(transfer_command_pool)
                    .level(CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .context("Failed to allocate a staging transfer command buffer.")?[0];

        // record the transfer command from staging to GPU buffer
        device
            .begin_command_buffer(
                command_buffer,
                &CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .context("Failed to begin recording the command buffer.")?;

        // transfer.
        device.cmd_copy_buffer(
            command_buffer,
            staging_buffer,
            gpu_buffer,
            &[*BufferCopy::builder().size(size)],
        );

        // end command buffer recording
        device
            .end_command_buffer(command_buffer)
            .context("Failed to end recording the command buffer.")?;

        // submit the command buffer to the transfer queue
        let command_buffers = [command_buffer];
        let submit_infos = [*SubmitInfo::builder().command_buffers(&command_buffers)];
        device
            .queue_submit(transfer_queue, &submit_infos, Fence::null())
            .context("Failed to submit the command buffer to the queue.")?;
        device
            .queue_wait_idle(transfer_queue)
            .context("Failed to wait for the transfer to finish.")?;

        // free the command buffer
        device.free_command_buffers(transfer_command_pool, &[command_buffer]);

        // free staging buffer
        device.free_memory(staging_buffer_memory, None);
        device.destroy_buffer(staging_buffer, None);
    }

    Ok((gpu_buffer, gpu_buffer_memory))
}

fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: PhysicalDevice,
    vertices: &[Vertex],
    transfer_command_pool: CommandPool,
    transfer_queue: Queue,
) -> Result<(Buffer, DeviceMemory)> {
    create_staged_buffer(
        instance,
        device,
        physical_device,
        vertices,
        BufferUsageFlags::VERTEX_BUFFER,
        transfer_command_pool,
        transfer_queue,
    )
    .context("Failed to create a vertex buffer.")
}

fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: PhysicalDevice,
    indices: &[u16],
    transfer_command_pool: CommandPool,
    transfer_queue: Queue,
) -> Result<(Buffer, DeviceMemory)> {
    create_staged_buffer(
        instance,
        device,
        physical_device,
        indices,
        BufferUsageFlags::INDEX_BUFFER,
        transfer_command_pool,
        transfer_queue,
    )
    .context("Failed to create an index buffer.")
}

fn find_valid_memory_type_index(
    memory_properties: PhysicalDeviceMemoryProperties,
    memory_requirements: MemoryRequirements,
    flags: MemoryPropertyFlags,
) -> Option<usize> {
    memory_properties
        .memory_types
        .into_iter()
        .enumerate()
        .position(|(index, memory_type)| {
            (memory_requirements.memory_type_bits & (1 << index as u32)) != 0
                && memory_type.property_flags.contains(flags)
        })
}

////////////////////////// Drawing ////////////////////////////////////
#[allow(clippy::too_many_arguments)]
fn draw(
    device: &Device,
    swapchain: SwapchainKHR,
    image_available_semaphore: Semaphore,
    queue_submit_complete_semaphore: Semaphore,
    queue_submit_complete_fence: Fence,
    swapchain_ext: &Swapchain,
    graphics_queue: Queue,
    present_queue: Queue,
    command_buffers: &[CommandBuffer],
) -> Result<()> {
    // wait on the fence being ready from the previou submit and reset after proceeding.
    unsafe {
        device
            .wait_for_fences(&[queue_submit_complete_fence], true, u64::MAX)
            .context("Failed to wait for fence while drawing image.")?;
        device
            .reset_fences(&[queue_submit_complete_fence])
            .context("Failed to reset fence while drawing image.")?;

        // acquire next image index.
        let (image_index, _) = swapchain_ext
            .acquire_next_image(
                swapchain,
                u64::MAX,
                image_available_semaphore,
                Fence::null(),
            )
            .context("Failed to acquire next image while drawing.")?;

        // submit correct command buffer to the graphics queue.
        let wait_semaphores = [image_available_semaphore];
        let wait_dst_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [command_buffers[image_index as usize]];
        let signal_semaphores = [queue_submit_complete_semaphore];
        let submit_infos = [*SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)];
        device
            .queue_submit(graphics_queue, &submit_infos, queue_submit_complete_fence)
            .context("Error while submitting command buffer to he queue during rendering.")?;

        // present the image for rendering.
        let wait_semaphores = [queue_submit_complete_semaphore];
        let swapchains = [swapchain];
        let image_indices = [image_index];
        swapchain_ext
            .queue_present(
                present_queue,
                &PresentInfoKHR::builder()
                    .wait_semaphores(&wait_semaphores)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices),
            )
            .context("Error while presenting image to the swapchain.")?;
    }
    Ok(())
}

////////////////////////// Synchronization ////////////////////////////
fn create_synchronization(
    device: &Device,
    amount: usize,
) -> Result<(Vec<Semaphore>, Vec<Semaphore>, Vec<Fence>)> {
    let semaphore_builder = SemaphoreCreateInfo::builder();
    let fence_builder = FenceCreateInfo::builder().flags(FenceCreateFlags::SIGNALED);

    let image_available_semaphores = (0..amount)
        .map(|_| unsafe {
            device
                .create_semaphore(&semaphore_builder, None)
                .context("Failed to create an image available semaphore.")
        })
        .collect::<Result<Vec<_>>>()?;
    let queue_submit_complete_semaphores = (0..amount)
        .map(|_| unsafe {
            device
                .create_semaphore(&semaphore_builder, None)
                .context("Failed to create a queue submit complete semaphore.")
        })
        .collect::<Result<Vec<_>>>()?;
    let queue_submit_complete_fences = (0..amount)
        .map(|_| unsafe {
            device
                .create_fence(&fence_builder, None)
                .context("Failed to create a queue submit complete fence.")
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((
        image_available_semaphores,
        queue_submit_complete_semaphores,
        queue_submit_complete_fences,
    ))
}

////////////////////////// Command Buffers ////////////////////////////
#[allow(clippy::too_many_arguments)]
fn record_command_buffers(
    device: &Device,
    command_buffers: &[CommandBuffer],
    framebuffers: &[Framebuffer],
    render_pass: RenderPass,
    swapchain_extent: Extent2D,
    graphics_pipeline: Pipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: usize,
) -> Result<()> {
    for (command_buffer, framebuffer) in command_buffers.iter().zip(framebuffers) {
        unsafe {
            // begin command buffer
            device
                .begin_command_buffer(*command_buffer, &CommandBufferBeginInfo::builder())
                .context("Failed to begin command buffer.")?;

            // begin render pass
            let clear_values = [ClearValue {
                color: ClearColorValue {
                    float32: [0.6, 0.65, 0.4, 1.0],
                },
            }];
            device.cmd_begin_render_pass(
                *command_buffer,
                &RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(*Rect2D::builder().extent(swapchain_extent))
                    .clear_values(&clear_values),
                SubpassContents::INLINE,
            );

            // bind pipeline
            device.cmd_bind_pipeline(
                *command_buffer,
                PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            // bind the vertex and index buffers
            device.cmd_bind_vertex_buffers(*command_buffer, 0, &[vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(*command_buffer, index_buffer, 0, IndexType::UINT16);

            // draw vertices
            device.cmd_draw_indexed(*command_buffer, index_count.try_into().unwrap(), 1, 0, 0, 0);

            // end render pass
            device.cmd_end_render_pass(*command_buffer);

            // end command buffer
            device
                .end_command_buffer(*command_buffer)
                .context("Failed to end command buffer.")?;
        }
    }
    Ok(())
}

fn allocate_command_buffers(
    device: &Device,
    command_pool: CommandPool,
    buffer_count: u32,
) -> Result<Vec<CommandBuffer>> {
    unsafe {
        device
            .allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(CommandBufferLevel::PRIMARY)
                    .command_buffer_count(buffer_count),
            )
            .context("Failed to allocate command buffers.")
    }
}

fn create_command_pool(device: &Device, queue_family_index: u32) -> Result<CommandPool> {
    unsafe {
        device
            .create_command_pool(
                &CommandPoolCreateInfo::builder().queue_family_index(queue_family_index),
                None,
            )
            .context("Failed to create a command pool.")
    }
}

////////////////////////// Framebuffer ////////////////////////////////
fn create_framebuffer(
    device: &Device,
    render_pass: RenderPass,
    image_view: ImageView,
    swapchain_extent: Extent2D,
) -> Result<Framebuffer> {
    let attachments = [image_view];
    unsafe {
        device
            .create_framebuffer(
                &FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .layers(1),
                None,
            )
            .context("Failed to create a framebuffer.")
    }
}

////////////////////////// Pipeline ///////////////////////////////////
fn create_pipeline_layout(device: &Device) -> Result<PipelineLayout> {
    unsafe {
        device.create_pipeline_layout(
            &PipelineLayoutCreateInfo::builder()
                .set_layouts(&[])
                .push_constant_ranges(&[]),
            None,
        )
    }
    .context("Error trying to create a pipeline layout.")
}

fn create_render_pass(device: &Device, format: Format) -> Result<RenderPass> {
    unsafe {
        let attachment_descriptions = [*AttachmentDescription::builder()
            .format(format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)];
        let attachment_references = [*AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpass_descriptions = [*SubpassDescription::builder()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_references)];
        let subpass_dependencies = [*SubpassDependency::builder()
            .src_subpass(SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(AccessFlags::empty())
            .dst_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE)];
        device.create_render_pass(
            &RenderPassCreateInfo::builder()
                .attachments(&attachment_descriptions)
                .subpasses(&subpass_descriptions)
                .dependencies(&subpass_dependencies),
            None,
        )
    }
    .context("Error trying to create a render pass.")
}

fn create_graphics_pipeline(
    vertex_shader: ShaderModule,
    fragment_shader: ShaderModule,
    swapchain_extents: Extent2D,
    pipeline_layout: PipelineLayout,
    render_pass: RenderPass,
    device: &Device,
) -> Result<Pipeline> {
    let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
    let shader_stages = [
        *PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(name),
        *PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(name),
    ];
    let binding_descriptions = [*VertexInputBindingDescription::builder()
        .stride(mem::size_of::<Vertex>().try_into().unwrap())
        .input_rate(VertexInputRate::VERTEX)];
    let attribute_descriptions = [
        *VertexInputAttributeDescription::builder().format(Format::R32G32B32_SFLOAT),
        *VertexInputAttributeDescription::builder()
            .location(1)
            .format(Format::R32G32B32_SFLOAT)
            .offset(12),
    ];
    let vertex_input = PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);
    let input_assembly = PipelineInputAssemblyStateCreateInfo::builder()
        .topology(PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewports = [Viewport {
        width: swapchain_extents.width as f32,
        height: -(swapchain_extents.height as f32),
        y: swapchain_extents.height as f32,
        max_depth: 1.0,
        ..Viewport::default()
    }];
    let scissors = [Rect2D {
        extent: swapchain_extents,
        ..Rect2D::default()
    }];
    let viewport_state = PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization_state = PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(PolygonMode::FILL)
        .cull_mode(CullModeFlags::BACK)
        .front_face(FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(SampleCountFlags::TYPE_1);
    let color_blend_attachments = [*PipelineColorBlendAttachmentState::builder()
        .blend_enable(true)
        .color_write_mask(ColorComponentFlags::RGBA)
        .src_color_blend_factor(BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(BlendOp::ADD)
        .src_alpha_blend_factor(BlendFactor::ONE)
        .dst_alpha_blend_factor(BlendFactor::ZERO)
        .alpha_blend_op(BlendOp::ADD)];
    let color_blend =
        PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);
    let graphics_pipeline_create_infos = [*GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)];
    unsafe {
        let pipelines = device
            .create_graphics_pipelines(PipelineCache::null(), &graphics_pipeline_create_infos, None)
            .map_err(|(_, e)| e)?;
        Ok(pipelines[0])
    }
}

////////////////////////// Shaders ////////////////////////////////////
fn create_shader_module(device: &Device, code: &[u32]) -> Result<ShaderModule> {
    unsafe { device.create_shader_module(&ShaderModuleCreateInfo::builder().code(code), None) }
        .context("Error while creating a shader module.")
}

////////////////////////// Swapchain //////////////////////////////////
struct SwapchainImage {
    #[allow(unused)]
    image: Image,
    image_view: ImageView,
}

fn create_swapchain(
    swapchain_ext: &Swapchain,
    surface: SurfaceKHR,
    surface_info: &SurfaceInfo,
    queue_family_indices: &QueueFamilyIndices,
    window: &Window,
) -> Result<(SwapchainKHR, Format, Extent2D)> {
    unsafe {
        let min_image_count = cmp::min(
            surface_info.surface_capabilities.max_image_count,
            surface_info.surface_capabilities.min_image_count + 1,
        );
        let best_format = surface_info.choose_best_color_format();
        let swapchain_extent = surface_info.choose_swapchain_extents(window);
        let queue_family_indices = [
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.presentation_family.unwrap(),
        ];
        let is_concurrent = queue_family_indices[0] != queue_family_indices[1];
        let create_info = SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(best_format.format)
            .image_color_space(best_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(surface_info.surface_capabilities.current_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(surface_info.choose_best_present_mode())
            .clipped(true);
        let create_info = if is_concurrent {
            create_info
                .image_sharing_mode(SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices)
        } else {
            create_info.image_sharing_mode(SharingMode::EXCLUSIVE)
        };
        let swapchain = swapchain_ext
            .create_swapchain(&create_info, None)
            .context("Error while creating a swapchain.")?;
        Ok((swapchain, best_format.format, swapchain_extent))
    }
}

fn get_swapchain_images(
    swapchain_ext: &Swapchain,
    swapchain: SwapchainKHR,
    format: Format,
    device: &Device,
) -> Result<Vec<SwapchainImage>> {
    unsafe {
        let swapchain_images = swapchain_ext.get_swapchain_images(swapchain)?;
        let mut swapchain_images_output = Vec::with_capacity(swapchain_images.len());
        for image in swapchain_images {
            let image_view = create_image_view(image, format, ImageAspectFlags::COLOR, device)?;
            swapchain_images_output.push(SwapchainImage { image, image_view })
        }
        Ok::<_, Error>(swapchain_images_output)
    }
    .context("Error while trying to get swapchain images and make image views.")
}

////////////////////////// Image Views ////////////////////////////////
fn create_image_view(
    image: Image,
    format: Format,
    aspect_flags: ImageAspectFlags,
    device: &Device,
) -> Result<ImageView> {
    let component_mapping_builder = ComponentMapping::builder()
        .r(ComponentSwizzle::IDENTITY)
        .g(ComponentSwizzle::IDENTITY)
        .b(ComponentSwizzle::IDENTITY)
        .a(ComponentSwizzle::IDENTITY);
    let subresource_range_builder = ImageSubresourceRange::builder()
        .aspect_mask(aspect_flags)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let builder = ImageViewCreateInfo::builder()
        .image(image)
        .view_type(ImageViewType::TYPE_2D)
        .format(format)
        .components(*component_mapping_builder)
        .subresource_range(*subresource_range_builder);
    unsafe { device.create_image_view(&builder, None) }
        .context("Error while trying to create an image view.")
}

////////////////////////// Debugging //////////////////////////////////
fn create_debug_utils_messenger(debug_utils: &DebugUtils) -> Result<DebugUtilsMessengerEXT> {
    unsafe {
        debug_utils.create_debug_utils_messenger(
            &DebugUtilsMessengerCreateInfoEXT::builder()
                .pfn_user_callback(Some(debug_callback))
                .message_severity(DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(DebugUtilsMessageTypeFlagsEXT::VALIDATION),
            None,
        )
    }
    .context("Error while creating a debug utils messenger")
}

unsafe extern "system" fn debug_callback(
    _severity_flags: DebugUtilsMessageSeverityFlagsEXT,
    _type_flags: DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> u32 {
    let message = CStr::from_ptr((*callback_data).p_message);
    println!("{message:?}");
    0
}

////////////////////////// Windowing //////////////////////////////////

fn create_event_loop_and_window() -> Result<(EventLoop<()>, Window)> {
    let event_loop = EventLoop::new();
    let window = create_window(&event_loop).context("Error in create_event_loop_and_window")?;
    Ok((event_loop, window))
}

fn create_window(event_loop: &EventLoop<()>) -> Result<Window> {
    WindowBuilder::new()
        .with_resizable(false)
        .with_title("Let's Learn Vulkan")
        .with_inner_size(Size::Physical(PhysicalSize {
            width: 800,
            height: 600,
        }))
        .build(event_loop)
        .context("Error in create_window.")
}

#[allow(clippy::too_many_arguments)]
fn run_event_loop(
    mut event_loop: EventLoop<()>,
    window: Window,
    device: &Device,
    swapchain: SwapchainKHR,
    image_available_semaphores: &[Semaphore],
    queue_submit_complete_semaphores: &[Semaphore],
    queue_submit_complete_fences: &[Fence],
    swapchain_ext: &Swapchain,
    graphics_queue: Queue,
    present_queue: Queue,
    command_buffers: &[CommandBuffer],
) -> i32 {
    let mut current_frame = 0;
    let max_frames = image_available_semaphores.len();
    event_loop.run_return(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                draw(
                    device,
                    swapchain,
                    image_available_semaphores[current_frame],
                    queue_submit_complete_semaphores[current_frame],
                    queue_submit_complete_fences[current_frame],
                    swapchain_ext,
                    graphics_queue,
                    present_queue,
                    command_buffers,
                )
                .unwrap();
                current_frame = (current_frame + 1) % max_frames;
            }
            _ => (),
        }
    })
}

////////////////////////// Instance //////////////////////////////////

fn create_vulkan_instance(entry: &Entry, raw_display_handle: RawDisplayHandle) -> Result<Instance> {
    {
        // # Safety
        // This is safe because we're hard-coding the title as a byte string with a null terminator.
        // This hard-coding is inside the unsafe block and so it's impossible for any safe code outside to
        // cause unsoundness.
        let application_name =
            unsafe { CStr::from_bytes_with_nul_unchecked(b"Let's Learn Vulkan\0") };
        let application_info = vk::ApplicationInfo::builder()
            .application_name(application_name)
            .application_version(vk::make_api_version(1, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);
        let mut required_extensions = enumerate_required_extensions(raw_display_handle)?.to_vec();
        required_extensions
            .extend([
                unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_debug_utils\0") }.as_ptr(),
            ]);
        let required_layers = [unsafe {
            CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()
        }];
        // # Safety
        // This call is safe because required_extensions comes from enumerate_required_extensions
        // in the ash-window crate. We assume that this crate behaves correctly and that it returns valid
        // null-terminated UTF-8 strings.
        unsafe {
            validate_required_extensions(&required_extensions, entry)?;
            validate_required_layers(&required_layers, entry)?;
        }
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(&required_extensions)
            .enabled_layer_names(&required_layers);

        // # Safety
        // This is safe because although it's an FFI function, it's from a trusted source, and we've
        // validated that we're calling it correctly and with a valid InstanceCreateInfo struct due
        // to the builder helping us a little with that.
        unsafe { entry.create_instance(&instance_create_info, None) }
    }
    .context("Error in create_vulkan_instance.")
}

/// # Safety
/// This function doesn't validate that the *const i8 pointers
/// inside the required_extensions slice are all valid null-terminated
/// UTF-8 strings. Caller must ensure that these are correctly formatted
/// or retrieved from a verified source such as the Vulkan API itself.
unsafe fn validate_required_extensions(
    required_extensions: &[*const i8],
    entry: &Entry,
) -> Result<()> {
    {
        let instance_extension_properties = entry.enumerate_instance_extension_properties(None)?;
        let required_extensions = required_extensions
            .iter()
            .copied()
            .map(|ptr| CStr::from_ptr(ptr))
            .collect::<Vec<_>>();
        let available_extensions = instance_extension_properties
            .iter()
            .map(|prop| CStr::from_ptr(prop.extension_name.as_ptr()))
            .collect::<HashSet<_>>();
        for required_extension in required_extensions {
            if !available_extensions.contains(required_extension) {
                return Err(anyhow::anyhow!(format!(
                    "Required extension {} is not available",
                    required_extension.to_str()?
                )));
            }
        }
        Ok::<_, Error>(())
    }
    .context("Error in validate_required_extensions.")
}

/// # Safety
/// This function doesn't validate that the *const i8 pointers
/// inside the required_layers slice are all valid null-terminated
/// UTF-8 strings. Caller must ensure that these are correctly formatted
/// or retrieved from a verified source such as the Vulkan API itself.
unsafe fn validate_required_layers(required_layers: &[*const i8], entry: &Entry) -> Result<()> {
    {
        let instance_layer_properties = entry.enumerate_instance_layer_properties()?;
        let required_layers = required_layers
            .iter()
            .copied()
            .map(|ptr| CStr::from_ptr(ptr))
            .collect::<Vec<_>>();
        let available_layers = instance_layer_properties
            .iter()
            .map(|prop| CStr::from_ptr(prop.layer_name.as_ptr()))
            .collect::<HashSet<_>>();
        for required_layer in required_layers {
            if !available_layers.contains(required_layer) {
                return Err(anyhow::anyhow!(format!(
                    "Required layer {} is not available",
                    required_layer.to_str()?
                )));
            }
        }
        Ok::<_, Error>(())
    }
    .context("Error in validate_required_layers.")
}

////////////////////////// Physical Device //////////////////////////////////

fn get_physical_device(
    instance: &Instance,
    surface_ext: &Surface,
    surface: SurfaceKHR,
    required_extensions: &[&CStr],
) -> Result<(vk::PhysicalDevice, QueueFamilyIndices, SurfaceInfo)> {
    {
        // # Safety
        // This is safe because the only way to get an instance is via the Ash API and we would've
        // aborted earlier if one can't be made. The function itself is an FFI function which is why
        // it's marked unsafe but we're confident we're calling it correctly.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        for physical_device in physical_devices {
            if let Some(queue_family_indices) =
                get_queue_family_indices(physical_device, instance, surface_ext, surface)
            {
                let has_required_extensions = validate_required_device_extensions(
                    instance,
                    physical_device,
                    required_extensions,
                )?;
                if has_required_extensions {
                    let surface_info = get_surface_info(surface_ext, physical_device, surface)?;
                    if surface_info.is_valid() {
                        return Ok((physical_device, queue_family_indices, surface_info));
                    }
                }
            }
        }
        Err(anyhow::anyhow!("Could not find a valid PhysicalDevice."))
    }
    .context("Error in get_physical_device.")
}

fn get_queue_family_indices(
    device: vk::PhysicalDevice,
    instance: &Instance,
    surface_ext: &Surface,
    surface: SurfaceKHR,
) -> Option<QueueFamilyIndices> {
    // Safety: This is marked as unsafe because it's an FFI function, but the fact we have an Instance
    // indicates that we're calling it correctly.
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(device) };
    let mut queue_family_indices = QueueFamilyIndices {
        graphics_family: None,
        presentation_family: None,
    };
    for (idx, props) in queue_family_properties.into_iter().enumerate() {
        // if queue family supports graphics then go ahead and record the graphics family.
        if props.queue_count > 0 && props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            queue_family_indices.graphics_family = Some(idx as u32);
        }

        // Safety: This is marked as unsafe because it's an FFI function, but we have ensured that there's
        // a valid device and the queue family passed is one of the queue family indices for this device.
        if let Ok(surface_supported) =
            unsafe { surface_ext.get_physical_device_surface_support(device, idx as u32, surface) }
        {
            if surface_supported {
                queue_family_indices.presentation_family = Some(idx as u32);
            }
        }

        // if both queue families on this device are found, we can use this device.
        if queue_family_indices.is_complete() {
            return Some(queue_family_indices);
        }
    }
    None
}

////////////////////////// Logical Device //////////////////////////////////

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &QueueFamilyIndices,
    required_extensions: &[&CStr],
) -> Result<(Device, QueueHandles)> {
    {
        // get the indices we'll actually process, remove duplicates - means same family has multiple roles.
        let mut indices = vec![
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.presentation_family.unwrap(),
        ];
        indices.dedup();
        let infos = indices
            .into_iter()
            .map(|idx| {
                *vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(idx)
                    .queue_priorities(&[1.0])
            })
            .collect::<Vec<_>>();

        // # Safety
        // This function call is marked as unsafe because it's an FFI function, however we guarantee we
        // are calling it correctly, and are trusting that it does the correct thing.
        let device = unsafe {
            let required_extensions = required_extensions
                .iter()
                .map(|cstr| cstr.as_ptr())
                .collect::<Vec<_>>();
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::builder()
                    .queue_create_infos(&infos)
                    .enabled_extension_names(&required_extensions),
                None,
            )
        }?;

        // # Safety
        // This is safe to call due to the same assumptions as above.
        let queues = QueueHandles {
            graphics_queue: unsafe {
                device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0)
            },
            presentation_queue: unsafe {
                device.get_device_queue(queue_family_indices.presentation_family.unwrap(), 0)
            },
        };

        Ok::<_, Error>((device, queues))
    }
    .context("Error in create_logical_device.")
}

fn validate_required_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    required_extensions: &[&CStr],
) -> Result<bool> {
    {
        unsafe {
            let device_extension_properties =
                instance.enumerate_device_extension_properties(physical_device)?;
            let available_extensions = device_extension_properties
                .iter()
                .map(|prop| CStr::from_ptr(prop.extension_name.as_ptr()))
                .collect::<HashSet<_>>();
            for required_extension in required_extensions {
                if !available_extensions.contains(required_extension) {
                    return Ok::<_, Error>(false);
                }
            }
        }
        Ok::<_, Error>(true)
    }
    .context("Error in validate_required_device_extensions.")
}

////////////////////////// Surface ///////////////////////////////////
fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> Result<SurfaceKHR> {
    unsafe {
        let display_handle = window.raw_display_handle();
        let window_handle = window.raw_window_handle();
        ash_window::create_surface(entry, instance, display_handle, window_handle, None)
            .context("Error while creating a surface to use.")
    }
}

fn get_surface_info(
    surface_ext: &Surface,
    physical_device: vk::PhysicalDevice,
    surface: SurfaceKHR,
) -> Result<SurfaceInfo> {
    {
        let present_modes = unsafe {
            surface_ext.get_physical_device_surface_present_modes(physical_device, surface)
        }?;
        let surface_capabilities = unsafe {
            surface_ext.get_physical_device_surface_capabilities(physical_device, surface)
        }?;
        let surface_formats =
            unsafe { surface_ext.get_physical_device_surface_formats(physical_device, surface) }?;
        Ok::<_, Error>(SurfaceInfo {
            present_modes,
            surface_capabilities,
            surface_formats,
        })
    }
    .context("Error when trying to get surface information for a physical device.")
}

////////////////////////// Clean Up //////////////////////////////////
#[allow(clippy::too_many_arguments)]
fn cleanup(
    instance: Instance,
    device: Device,
    debug_utils: DebugUtils,
    messenger: DebugUtilsMessengerEXT,
    surface_ext: Surface,
    surface: SurfaceKHR,
    swapchain_ext: Swapchain,
    swapchain: SwapchainKHR,
    swapchain_images: Vec<SwapchainImage>,
    shader_modules: &[ShaderModule],
    pipeline_layout: PipelineLayout,
    render_pass: RenderPass,
    graphics_pipeline: Pipeline,
    framebuffers: Vec<Framebuffer>,
    command_pool: CommandPool,
    image_available_semaphores: Vec<Semaphore>,
    queue_submit_complete_semaphores: Vec<Semaphore>,
    queue_submit_complete_fences: Vec<Fence>,
    vertex_buffer: Buffer,
    vertex_buffer_memory: DeviceMemory,
    index_buffer: Buffer,
    index_buffer_memory: DeviceMemory,
) {
    unsafe {
        device.device_wait_idle().unwrap();
        device.free_memory(index_buffer_memory, None);
        device.destroy_buffer(index_buffer, None);
        device.free_memory(vertex_buffer_memory, None);
        device.destroy_buffer(vertex_buffer, None);
        for semaphore in image_available_semaphores {
            device.destroy_semaphore(semaphore, None);
        }
        for semaphore in queue_submit_complete_semaphores {
            device.destroy_semaphore(semaphore, None);
        }
        for fence in queue_submit_complete_fences {
            device.destroy_fence(fence, None);
        }
        device.destroy_command_pool(command_pool, None);
        for framebuffer in framebuffers {
            device.destroy_framebuffer(framebuffer, None);
        }
        for shader_module in shader_modules {
            device.destroy_shader_module(*shader_module, None);
        }
        device.destroy_pipeline(graphics_pipeline, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        for swapchain_image in swapchain_images {
            device.destroy_image_view(swapchain_image.image_view, None);
        }
        swapchain_ext.destroy_swapchain(swapchain, None);
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        surface_ext.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
