use {
    anyhow::{Context, Error, Result},
    ash::{
        extensions::{ext::DebugUtils, khr::Surface},
        vk,
        vk::{
            ColorSpaceKHR, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
            DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT,
            DebugUtilsMessengerEXT, Extent2D, Format, PresentModeKHR, SurfaceCapabilitiesKHR,
            SurfaceFormatKHR, SurfaceKHR,
        },
        Device, Entry, Instance,
    },
    ash_window::enumerate_required_extensions,
    core::ffi::c_void,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle},
    std::{cmp, collections::HashSet, ffi::CStr},
    winit::{
        dpi::{PhysicalSize, Size},
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        platform::run_return::EventLoopExtRunReturn,
        window::{Window, WindowBuilder},
    },
};

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
    _graphics_queue: vk::Queue,
    _presentation_queue: vk::Queue,
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

    #[allow(unused)]
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

    #[allow(unused)]
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

    #[allow(unused)]
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
        let (event_loop, window) = create_event_loop_and_window()?;
        let entry = Entry::linked();
        let instance = create_vulkan_instance(&entry, window.raw_display_handle())?;
        let device_extensions =
            [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain\0") }];

        // Safety: We must ensure that from the moment of surface creation to the surface being
        // destroyed, that the instance isn't destroyed. The cleanup function is safe and destroys
        // in the correct order, so unsafe block ends there.
        let error_code = unsafe {
            let surface = create_surface(&entry, &instance, &window)?;
            let debug_utils = DebugUtils::new(&entry, &instance);
            let messenger = create_debug_utils_messenger(&debug_utils)?;
            let surface_ext = Surface::new(&entry, &instance);
            #[allow(unused)]
            let (physical_device, queue_family_indices, surface_info) =
                get_physical_device(&instance, &surface_ext, surface, &device_extensions)?;
            let (logical_device, _queues) = create_logical_device(
                &instance,
                physical_device,
                &queue_family_indices,
                &device_extensions,
            )?;
            let error_code = run_event_loop(event_loop, window);
            cleanup(
                instance,
                logical_device,
                debug_utils,
                messenger,
                surface_ext,
                surface,
            );
            error_code
        };
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

fn run_event_loop(mut event_loop: EventLoop<()>, window: Window) -> i32 {
    event_loop.run_return(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
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
            _graphics_queue: unsafe {
                device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0)
            },
            _presentation_queue: unsafe {
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
/// # Safety
/// This function is unsafe because we can't guarantee that the resulting surface doesn't outlive Instance
/// which is a precondition violation - surface should be destroyed before the instance that created it.
unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<SurfaceKHR> {
    let display_handle = window.raw_display_handle();
    let window_handle = window.raw_window_handle();
    ash_window::create_surface(entry, instance, display_handle, window_handle, None)
        .context("Error while creating a surface to use.")
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

fn cleanup(
    instance: Instance,
    device: Device,
    debug_utils: DebugUtils,
    messenger: DebugUtilsMessengerEXT,
    surface_ext: Surface,
    surface: SurfaceKHR,
) {
    unsafe {
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        surface_ext.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
