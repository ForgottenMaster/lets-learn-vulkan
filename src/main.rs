use {
    anyhow::{Context, Error, Result},
    ash::{
        extensions::{ext::DebugUtils, khr::Surface},
        vk,
        vk::{
            DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
            DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT,
            DebugUtilsMessengerEXT, SurfaceKHR,
        },
        Device, Entry, Instance,
    },
    ash_window::enumerate_required_extensions,
    core::ffi::c_void,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle},
    std::{collections::HashSet, ffi::CStr},
    winit::{
        dpi::{PhysicalSize, Size},
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        platform::run_return::EventLoopExtRunReturn,
        window::{Window, WindowBuilder},
    },
};

struct QueueFamilyIndices {
    graphics_family: u32,
}

struct QueueHandles {
    _graphics_queue: vk::Queue,
}

fn main() -> Result<()> {
    {
        let (event_loop, window) = create_event_loop_and_window()?;
        let entry = Entry::linked();
        let instance = create_vulkan_instance(&entry, window.raw_display_handle())?;
        // Safety: We must ensure that from the moment of surface creation to the surface being
        // destroyed, that the instance isn't destroyed. The cleanup function is safe and destroys
        // in the correct order, so unsafe block ends there.
        let error_code = unsafe {
            let surface = create_surface(&entry, &instance, &window)?;
            let debug_utils = DebugUtils::new(&entry, &instance);
            let messenger = create_debug_utils_messenger(&debug_utils)?;
            let (physical_device, queue_family_indices) = get_physical_device(&instance)?;
            let (logical_device, _queues) =
                create_logical_device(&instance, physical_device, &queue_family_indices)?;
            let error_code = run_event_loop(event_loop, window);
            cleanup(
                entry,
                instance,
                logical_device,
                debug_utils,
                messenger,
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

fn get_physical_device(instance: &Instance) -> Result<(vk::PhysicalDevice, QueueFamilyIndices)> {
    {
        // # Safety
        // This is safe because the only way to get an instance is via the Ash API and we would've
        // aborted earlier if one can't be made. The function itself is an FFI function which is why
        // it's marked unsafe but we're confident we're calling it correctly.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        for physical_device in physical_devices {
            if let Some(queue_family_indices) = get_queue_family_indices(physical_device, instance)
            {
                return Ok((physical_device, queue_family_indices));
            }
        }
        Err(anyhow::anyhow!("Could not find a valid PhysicalDevice."))
    }
    .context("Error in get_physical_device.")
}

fn get_queue_family_indices(
    device: vk::PhysicalDevice,
    instance: &Instance,
) -> Option<QueueFamilyIndices> {
    // # Safety
    // This is marked as unsafe because it's an FFI function, but the fact we have an Instance
    // indicates that we're calling it correctly.
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (idx, props) in queue_family_properties.into_iter().enumerate() {
        if props.queue_count > 0 && props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            return Some(QueueFamilyIndices {
                graphics_family: idx as u32,
            });
        }
    }
    None
}

////////////////////////// Logical Device //////////////////////////////////

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &QueueFamilyIndices,
) -> Result<(Device, QueueHandles)> {
    {
        // # Safety
        // This function call is marked as unsafe because it's an FFI function, however we guarantee we
        // are calling it correctly, and are trusting that it does the correct thing.
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::builder().queue_create_infos(&[
                    *vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(queue_family_indices.graphics_family)
                        .queue_priorities(&[1.0]),
                ]),
                None,
            )
        }?;

        // # Safety
        // This is safe to call due to the same assumptions as above.
        let queues = QueueHandles {
            _graphics_queue: unsafe {
                device.get_device_queue(queue_family_indices.graphics_family, 0)
            },
        };

        Ok::<_, Error>((device, queues))
    }
    .context("Error in create_logical_device.")
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

////////////////////////// Clean Up //////////////////////////////////

fn cleanup(
    entry: Entry,
    instance: Instance,
    device: Device,
    debug_utils: DebugUtils,
    messenger: DebugUtilsMessengerEXT,
    surface: SurfaceKHR,
) {
    let surface_ext = Surface::new(&entry, &instance);
    unsafe {
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        surface_ext.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
