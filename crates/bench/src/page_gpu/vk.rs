//! Minimal Vulkan host for `page_raster.slang`: one device, four compute
//! pipelines sharing one descriptor set, persistent buffers grown on
//! demand, and a timestamp query pool around the stages of a page.
//!
//! Memory placement:
//! - command list, intersections, row offsets: host-visible memory the
//!   host writes in place, device-local when such a type exists;
//! - tile counts, offsets and lists: device-local;
//! - page: device-local and host-cached when such a type exists (integrated
//!   GPUs: the host reads the page in place), otherwise device-local with a
//!   copy into a host-cached readback buffer at the end of the submission.
#![expect(
    unsafe_code,
    reason = "Vulkan calls through ash are unsafe; every handle used is created and \
              destroyed by this module"
)]

use std::ffi::{CStr, c_char};
use std::time::Instant;

use ash::vk;

use crate::cmdlist::{CmdList, TILE};

static SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/page_raster.spv"));

const ENTRIES: [&CStr; 5] = [
    c"bin_count",
    c"bin_scan",
    c"bin_scatter",
    c"fine_raster",
    c"fine_raster_rows",
];
const BINDINGS: u32 = 7;
/// Timestamps: start, after binning, after fine raster, after readback.
const STAMPS: u32 = 4;

/// Push constants; layout matches `Params` in `page_raster.slang`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n_cmds: u32,
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    aa_width: u32,
}

type VkResult<T> = Result<T, String>;

fn vkerr(what: &'static str) -> impl Fn(vk::Result) -> String {
    move |e| format!("{what}: {e}")
}

struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ptr: *mut u8,
    coherent: bool,
}

/// Where a buffer lives.
#[derive(Clone, Copy)]
enum Place {
    /// Written by the host in place.
    Upload,
    /// Device only.
    Device,
    /// Written by the device, read by the host in place.
    Readback,
}

/// Host and device times of one page, in nanoseconds.
#[derive(Clone, Copy, Default, Debug)]
pub struct PageTimes {
    /// Host writes of the command list into mapped memory.
    pub upload: u64,
    /// Command-buffer recording and submission.
    pub submit: u64,
    /// Submission to fence signalled, as the host sees it.
    pub wait: u64,
    /// Host copy of the page out of mapped memory.
    pub read: u64,
    /// Device: binning (count, scan, scatter).
    pub gpu_bin: u64,
    /// Device: fine raster and compositing.
    pub gpu_raster: u64,
    /// Device: page copy into the readback buffer (0 when read in place).
    pub gpu_copy: u64,
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pub name: String,
    pub integrated: bool,
    /// Whether the device writes the page where the host reads it.
    pub page_in_place: bool,
    /// Fine raster by `fine_raster_rows` (one scan-table walk per tile
    /// row) instead of `fine_raster` (one per pixel).
    pub raster_rows: bool,
    /// Start-up steps in ns: loader, instance, device, the rest.
    pub startup: [u64; 4],
    mem: vk::PhysicalDeviceMemoryProperties,
    ts_period_ns: f64,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipelines: [vk::Pipeline; 5],
    dpool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    cpool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    queries: vk::QueryPool,
    /// Bindings 0..=6, then the readback buffer when the page is not read
    /// in place.
    bufs: Vec<Option<Buf>>,
}

fn load_entry() -> VkResult<ash::Entry> {
    if let Ok(e) = unsafe { ash::Entry::load() } {
        return Ok(e);
    }
    for path in [
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
    ] {
        if let Ok(e) = unsafe { ash::Entry::load_from(path) } {
            return Ok(e);
        }
    }
    Err("no Vulkan loader".to_owned())
}

fn names(props: &[vk::ExtensionProperties]) -> Vec<&CStr> {
    props
        .iter()
        .filter_map(|p| p.extension_name_as_c_str().ok())
        .collect()
}

/// Rank for the default choice: discrete, integrated, other.
fn type_rank(t: vk::PhysicalDeviceType) -> u8 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 2,
    }
}

impl Gpu {
    /// Open the device `selector` names (an enumeration index, or a
    /// case-insensitive substring of the device name), else the first
    /// discrete GPU, else the first integrated one.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear sequence of object creations"
    )]
    pub fn new(selector: Option<&str>) -> VkResult<Self> {
        let mut startup = [0u64; 4];
        let t = Instant::now();
        let entry = load_entry()?;
        startup[0] = ns(t);
        let t = Instant::now();
        let offered = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(vkerr("instance extensions"))?;
        let portability = names(&offered).contains(&ash::khr::portability_enumeration::NAME);
        let inst_exts: Vec<*const c_char> = if portability {
            vec![ash::khr::portability_enumeration::NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let app = vk::ApplicationInfo::default()
            .application_name(c"page_gpu_bench")
            .api_version(vk::API_VERSION_1_3);
        let flags = if portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&inst_exts)
                    .flags(flags),
                None,
            )
        }
        .map_err(vkerr("vkCreateInstance"))?;
        startup[1] = ns(t);
        let t = Instant::now();

        let physicals =
            unsafe { instance.enumerate_physical_devices() }.map_err(vkerr("enumerate devices"))?;
        let mut cands = Vec::new();
        for (i, &p) in physicals.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(p) };
            let name = props
                .device_name_as_c_str()
                .map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned());
            let families = unsafe { instance.get_physical_device_queue_family_properties(p) };
            let Some(qf) = families.iter().position(|f| {
                f.queue_flags.contains(vk::QueueFlags::COMPUTE) && f.timestamp_valid_bits > 0
            }) else {
                continue;
            };
            cands.push((i, p, props, name, u32::try_from(qf).expect("few families")));
        }
        let chosen = match selector {
            Some(sel) => {
                let lower = sel.to_lowercase();
                cands
                    .iter()
                    .find(|c| sel.parse::<usize>().ok() == Some(c.0))
                    .or_else(|| cands.iter().find(|c| c.3.to_lowercase().contains(&lower)))
            }
            None => cands
                .iter()
                .min_by_key(|c| (type_rank(c.2.device_type), c.0)),
        };
        let Some(&(_, phys, props, ref name, qf)) = chosen else {
            unsafe { instance.destroy_instance(None) };
            return Err(format!("no Vulkan device matches {selector:?}"));
        };
        let name = name.clone();

        let dev_offered = unsafe { instance.enumerate_device_extension_properties(phys) }
            .map_err(vkerr("device extensions"))?;
        let dev_exts: Vec<*const c_char> =
            if names(&dev_offered).contains(&ash::khr::portability_subset::NAME) {
                vec![ash::khr::portability_subset::NAME.as_ptr()]
            } else {
                Vec::new()
            };
        let prio = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf)
            .queue_priorities(&prio)];
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .storage_buffer8_bit_access(true)
            .shader_int8(true);
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12);
        let device = unsafe {
            instance.create_device(
                phys,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&qinfo)
                    .enabled_extension_names(&dev_exts)
                    .push_next(&mut f2),
                None,
            )
        }
        .map_err(vkerr("vkCreateDevice"))?;
        let queue = unsafe { device.get_device_queue(qf, 0) };
        let mem = unsafe { instance.get_physical_device_memory_properties(phys) };
        startup[2] = ns(t);
        let t = Instant::now();

        let words: Vec<u32> = SPIRV
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_le_bytes(b))
            .collect();
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(vkerr("shader module"))?;
        let binds: Vec<_> = (0..BINDINGS)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
                None,
            )
        }
        .map_err(vkerr("set layout"))?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(u32::try_from(size_of::<Params>()).expect("small"))];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&dsl))
                    .push_constant_ranges(&push),
                None,
            )
        }
        .map_err(vkerr("pipeline layout"))?;
        let infos: Vec<_> = ENTRIES
            .iter()
            .map(|e| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(e),
                    )
                    .layout(layout)
            })
            .collect();
        let pipes =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None) }
                .map_err(|(_, e)| format!("compute pipelines: {e}"))?;
        unsafe { device.destroy_shader_module(module, None) };
        let pipelines = [pipes[0], pipes[1], pipes[2], pipes[3], pipes[4]];

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(BINDINGS)];
        let dpool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(vkerr("descriptor pool"))?;
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(dpool)
                    .set_layouts(std::slice::from_ref(&dsl)),
            )
        }
        .map_err(vkerr("descriptor set"))?[0];
        let cpool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(vkerr("command pool"))?;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cpool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(vkerr("command buffer"))?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vkerr("fence"))?;
        let queries = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(STAMPS),
                None,
            )
        }
        .map_err(vkerr("query pool"))?;

        let page_in_place = find_type(
            &mem,
            u32::MAX,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_CACHED,
        )
        .is_some();
        let mut bufs = Vec::new();
        bufs.resize_with(BINDINGS as usize + 1, || None);
        startup[3] = ns(t);
        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            name,
            integrated: props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU,
            page_in_place,
            raster_rows: true,
            startup,
            mem,
            ts_period_ns: f64::from(props.limits.timestamp_period),
            dsl,
            layout,
            pipelines,
            dpool,
            set,
            cpool,
            cmd,
            fence,
            queries,
            bufs,
        })
    }

    fn place_of(&self, slot: usize) -> Place {
        match slot {
            0..=2 => Place::Upload,
            6 if self.page_in_place => Place::Readback,
            7 => Place::Readback,
            _ => Place::Device,
        }
    }

    /// Make buffer `slot` at least `size` bytes; returns whether it was
    /// (re)created.
    fn ensure(&mut self, slot: usize, size: u64) -> VkResult<bool> {
        let size = size.max(16);
        if self.bufs[slot].as_ref().is_some_and(|b| b.size >= size) {
            return Ok(false);
        }
        if let Some(old) = self.bufs[slot].take() {
            self.free(&old);
        }
        let size = size.next_power_of_two();
        let place = self.place_of(slot);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(vkerr("create buffer"))?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let (dl, hv, hc, hcached) = (
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_CACHED,
        );
        let wanted: &[vk::MemoryPropertyFlags] = match place {
            Place::Upload => &[dl | hv | hc, hv | hc],
            Place::Device => &[dl],
            Place::Readback => &[dl | hv | hcached, hv | hcached, hv],
        };
        let ty = wanted
            .iter()
            .find_map(|&f| find_type(&self.mem, req.memory_type_bits, f))
            .ok_or("no memory type")?;
        let flags = self.mem.memory_types[ty as usize].property_flags;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            )
        }
        .map_err(vkerr("allocate memory"))?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.map_err(vkerr("bind"))?;
        let ptr = if flags.contains(hv) {
            unsafe {
                self.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .map_err(vkerr("map"))?
            .cast::<u8>()
        } else {
            std::ptr::null_mut()
        };
        self.bufs[slot] = Some(Buf {
            buffer,
            memory,
            size,
            ptr,
            coherent: flags.contains(hc),
        });
        if slot < BINDINGS as usize {
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(u32::try_from(slot).expect("slot < 8"))
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info);
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }
        Ok(true)
    }

    fn free(&self, b: &Buf) {
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    fn write(&self, slot: usize, data: &[u8]) {
        let b = self.bufs[slot].as_ref().expect("ensured");
        assert!(data.len() as u64 <= b.size && !b.ptr.is_null());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.ptr, data.len()) };
        if !b.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(b.memory)
                .size(vk::WHOLE_SIZE);
            unsafe { self.device.flush_mapped_memory_ranges(&[range]) }.expect("flush");
        }
    }

    /// Render `cl` into `out` (packed RGB rows, `width × height × 3` bytes).
    pub fn render(&mut self, cl: &CmdList, out: &mut [u8]) -> VkResult<PageTimes> {
        let mut t = PageTimes::default();
        let (w, h) = (cl.width, cl.height);
        let (tiles_x, tiles_y) = (w.div_ceil(TILE), h.div_ceil(TILE));
        let n_tiles = u64::from(tiles_x * tiles_y);
        let page_bytes = u64::from(w) * u64::from(h) * 3;
        assert_eq!(out.len() as u64, page_bytes);

        let start = Instant::now();
        let cmd_bytes: &[u8] = bytemuck::cast_slice(&cl.cmds);
        let ix_bytes: &[u8] = bytemuck::cast_slice(&cl.intersects);
        let rs_bytes: &[u8] = bytemuck::cast_slice(&cl.row_starts);
        self.ensure(0, cmd_bytes.len() as u64)?;
        self.ensure(1, ix_bytes.len() as u64)?;
        self.ensure(2, rs_bytes.len() as u64)?;
        self.ensure(3, n_tiles * 4)?;
        self.ensure(4, (n_tiles + 1) * 4)?;
        self.ensure(5, u64::from(cl.tile_refs) * 4)?;
        self.ensure(6, page_bytes)?;
        if !self.page_in_place {
            self.ensure(7, page_bytes)?;
        }
        self.write(0, cmd_bytes);
        self.write(1, ix_bytes);
        self.write(2, rs_bytes);
        t.upload = ns(start);

        let start = Instant::now();
        let params = Params {
            n_cmds: u32::try_from(cl.cmds.len()).expect("few commands"),
            width: w,
            height: h,
            tiles_x,
            tiles_y,
            aa_width: w * 4,
        };
        self.record(&params, page_bytes)?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd));
        unsafe { self.device.queue_submit(self.queue, &[submit], self.fence) }
            .map_err(vkerr("submit"))?;
        t.submit = ns(start);
        let start = Instant::now();
        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(vkerr("wait"))?;
        t.wait = ns(start);
        unsafe { self.device.reset_fences(&[self.fence]) }.map_err(vkerr("reset fence"))?;

        let mut stamps = [0u64; STAMPS as usize];
        unsafe {
            self.device.get_query_pool_results(
                self.queries,
                0,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(vkerr("query results"))?;
        let span = |a: u64, b: u64| {
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "tick deltas of one page fit f64 and u64"
            )]
            {
                (b.saturating_sub(a) as f64 * self.ts_period_ns) as u64
            }
        };
        t.gpu_bin = span(stamps[0], stamps[1]);
        t.gpu_raster = span(stamps[1], stamps[2]);
        t.gpu_copy = span(stamps[2], stamps[3]);

        let start = Instant::now();
        let src = self.bufs[if self.page_in_place { 6 } else { 7 }]
            .as_ref()
            .expect("ensured");
        if !src.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(src.memory)
                .size(vk::WHOLE_SIZE);
            unsafe { self.device.invalidate_mapped_memory_ranges(&[range]) }
                .map_err(vkerr("invalidate"))?;
        }
        unsafe { std::ptr::copy_nonoverlapping(src.ptr, out.as_mut_ptr(), out.len()) };
        t.read = ns(start);
        Ok(t)
    }

    fn record(&self, p: &Params, page_bytes: u64) -> VkResult<()> {
        let d = &self.device;
        let c = self.cmd;
        let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
        let barrier = |src_stage, src, dst_stage, dst| {
            let m = vk::MemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst);
            unsafe {
                d.cmd_pipeline_barrier(
                    c,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[m],
                    &[],
                    &[],
                );
            }
        };
        let rw = vk::AccessFlags::SHADER_WRITE;
        let rd = vk::AccessFlags::SHADER_READ;
        let n_tiles = p.tiles_x * p.tiles_y;
        let pc: [u32; 6] = [
            p.n_cmds, p.width, p.height, p.tiles_x, p.tiles_y, p.aa_width,
        ];
        unsafe {
            d.reset_command_buffer(c, vk::CommandBufferResetFlags::empty())
                .map_err(vkerr("reset cb"))?;
            d.begin_command_buffer(
                c,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vkerr("begin cb"))?;
            d.cmd_reset_query_pool(c, self.queries, 0, STAMPS);
            d.cmd_write_timestamp(c, vk::PipelineStageFlags::TOP_OF_PIPE, self.queries, 0);
            d.cmd_bind_descriptor_sets(
                c,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            d.cmd_push_constants(
                c,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&pc),
            );
            d.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, self.pipelines[0]);
            d.cmd_dispatch(c, n_tiles.div_ceil(64), 1, 1);
            barrier(compute, rw, compute, rd | rw);
            d.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, self.pipelines[1]);
            d.cmd_dispatch(c, 1, 1, 1);
            barrier(compute, rw, compute, rd | rw);
            d.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, self.pipelines[2]);
            d.cmd_dispatch(c, n_tiles.div_ceil(64), 1, 1);
            barrier(compute, rw, compute, rd | rw);
            d.cmd_write_timestamp(c, compute, self.queries, 1);
            let fine = if self.raster_rows { 4 } else { 3 };
            d.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, self.pipelines[fine]);
            d.cmd_dispatch(c, p.tiles_x, p.tiles_y, 1);
            d.cmd_write_timestamp(c, compute, self.queries, 2);
            if self.page_in_place {
                barrier(
                    compute,
                    rw,
                    vk::PipelineStageFlags::HOST,
                    vk::AccessFlags::HOST_READ,
                );
            } else {
                barrier(
                    compute,
                    rw,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                );
                let region = vk::BufferCopy::default().size(page_bytes);
                d.cmd_copy_buffer(
                    c,
                    self.bufs[6].as_ref().expect("ensured").buffer,
                    self.bufs[7].as_ref().expect("ensured").buffer,
                    &[region],
                );
                barrier(
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::HOST,
                    vk::AccessFlags::HOST_READ,
                );
            }
            d.cmd_write_timestamp(c, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.queries, 3);
            d.end_command_buffer(c).map_err(vkerr("end cb"))?;
        }
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for b in std::mem::take(&mut self.bufs).into_iter().flatten() {
                self.free(&b);
            }
            self.device.destroy_query_pool(self.queries, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cpool, None);
            self.device.destroy_descriptor_pool(self.dpool, None);
            for p in self.pipelines {
                self.device.destroy_pipeline(p, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.dsl, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn find_type(
    mem: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem.memory_type_count).find(|&i| {
        bits & (1 << i) != 0 && mem.memory_types[i as usize].property_flags.contains(flags)
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "page steps take well under 584 years"
)]
fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}
