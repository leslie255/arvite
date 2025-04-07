use crate::{
    buffer::{Vertex, VertexBuffer},
    context::Context,
};

#[macro_export]
macro_rules! include_wgsl {
    ($context:expr, $path:literal $(,)?) => {
        $crate::pipeline::wgsl_shader($context, include_str!($path))
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgslShaderSource<'a>(pub &'a str);

impl WgslShaderSource<'_> {
    pub fn compile(self, context: &Context) -> wgpu::ShaderModule {
        let source = self.0;
        context
            .wgpu_device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }
}

#[macro_export]
macro_rules! wgsl {
    ($($tts:tt)*) => {
        $crate::pipeline::WgslShaderSource(stringify!($($tts)*))
    };
}

pub fn wgsl_shader(context: &Context, source: &str) -> wgpu::ShaderModule {
    context
        .wgpu_device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
}

pub trait BindableResource {
    fn default_visibility(&self) -> wgpu::ShaderStages;
    fn binding_type(&self) -> wgpu::BindingType;
    fn as_binding_resource(&self) -> wgpu::BindingResource;
}

#[derive(Debug)]
pub struct PipelineBuilder<'cx, 'a> {
    pub(crate) context: &'cx Context,
    pub(crate) shader: &'a wgpu::ShaderModule,
    pub(crate) bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub(crate) vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout<'a>>,
    pub(crate) label: Option<&'a str>,
    pub(crate) texture_format: Option<wgpu::TextureFormat>,
}

impl<'cx, 'a> PipelineBuilder<'cx, 'a> {
    pub fn new(context: &'cx Context, shader: &'a wgpu::ShaderModule) -> Self {
        Self {
            context,
            shader,
            bind_group_layout_entries: Vec::new(),
            vertex_buffer_layouts: Vec::new(),
            label: None,
            texture_format: None,
        }
    }

    pub fn label(&mut self, label: &'a str) {
        self.label = Some(label);
    }

    pub fn texture_format(&mut self, texture_format: wgpu::TextureFormat) {
        self.texture_format = Some(texture_format);
    }

    pub fn bind_resource(&mut self, binding: u32, item: &impl BindableResource) {
        self.bind_group_layout_entries
            .push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: item.default_visibility(),
                ty: item.binding_type(),
                count: None,
            });
    }

    pub fn add_vertex_buffer<V: Vertex>(&mut self, vertex_buffer: &'a VertexBuffer<V>) {
        self.vertex_buffer_layouts.push(vertex_buffer.layouts());
    }

    pub fn build(&self) -> wgpu::RenderPipeline {
        let bind_group_layout =
            self.context
                .wgpu_device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &self.bind_group_layout_entries,
                });

        let fragment_targets: &[Option<wgpu::ColorTargetState>] = &[Some(wgpu::ColorTargetState {
            format: self
                .texture_format
                .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb),
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let pipeline_layout =
            self.context
                .wgpu_device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let descriptor = wgpu::RenderPipelineDescriptor {
            label: self.label,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: self.shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &self.vertex_buffer_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: self.shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: fragment_targets,
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        };

        self.context.wgpu_device.create_render_pipeline(&descriptor)
    }
}
