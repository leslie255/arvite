use cgmath::Vector2;

use crate::{context::Context, pipeline::BindableResource};

pub(crate) fn extent_2d(size: Vector2<u32>) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size.x,
        height: size.y,
        depth_or_array_layers: 1,
    }
}

/// Abstraction of uncompressed, 2D textures.
/// Does not keep-alive its data.
#[derive(Debug, Clone)]
pub struct Texture2d {
    /// The WGPU handle.
    pub(crate) wgpu_texture: wgpu::Texture,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) size: Vector2<u32>,
}

impl Texture2d {
    /// Creates a texture that is bindable and able to be copied-onto.
    pub fn new(context: &Context, size: Vector2<u32>, format: wgpu::TextureFormat) -> Self {
        Self::with_usage(
            context,
            size,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )
    }

    /// Creates a texture that can be used as a render attachment and is bindable.
    pub fn drawable_bindable(
        context: &Context,
        size: Vector2<u32>,
        format: wgpu::TextureFormat,
    ) -> Self {
        Texture2d::with_usage(
            context,
            size,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        )
    }

    /// Creates a texture that can be used as a render attachment.
    pub fn drawable(context: &Context, size: Vector2<u32>, format: wgpu::TextureFormat) -> Self {
        Texture2d::with_usage(
            context,
            size,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        )
    }

    pub fn with_usage(
        context: &Context,
        size: Vector2<u32>,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        assert!(!format.is_compressed());
        let wgpu_texture = context
            .wgpu_device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: extent_2d(size),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
        Self {
            wgpu_texture,
            format,
            size,
        }
    }

    pub(crate) fn bytes_per_row(&self) -> u32 {
        self.size.x * self.format.block_copy_size(None).unwrap()
    }

    pub fn write(&self, context: &Context, data: &[u8]) {
        context.wgpu_queue.write_texture(
            self.wgpu_texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.bytes_per_row()),
                rows_per_image: None,
            },
            extent_2d(self.size),
        );
    }

    pub fn create_view(&self) -> TextureView2d {
        TextureView2d::new(self)
    }
}

#[derive(Debug)]
pub struct TextureView2d {
    /// The WGPU handle.
    pub(crate) wgpu_texture_view: wgpu::TextureView,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) size: Vector2<u32>,
}

impl TextureView2d {
    /// Use `Texture2d::create_view`.
    pub(crate) fn new(texture: &Texture2d) -> Self {
        let wgpu_texture_view = texture
            .wgpu_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            wgpu_texture_view,
            format: texture.format,
            size: texture.size,
        }
    }
}

impl BindableResource for TextureView2d {
    fn default_visibility(&self) -> wgpu::ShaderStages {
        wgpu::ShaderStages::FRAGMENT
    }

    fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: self.format.sample_type(None, None).unwrap(),
            view_dimension: wgpu::TextureViewDimension::D2,
        }
    }

    fn as_binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::TextureView(&self.wgpu_texture_view)
    }
}

#[derive(Debug, Clone)]
pub struct SamplerBuilder<'cx> {
    pub(crate) context: &'cx Context,
    pub(crate) descriptor: wgpu::SamplerDescriptor<'static>,
}

impl<'cx> SamplerBuilder<'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            context,
            descriptor: wgpu::SamplerDescriptor::default(),
        }
    }

    pub fn build(&self) -> Sampler {
        Sampler::with_descriptor(self.context, &self.descriptor)
    }

    pub fn with_descriptor(mut self, f: impl FnOnce(&mut wgpu::SamplerDescriptor)) -> Self {
        f(&mut self.descriptor);
        self
    }

    /// For convenience sake, this actually sets 3 parameters, `address_mode_{u, v, w}`.
    /// Use `with_descriptor` for finer adjustments.
    pub fn clamp(mut self, address_mode: wgpu::AddressMode) -> Self {
        self.descriptor.address_mode_u = address_mode;
        self.descriptor.address_mode_v = address_mode;
        self.descriptor.address_mode_w = address_mode;
        self
    }

    pub fn mag_filter(mut self, filter: wgpu::FilterMode) -> Self {
        self.descriptor.mag_filter = filter;
        self
    }

    pub fn min_filter(mut self, filter: wgpu::FilterMode) -> Self {
        self.descriptor.min_filter = filter;
        self
    }

    pub fn mipmap_filter(mut self, filter: wgpu::FilterMode) -> Self {
        self.descriptor.mipmap_filter = filter;
        self
    }
}

#[derive(Debug)]
pub struct Sampler {
    /// The WGPU handle.
    pub(crate) wgpu_sampler: wgpu::Sampler,
}

impl Sampler {
    pub fn with_descriptor(context: &Context, descriptor: &wgpu::SamplerDescriptor) -> Self {
        let wgpu_sampler = context.wgpu_device.create_sampler(descriptor);
        Self { wgpu_sampler }
    }
}

impl BindableResource for Sampler {
    fn default_visibility(&self) -> wgpu::ShaderStages {
        wgpu::ShaderStages::FRAGMENT
    }

    fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
    }

    fn as_binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Sampler(&self.wgpu_sampler)
    }
}

/// For creating a layout in the rendering pipeline before a uniform buffer is actually created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhantomTextureView2d {
    pub(crate) format: wgpu::TextureFormat,
}

impl PhantomTextureView2d {
    pub const fn new(format: wgpu::TextureFormat) -> Self {
        Self { format }
    }
}

impl BindableResource for PhantomTextureView2d {
    fn default_visibility(&self) -> wgpu::ShaderStages {
        wgpu::ShaderStages::FRAGMENT
    }

    fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: self.format.sample_type(None, None).unwrap(),
            view_dimension: wgpu::TextureViewDimension::D2,
        }
    }

    fn as_binding_resource(&self) -> wgpu::BindingResource {
        panic!("Phantom 2D texture views cannot be binded in a binding group")
    }
}

/// For creating a layout in the rendering pipeline before a uniform buffer is actually created.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PhantomSampler {
    _private: (),
}

impl PhantomSampler {
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

impl BindableResource for PhantomSampler {
    fn default_visibility(&self) -> wgpu::ShaderStages {
        wgpu::ShaderStages::FRAGMENT
    }

    fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
    }

    fn as_binding_resource(&self) -> wgpu::BindingResource {
        panic!("Phantom samplers cannot be binded in a binding group")
    }
}
