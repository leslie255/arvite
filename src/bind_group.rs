use crate::{context::Context, pipeline::BindableResource};

#[derive(Debug, Clone)]
pub struct BindGroupBuilder<'cx, 'a> {
    pub(crate) context: &'cx Context,
    pub(crate) layout: &'a wgpu::BindGroupLayout,
    pub(crate) entries: Vec<wgpu::BindGroupEntry<'a>>,
}

impl<'cx, 'a> BindGroupBuilder<'cx, 'a> {
    pub fn new(context: &'cx Context, layout: &'a wgpu::BindGroupLayout) -> Self {
        Self {
            context,
            layout,
            entries: Vec::new(),
        }
    }

    pub fn bind_resource(&mut self, binding: u32, resource: &'a impl BindableResource) {
        self.entries.push(wgpu::BindGroupEntry {
            binding,
            resource: resource.as_binding_resource(),
        });
    }

    pub fn build(&self) -> wgpu::BindGroup {
        self.context
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.layout,
                entries: &self.entries,
            })
    }
}
