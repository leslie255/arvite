//! Convenience abstraction for buffers.

#![allow(dead_code)]

use std::{marker::PhantomData, num::NonZeroU64, ops::RangeBounds};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use crate::{context::Context, pipeline::BindableResource};

pub trait Vertex: Pod {
    fn attributes() -> &'static [wgpu::VertexAttribute];
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub struct Vertex2d {
    pub position: [f32; 2],
}

impl Vertex2d {
    pub const fn new(position: [f32; 2]) -> Self {
        Self { position }
    }
}

impl Vertex for Vertex2d {
    fn attributes() -> &'static [wgpu::VertexAttribute] {
        &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        }]
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub struct Vertex2dUV {
    pub position: [f32; 2],
    pub texture_uv: [f32; 2],
}

impl Vertex2dUV {
    pub const fn new(position: [f32; 2], texture_uv: [f32; 2]) -> Self {
        Self {
            position,
            texture_uv,
        }
    }
}

impl Vertex for Vertex2dUV {
    fn attributes() -> &'static [wgpu::VertexAttribute] {
        &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: size_of::<[f32; 2]>() as u64,
                shader_location: 1,
            },
        ]
    }
}

#[derive(Debug, Clone)]
pub struct VertexBuffer<V: Vertex> {
    pub(crate) wgpu_buffer: wgpu::Buffer,
    pub(crate) _marker: PhantomData<V>,
}

impl<V: Vertex> VertexBuffer<V> {
    pub fn new_initialized(context: &Context, contents: &[V]) -> Self {
        Self {
            wgpu_buffer: context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(contents),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            _marker: PhantomData,
        }
    }

    pub(crate) fn layouts(&self) -> wgpu::VertexBufferLayout {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<V>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: V::attributes(),
        }
    }

    pub(crate) fn set(
        &self,
        render_pass: &mut wgpu::RenderPass,
        slot: u32,
        range: impl RangeBounds<wgpu::BufferAddress>,
    ) {
        render_pass.set_vertex_buffer(slot, self.wgpu_buffer.slice(range));
    }
}

pub trait Index: Pod {
    fn format() -> wgpu::IndexFormat;
}

impl Index for u32 {
    fn format() -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint32
    }
}

impl Index for u16 {
    fn format() -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint16
    }
}

#[derive(Debug, Clone)]
pub struct IndexBuffer<I: Index> {
    pub(crate) wgpu_buffer: wgpu::Buffer,
    pub(crate) _marker: PhantomData<I>,
}

impl<I: Index> IndexBuffer<I> {
    pub fn new_initialized(context: &Context, contents: &[I]) -> Self {
        Self {
            wgpu_buffer: context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(contents),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            _marker: PhantomData,
        }
    }

    pub(crate) fn set(
        &self,
        render_pass: &mut wgpu::RenderPass,
        range: impl RangeBounds<wgpu::BufferAddress>,
    ) {
        render_pass.set_index_buffer(self.wgpu_buffer.slice(range), I::format());
    }
}

#[derive(Debug, Clone)]
pub struct UniformBuffer<T: Pod> {
    pub(crate) wgpu_buffer: wgpu::Buffer,
    pub(crate) _marker: PhantomData<T>,
}

impl<T: Pod> UniformBuffer<T> {
    pub fn new_initialized(context: &Context, value: T) -> Self {
        Self {
            wgpu_buffer: context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::cast_slice(&[value]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
            _marker: PhantomData,
        }
    }

    pub fn new_zeroed(context: &Context) -> Self {
        Self::new_initialized(context, T::zeroed())
    }

    /// Write parts of the buffer.
    pub fn write_parts(&self, context: &Context, offset: wgpu::BufferAddress, data: &[u8]) {
        context.wgpu_queue.write_buffer(&self.wgpu_buffer, offset, data);
    }

    pub fn write(&self, context: &Context, value: T) {
        self.write_parts(context, 0, bytemuck::cast_slice(&[value]));
    }
}

impl<T: Pod> BindableResource for UniformBuffer<T> {
    fn default_visibility(&self) -> wgpu::ShaderStages {
        wgpu::ShaderStages::VERTEX_FRAGMENT
    }

    fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(NonZeroU64::new(size_of::<T>() as u64).unwrap()),
        }
    }

    fn as_binding_resource(&self) -> wgpu::BindingResource {
        self.wgpu_buffer.as_entire_binding()
    }
}

/// For creating a layout in the rendering pipeline before a uniform buffer is actually created.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PhantomUniformBuffer<T: Pod> {
    _marker: PhantomData<T>,
}

impl<T: Pod> PhantomUniformBuffer<T> {
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<T: Pod> BindableResource for PhantomUniformBuffer<T> {
    fn default_visibility(&self) -> wgpu::ShaderStages {
        wgpu::ShaderStages::VERTEX_FRAGMENT
    }

    fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(NonZeroU64::new(size_of::<T>() as u64).unwrap()),
        }
    }

    fn as_binding_resource(&self) -> wgpu::BindingResource {
        panic!("Phantom uniform buffers cannot be binded in a binding group")
    }
}
