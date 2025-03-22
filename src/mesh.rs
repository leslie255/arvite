//! Utilities for drawing.

use std::{
    fmt::{self, Debug},
    ops::Add,
};

use glium::Surface as _;

use cgmath::*;

use crate::{context::Context, utils::MainThreadOnly};

pub fn matrix4_to_array<T>(matrix: Matrix4<T>) -> [[T; 4]; 4] {
    matrix.into()
}

pub fn matrix3_to_array<T>(matrix: Matrix3<T>) -> [[T; 3]; 3] {
    matrix.into()
}

pub fn texture_sampler(texture: &glium::Texture2d) -> glium::uniforms::Sampler<glium::Texture2d> {
    texture
        .sampled()
        .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
        .minify_filter(glium::uniforms::MinifySamplerFilter::Nearest)
        .wrap_function(glium::uniforms::SamplerWrapFunction::Repeat)
}

pub fn texture_sampler_linear(texture: &glium::Texture2d) -> glium::uniforms::Sampler<glium::Texture2d> {
    texture
        .sampled()
        .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
        .minify_filter(glium::uniforms::MinifySamplerFilter::Linear)
        .wrap_function(glium::uniforms::SamplerWrapFunction::Repeat)
}

pub fn default_3d_draw_parameters() -> glium::DrawParameters<'static> {
    glium::DrawParameters {
        depth: glium::Depth {
            test: glium::DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        backface_culling: glium::BackfaceCullingMode::CullClockwise,
        ..Default::default()
    }
}

pub fn default_2d_draw_parameters() -> glium::DrawParameters<'static> {
    glium::DrawParameters {
        depth: glium::Depth {
            test: glium::DepthTest::Overwrite,
            write: false,
            ..Default::default()
        },
        backface_culling: glium::BackfaceCullingMode::CullCounterClockwise,
        blend: glium::draw_parameters::Blend::alpha_blending(),
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad2 {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
}

impl Quad2 {
    pub fn width(self) -> f32 {
        (self.right - self.left).abs()
    }

    pub fn height(self) -> f32 {
        (self.top - self.bottom).abs()
    }
}

/// A mesh that is `Send` and `Sync`.
/// However only the main thread can make draw calls, this is made sure with `MainThreadOnly`.
pub struct Mesh<'cx, V: Copy + glium::Vertex, I: Copy + glium::index::Index = u32> {
    context: &'cx Context,
    vertex_buffer: MainThreadOnly<Option<Box<glium::VertexBuffer<V>>>>,
    index_buffer: MainThreadOnly<Option<Box<glium::IndexBuffer<I>>>>,
    vertices: Vec<V>,
    indices: Vec<I>,
    /// Whether the GL vertex and index buffer reflects the up-to-date content of the vertices/indices.
    /// `true` for not updated.
    needs_update: bool,
    primitive_type: glium::index::PrimitiveType,
}

impl<V: Copy + glium::Vertex + Debug, I: Copy + glium::index::Index + Debug> Debug
    for Mesh<'_, V, I>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SharedMesh")
            .field("vertices", &self.vertices)
            .field("indices", &self.indices)
            .field("needs_update", &self.needs_update)
            .finish_non_exhaustive()
    }
}

impl<'cx, V: Copy + glium::Vertex, I: Copy + glium::index::Index> Mesh<'cx, V, I> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            context,
            vertex_buffer: MainThreadOnly::new(None),
            index_buffer: MainThreadOnly::new(None),
            vertices: Vec::new(),
            indices: Vec::new(),
            primitive_type: glium::index::PrimitiveType::TrianglesList,
            needs_update: false,
        }
    }

    #[track_caller]
    pub fn vertices(&self) -> &[V] {
        &self.vertices
    }

    #[track_caller]
    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    pub fn vertices_indices_mut(&mut self) -> (&mut Vec<V>, &mut Vec<I>) {
        self.needs_update = true;
        (&mut self.vertices, &mut self.indices)
    }

    pub fn vertices_mut(&mut self) -> &mut Vec<V> {
        self.vertices_indices_mut().0
    }

    pub fn indices_mut(&mut self) -> &mut Vec<I> {
        self.vertices_indices_mut().1
    }

    pub fn append(&mut self, vertices: &[V], indices: &[I])
    where
        I: Add<u32, Output = I>,
    {
        let old_length = self.vertices().len() as u32;
        let (self_vertices, self_indices) = self.vertices_indices_mut();
        self_vertices.extend_from_slice(vertices);
        self_indices.extend(indices.iter().map(|&i| i + old_length));
    }

    /// Must be called on main thread only.
    pub fn update_if_needed(&mut self) {
        if !self.needs_update {
            return;
        }
        self.needs_update = false;
        let primitive_type = self.primitive_type();
        let vertex_buffer = self.vertex_buffer.get_mut();
        let index_buffer = self.index_buffer.get_mut();
        if self.vertices.is_empty() || self.indices.is_empty() {
            *vertex_buffer = None;
            *index_buffer = None;
            return;
        }
        *vertex_buffer = Some(Box::new(
            glium::VertexBuffer::dynamic(&self.context.display, &self.vertices).unwrap(),
        ));
        *index_buffer = Some(Box::new(
            glium::IndexBuffer::dynamic(&self.context.display, primitive_type, &self.indices)
                .unwrap(),
        ));
    }

    /// Must be called on main thread only.
    pub fn draw(
        &self,
        frame: &mut glium::Frame,
        uniforms: impl glium::uniforms::Uniforms,
        shader: &glium::Program,
        draw_parameters: &glium::DrawParameters,
    ) {
        let Some(vertex_buffer) = self.vertex_buffer.get() else {
            return;
        };
        let Some(index_buffer) = self.index_buffer.get() else {
            return;
        };
        frame
            .draw(
                vertex_buffer.as_ref(),
                index_buffer.as_ref(),
                shader,
                &uniforms,
                draw_parameters,
            )
            .unwrap();
    }

    pub fn context(&self) -> &'cx Context {
        self.context
    }

    pub fn primitive_type(&self) -> glium::index::PrimitiveType {
        self.primitive_type
    }

    pub fn primitive_type_mut(&mut self) -> &mut glium::index::PrimitiveType {
        self.needs_update = true;
        &mut self.primitive_type
    }
}
